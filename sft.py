import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import SFTConfig, SFTTrainer, clone_chat_template

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_id: str = field(default="HuggingFaceTB/SmolLM2-135M", metadata={"help": "Hugging Face 模型 ID 或本地路径"})
    dataset_path: str = field(default="HuggingFaceTB/smoltalk", metadata={"help": "数据集名称"})
    dataset_name: str = field(default="all", metadata={"help": "数据集配置子集"})
    streaming: bool = field(default=False, metadata={"help": "是否开启流式加载数据集"})
    chat_template_model: Optional[str] = field(default=None, metadata={"help": "用于 clone_chat_template 的参考模型 ID，如果为空则使用当前模型 ID"})
    verbose: bool = field(default=False, metadata={"help": "是否打印调试样本和 tokenization 信息"})
    # LoRA
    use_lora: bool = field(default=False, metadata={"help": "是否使用 LoRA"})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha，通常设为 2×lora_r"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(default="q_proj,v_proj", metadata={"help": "逗号分隔的目标模块名，如 q_proj,v_proj,k_proj,o_proj"})

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, SFTConfig))
    model_args, sft_config = parser.parse_args_into_dataclasses()

    # 2. 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # 3. 设置随机种子与分布式环境
    set_seed(sft_config.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # cuDNN SDP 在部分驱动版本下有数值问题，关闭以保证稳定性
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    if sft_config.local_rank <= 0 and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} (Compute {cap[0]}.{cap[1]})")

    # 在分布式模式下，每个进程只看到一张特定的卡
    if local_rank != -1:
        device_map = {"": local_rank}
    elif torch.cuda.is_available() or torch.backends.mps.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"

    # 4. 加载模型与分词器
    torch_dtype = torch.bfloat16 if sft_config.bf16 else (torch.float16 if sft_config.fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 统一设置聊天模板（必须在 LoRA 之前：若新增 token 需 resize embedding，新行应在可训练的 base model 上）
    template_model = model_args.chat_template_model or model_args.model_id
    model, tokenizer, added_tokens = clone_chat_template(model, tokenizer, template_model)

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    if sft_config.local_rank <= 0:
        logger.info(f"Chat template: {tokenizer.chat_template}")
        if added_tokens:
            logger.info(f"新增特殊 token: {[repr(t) for t in added_tokens]}")
        logger.info(f"Tokenizer 词汇表大小: {len(tokenizer)}")

    # 5. 参数量统计 (仅在主进程打印)
    if sft_config.local_rank <= 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("="*40)
        logger.info(f"模型加载成功: {model_args.model_id}")
        logger.info(f"总参数量: {total_params/1e6:.2f}M")
        logger.info(f"可训练参数量: {trainable_params/1e6:.2f}M  ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"精度模式: {torch_dtype}  LoRA: {model_args.use_lora}")
        logger.info("="*40)

    # 6. 加载数据
    raw_datasets = load_dataset(
        model_args.dataset_path,
        model_args.dataset_name,
        streaming=model_args.streaming,
    )

    train_dataset = raw_datasets["train"] if isinstance(raw_datasets, dict) and "train" in raw_datasets else raw_datasets

    if model_args.streaming:
        if sft_config.max_steps <= 0:
            raise ValueError("使用 streaming=True 时，必须设置 max_steps (例如 --max_steps 1000)")
        eval_dataset = None
    else:
        eval_dataset = None
        if isinstance(raw_datasets, dict):
            eval_dataset = raw_datasets.get("test") or raw_datasets.get("validation")

    # 7. 调试：打印第一条样本的格式（仅 verbose 模式）
    if model_args.verbose and sft_config.local_rank <= 0:
        sample = next(iter(train_dataset))
        if hasattr(sft_config, "dataset_text_field") and sft_config.dataset_text_field and isinstance(sample, dict) and sft_config.dataset_text_field in sample:
            formatted = sample[sft_config.dataset_text_field]
            debug_label = f"Raw text from dataset_text_field={sft_config.dataset_text_field}"
        elif isinstance(sample, dict) and "messages" in sample:
            formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)
            debug_label = "After apply_chat_template"
        elif isinstance(sample, dict) and "text" in sample:
            formatted = sample["text"]
            debug_label = "Raw text from sample['text']"
        elif isinstance(sample, str):
            formatted = sample
            debug_label = "Raw string sample"
        else:
            formatted = str(sample)
            debug_label = "Fallback str(sample)"

        print(f"=== {debug_label} ===")
        print(formatted)
        print("-" * 80)
        tokenized = tokenizer(formatted, return_tensors="pt")
        print("=== Tokenized input_ids (decode back) ===")
        print(tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=False))
        print("\n=== input_ids (raw, first 100) ===")
        print(tokenized["input_ids"][0].tolist()[:100])

    # 8. 初始化 Trainer
    # assistant_only_loss 由 SFTConfig 控制（--assistant_only_loss True），TRL 1.x 内置支持
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 9. 执行训练
    trainer.train(resume_from_checkpoint=sft_config.resume_from_checkpoint)

    # 10. 保存最终结果
    if trainer.is_world_process_zero():
        trainer.save_model(sft_config.output_dir)
        tokenizer.save_pretrained(sft_config.output_dir)
        if model_args.use_lora:
            logger.info(f"训练完成，LoRA adapter 已保存至: {sft_config.output_dir}")
        else:
            logger.info(f"训练完成，模型已保存至: {sft_config.output_dir}")

if __name__ == "__main__":
    main()
