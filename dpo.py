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
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    # 基础
    model_name_or_path: str = field(metadata={"help": "模型路径或 HF ID"})
    dataset_path: str = field(metadata={"help": "数据集路径或 HF ID"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "数据集子集名称"})
    dataset_split: str = field(default="train", metadata={"help": "数据集 split"})
    streaming: bool = field(default=False, metadata={"help": "是否流式加载数据集"})
    prompt_column: str = field(default="prompt", metadata={"help": "prompt 列名"})
    chosen_column: str = field(default="chosen", metadata={"help": "chosen 列名"})
    rejected_column: str = field(default="rejected", metadata={"help": "rejected 列名"})
    # LoRA（可选）
    use_lora: bool = field(default=False, metadata={"help": "是否使用 LoRA"})
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj")


def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    args, dpo_config = parser.parse_args_into_dataclasses()

    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = dpo_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    set_seed(dpo_config.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # device_map
    if local_rank != -1:
        device_map = {"": local_rank}
    elif torch.cuda.is_available():
        device_map = "auto"
    elif torch.backends.mps.is_available():
        # MPS 不支持 device_map="auto"，直接用 mps
        device_map = {"": "mps"}
    else:
        device_map = "cpu"

    torch_dtype = (
        torch.bfloat16 if dpo_config.bf16 else
        torch.float16 if dpo_config.fp16 else
        torch.float32
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载 policy model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("全参数 DPO 训练")

    # 加载 reference model（frozen）
    # use_lora 时 ref_model=None，DPOTrainer 自动从 policy 的 base 推断
    ref_model = None
    if not args.use_lora:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # 加载数据集
    load_kwargs = dict(streaming=args.streaming)
    if args.dataset_name:
        load_kwargs["name"] = args.dataset_name

    raw = load_dataset(args.dataset_path, split=args.dataset_split, **load_kwargs)

    if args.streaming and dpo_config.max_steps <= 0:
        raise ValueError("streaming=True 时必须设置 --max_steps")

    # 格式化为 DPOTrainer 期望的 prompt/chosen/rejected 字符串
    def format_chat(example):
        return {
            "prompt": tokenizer.apply_chat_template(
                example[args.prompt_column], tokenize=False, add_generation_prompt=True
            ),
            "chosen": tokenizer.apply_chat_template(
                example[args.chosen_column], tokenize=False
            ),
            "rejected": tokenizer.apply_chat_template(
                example[args.rejected_column], tokenize=False
            ),
        }

    train_dataset = raw.map(format_chat)

    if dpo_config.local_rank <= 0:
        logger.info("="*40)
        logger.info(f"模型: {args.model_name_or_path}")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total/1e6:.2f}M  可训练: {trainable/1e6:.2f}M")
        logger.info(f"精度: {torch_dtype}  LoRA: {args.use_lora}")
        logger.info("="*40)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(dpo_config.output_dir)
        tokenizer.save_pretrained(dpo_config.output_dir)
        logger.info(f"训练完成，模型已保存至: {dpo_config.output_dir}")


if __name__ == "__main__":
    main()
