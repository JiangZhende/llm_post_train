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
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "apply_chat_template 时是否开启思维链（Qwen3 think 模式）；"
                          "使用 no_think 数据时保持 False"},
    )
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
    base_load_kwargs = dict(streaming=args.streaming)
    if args.dataset_name:
        base_load_kwargs["name"] = args.dataset_name

    if args.streaming and dpo_config.max_steps <= 0:
        raise ValueError("streaming=True 时必须设置 --max_steps")

    split_arg = args.dataset_split.strip()

    if not split_arg or split_arg == "train":
        # 标准数据集（有 train split）
        raw = load_dataset(args.dataset_path, split="train", **base_load_kwargs)
        eval_split = "test"
    elif split_arg.upper() == "ALL" or "," in split_arg:
        # 合并多个 split（smoltalk2 Preference 无 train split）
        from datasets import load_dataset_builder, interleave_datasets, concatenate_datasets
        if split_arg.upper() == "ALL":
            builder = load_dataset_builder(
                args.dataset_path,
                args.dataset_name if args.dataset_name else None,
            )
            splits = list(builder.info.splits.keys())
            logger.info(f"ALL 模式：合并 {len(splits)} 个 split，共 "
                        f"{sum(v.num_examples for v in builder.info.splits.values()):,} 条样本")
        else:
            splits = [s.strip() for s in split_arg.split(",")]

        if args.streaming:
            raw = interleave_datasets(
                [load_dataset(args.dataset_path, split=s, **base_load_kwargs) for s in splits],
                seed=42, stopping_strategy="all_exhausted",
            )
        else:
            raw = load_dataset(args.dataset_path, split="+".join(splits), **base_load_kwargs)
        eval_split = None
    else:
        # 单个指定 split
        raw = load_dataset(args.dataset_path, split=split_arg, **base_load_kwargs)
        eval_split = None

    # Qwen3 的 apply_chat_template 支持 enable_thinking 参数控制是否输出 <think> 块；
    # 其他模型的 tokenizer 不认识这个参数，用 try/except 兼容。
    def _apply_template(messages, **kwargs):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=args.enable_thinking, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

    def format_chat(example):
        chosen = example[args.chosen_column]
        rejected = example[args.rejected_column]

        # smoltalk2 preference: prompt 是纯字符串，包一层 user message
        # ultrafeedback_binarized: 无 prompt 列，从 chosen 对话提取
        if args.prompt_column in example:
            prompt = example[args.prompt_column]
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
        else:
            prompt = chosen[:-1]

        return {
            "prompt":   _apply_template(prompt, add_generation_prompt=True),
            "chosen":   _apply_template(chosen),
            "rejected": _apply_template(rejected),
        }

    train_dataset = raw.map(format_chat)

    # eval dataset：有 test split 则加载，否则关闭 eval
    eval_dataset = None
    if not args.streaming and eval_split:
        try:
            raw_eval = load_dataset(args.dataset_path, split=eval_split, **base_load_kwargs)
            eval_dataset = raw_eval.map(format_chat)
        except Exception:
            logger.warning(f"未找到 {eval_split} split，跳过 eval")
            dpo_config.eval_strategy = "no"
    elif not eval_split:
        dpo_config.eval_strategy = "no"

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
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(dpo_config.output_dir)
        tokenizer.save_pretrained(dpo_config.output_dir)
        logger.info(f"训练完成，模型已保存至: {dpo_config.output_dir}")


if __name__ == "__main__":
    main()
