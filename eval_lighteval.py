import argparse
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# 友好任务名 → lighteval task string
# 格式: "{suite}|{task}|{num_few_shots}|{truncated_few_shot}"
# few-shot 数量与 SmolLM3 官方评测保持一致
TASK_REGISTRY = {
    "ifeval":         "lighteval|ifeval|0|0",
    "gsm8k":          "lighteval|gsm8k|8|0",
    "gsm8k_cot":      "lighteval|gsm8k|8|0",
    "arc_challenge":  "lighteval|arc:challenge|25|0",
    "hellaswag":      "lighteval|hellaswag|10|0",
    "truthfulqa_mc2": "lighteval|truthfulqa:mc|0|0",
    "mmlu":           "lighteval|mmlu:_average|5|0",
    "gpqa_diamond":   "lighteval|gpqa:diamond|0|0",
    "gsm_plus":       "community|gsm_plus|8|0",
}

DEFAULT_TASKS = "arc_challenge,hellaswag,ifeval,gsm8k_cot,truthfulqa_mc2"

_GENERATION_TASKS = {"ifeval", "gsm8k_cot", "gsm8k", "gsm_plus"}

# 各任务的主指标，按优先级匹配
TASK_PRIMARY_METRICS = {
    "ifeval":        ["prompt_level_strict_acc", "instruction_level_strict_acc"],
    "gsm8k":         ["exact_match", "qem"],
    "arc:challenge": ["acc_norm", "acc"],
    "hellaswag":     ["acc_norm", "acc"],
    "truthfulqa:mc": ["mc2", "acc"],
    "mmlu:_average": ["acc"],
    "gpqa:diamond":  ["acc", "pass@1"],
    "gsm_plus":      ["exact_match"],
}
_FALLBACK_METRICS = ["acc_norm", "acc", "exact_match", "mc2", "f1", "prompt_level_strict_acc", "qem"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="lighteval 评测脚本（SmolLM3 官方工具链），支持 base vs SFT/DPO 对比"
    )
    parser.add_argument("--model", required=True, help="待评测模型路径或 HF model ID")
    parser.add_argument("--base_model", default=None, help="对比基准模型路径（可选）")
    parser.add_argument("--stage", default=None, choices=["base", "sft", "dpo"],
                        help="评测阶段，写入 JSON 结果，影响表格标题（可选）")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="逗号分隔的任务名")
    parser.add_argument("--limit", type=int, default=None,
                        help="每任务最大样本数（整数），None=全量；注意 lighteval 只接受整数")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="batch size，None=使用模型默认值（通常为 1）")
    parser.add_argument("--device", default="auto",
                        help="设备：auto（自动检测）/ cuda / cpu / mps，或 GPU 序号如 0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["auto", "bfloat16", "float16", "float32", "4bit", "8bit"],
                        help="推理精度，lighteval 额外支持 4bit/8bit 量化")
    parser.add_argument("--apply_chat_template", action="store_true", default=False,
                        help="SFT/DPO 模型评测时传此 flag，强制套用 chat template；base 模型不传")
    parser.add_argument("--keep_reasoning_tags", action="store_true", default=False,
                        help="保留 <think> 等推理 tag（默认剥离后再 judge 答案）；"
                             "测试 think 模式时传此 flag")
    parser.add_argument("--output_dir", default="./eval_results")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_tasks(task_names: list) -> tuple:
    """将友好任务名转为 lighteval task string，返回 (task_string, unknown_list)"""
    task_strings, unknown = [], []
    for name in task_names:
        name = name.strip()
        if name in TASK_REGISTRY:
            task_strings.append(TASK_REGISTRY[name])
        elif "|" in name:
            # 已经是 lighteval 格式（如 lighteval|ifeval|0|0），直接使用
            task_strings.append(name)
        else:
            unknown.append(name)
    return ",".join(task_strings), unknown


def _task_short_name(task_string: str) -> str:
    """从 task string 提取短名，如 lighteval|arc:challenge|25|0 → arc:challenge"""
    parts = task_string.split("|")
    return parts[1] if len(parts) >= 2 else task_string


def run_eval(model_path, task_string, limit, batch_size, device, dtype, apply_chat_template,
             keep_reasoning_tags=False):
    from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.transformers.transformers_model import TransformersModelConfig

    # override_chat_template:
    #   True  → 强制使用 chat template（SFT/DPO 模型）
    #   False → 强制不用 chat template（base 模型）
    #   None  → 自动（tokenizer 有 template 就用，没有就不用）
    override_chat_template = apply_chat_template

    model_config = TransformersModelConfig(
        model_name=model_path,
        dtype=dtype,
        batch_size=batch_size,
        device=device,
        trust_remote_code=True,
        override_chat_template=override_chat_template,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=limit,
        # remove_reasoning_tags=True（默认）：judge 前剥离 <think> 块，只看最终答案
        # 传 --keep_reasoning_tags 时关闭，用于测试 think 模式的完整输出
        remove_reasoning_tags=not keep_reasoning_tags,
    )

    # 用临时目录存 lighteval 自身的输出，最终结果由本脚本统一保存
    tracker = EvaluationTracker(
        output_dir=os.path.join(os.path.dirname(__file__), ".lighteval_tmp"),
        save_details=False,
    )

    pipeline = Pipeline(
        tasks=task_string,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()

    return tracker.results


def extract_scores(results: dict, task_names: list) -> dict:
    """从 lighteval results 中提取每个任务的主指标分数"""
    raw = results.get("results", results)
    scores = {}

    for name in task_names:
        task_string = TASK_REGISTRY.get(name, name)
        short_name = _task_short_name(task_string)

        # lighteval results 结构：{short_name: {metric: score, ...}}
        task_metrics = None
        if short_name in raw:
            task_metrics = raw[short_name]
        else:
            # 宽松匹配：遍历所有 key，找包含 short_name 的
            for k in raw:
                if short_name in k and isinstance(raw[k], dict):
                    task_metrics = raw[k]
                    break

        if task_metrics is None:
            logger.warning(
                f"任务 {name!r}（short_name={short_name!r}）在 results 中未找到。"
                f"可用 key: {list(raw.keys())[:10]}"
            )
            continue

        # 按优先级取主指标
        value = None
        candidates = TASK_PRIMARY_METRICS.get(short_name, []) + TASK_PRIMARY_METRICS.get(name, [])
        for metric in candidates:
            if metric in task_metrics:
                value = task_metrics[metric]
                break

        # 兜底：按通用指标名匹配
        if value is None:
            for m in _FALLBACK_METRICS:
                if m in task_metrics:
                    value = task_metrics[m]
                    break

        # 最终兜底：取第一个数值型字段
        if value is None:
            for k, v in task_metrics.items():
                if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                    value = v
                    break

        if value is not None:
            scores[name] = float(value)
        else:
            logger.warning(f"任务 {name!r} 未找到可用指标，原始 metrics: {task_metrics}")

    return scores


def print_table(sft_scores, base_scores=None, model_label="Model"):
    col_task, col_val = 32, 9
    header = f"{'Task':<{col_task}} {model_label:>{col_val}}"
    if base_scores:
        header += f" {'Ref':>{col_val}} {'Delta':>{col_val}}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for task, sft in sft_scores.items():
        row = f"{task:<{col_task}} {sft:>{col_val}.2%}"
        if base_scores:
            base = base_scores.get(task, float("nan"))
            delta = sft - base
            sign = "+" if delta >= 0 else ""
            row += f" {base:>{col_val}.2%} {sign}{delta:>{col_val - 1}.2%}"
        print(row)
    print(f"{sep}\n")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    try:
        import lighteval  # noqa: F401
    except ImportError:
        raise SystemExit(
            "请先安装 lighteval:\n"
            "  pip install lighteval\n"
            "多卡或更快推理：\n"
            "  pip install lighteval[accelerate]   # 多卡\n"
            "  pip install lighteval[vllm]         # vLLM 后端"
        )

    args = parse_args()
    task_names = [t.strip() for t in args.tasks.split(",")]
    task_string, unknown = resolve_tasks(task_names)
    if unknown:
        logger.warning(f"未知任务名已跳过: {unknown}，可用任务: {list(TASK_REGISTRY.keys())}")
    if not task_string:
        raise SystemExit("没有可用的任务，退出")

    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"评测模型: {args.model}  stage={args.stage or '未指定'}")
    logger.info(f"任务: {task_names}")
    logger.info(f"task_string: {task_string}")
    logger.info(f"device={device}  dtype={args.dtype}  limit={args.limit}")
    logger.info(f"apply_chat_template={'是（SFT/DPO 模式）' if args.apply_chat_template else '否（base 模式）'}")

    if not args.apply_chat_template:
        gen_overlap = [t for t in task_names if t in _GENERATION_TASKS]
        if gen_overlap:
            logger.warning(
                f"以下任务为生成型任务，base 模型（无 chat template）的分数通常 <10%，仅供参考: {gen_overlap}"
            )

    model_scores = extract_scores(
        run_eval(args.model, task_string, args.limit, args.batch_size,
                 device, args.dtype, args.apply_chat_template,
                 keep_reasoning_tags=args.keep_reasoning_tags),
        task_names,
    )

    ref_scores = None
    if args.base_model:
        logger.info(f"评测对比模型: {args.base_model}（不套 chat template）")
        ref_scores = extract_scores(
            run_eval(args.base_model, task_string, args.limit, args.batch_size,
                     device, args.dtype, apply_chat_template=False,
                     keep_reasoning_tags=False),
            task_names,
        )

    model_label = (args.stage or "Model").upper()
    print_table(model_scores, ref_scores, model_label=model_label)

    out = {
        "timestamp": timestamp,
        "framework": "lighteval",
        "stage": args.stage,
        "model": args.model,
        "ref_model": args.base_model,
        "tasks": task_names,
        "task_strings": task_string.split(","),
        "model_scores": model_scores,
        "ref_scores": ref_scores,
        "delta": (
            {t: model_scores[t] - ref_scores.get(t, float("nan")) for t in model_scores}
            if ref_scores else None
        ),
    }
    out_path = os.path.join(args.output_dir, f"eval_lighteval_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
