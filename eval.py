import argparse
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# ifeval：指令遵循（SFT 主指标）
# arc_challenge/hellaswag：通用推理保底，检查 SFT 后有无退化
# truthfulqa_mc2：检查 SFT 后幻觉是否增加（smoltalk 场景下重要）
DEFAULT_TASKS = "arc_challenge,hellaswag,ifeval,truthfulqa_mc2"


def parse_args():
    parser = argparse.ArgumentParser(description="lm-evaluation-harness 评测脚本，支持 base vs SFT 对比")
    parser.add_argument("--model", required=True, help="SFT 模型路径或 HF model ID")
    parser.add_argument("--base_model", default=None, help="Base 模型路径，用于计算 SFT delta（可选）")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="逗号分隔的任务名")
    parser.add_argument("--limit", type=float, default=None,
                        help="每任务样本数（整数）或采样比例（0~1 小数），None=全量")
    parser.add_argument("--batch_size", default="auto",
                        help="batch size，auto 自动寻找不 OOM 的最大值")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="推理精度：A100/H100 用 bfloat16，旧卡用 float16，Mac 用 float32")
    parser.add_argument("--apply_chat_template", action="store_true", default=True,
                        help="对 SFT 模型评测时套用 chat template（默认开启）；"
                             "评测原始 base 模型时传 --no_apply_chat_template")
    parser.add_argument("--no_apply_chat_template", dest="apply_chat_template", action="store_false")
    parser.add_argument("--output_dir", default="./eval_results")
    return parser.parse_args()


def run_eval(model_path, tasks, limit, batch_size, device, dtype, apply_chat_template=True):
    from lm_eval import evaluator
    return evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={dtype}",
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        device=device,
        apply_chat_template=apply_chat_template,
        log_samples=False,
    )


def extract_scores(results):
    """每个任务取主指标：acc_norm > acc > f1 > exact_match"""
    scores = {}
    for task, metrics in results["results"].items():
        for key in ["acc_norm,none", "acc,none", "f1,none", "exact_match,none"]:
            if key in metrics:
                scores[task] = metrics[key]
                break
    return scores


def print_table(sft_scores, base_scores=None):
    col_task = 32
    col_val = 9
    header = f"{'Task':<{col_task}} {'SFT':>{col_val}}"
    if base_scores:
        header += f" {'Base':>{col_val}} {'Delta':>{col_val}}"
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
        import lm_eval  # noqa: F401
    except ImportError:
        raise SystemExit("请先安装: pip install lm-eval")

    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"评测模型: {args.model}")
    logger.info(f"任务: {tasks}")

    logger.info(f"apply_chat_template={'是' if args.apply_chat_template else '否（base 模式）'}")
    sft_results = run_eval(
        args.model, tasks, args.limit, args.batch_size, args.device, args.dtype,
        apply_chat_template=args.apply_chat_template,
    )
    sft_scores = extract_scores(sft_results)

    base_scores = None
    if args.base_model:
        logger.info(f"评测 base 模型: {args.base_model}（不套 chat template）")
        base_results = run_eval(
            args.base_model, tasks, args.limit, args.batch_size, args.device, args.dtype,
            apply_chat_template=False,
        )
        base_scores = extract_scores(base_results)

    print_table(sft_scores, base_scores)

    out = {
        "timestamp": timestamp,
        "model": args.model,
        "base_model": args.base_model,
        "tasks": tasks,
        "sft_scores": sft_scores,
        "base_scores": base_scores,
        "delta": (
            {t: sft_scores[t] - base_scores.get(t, float("nan")) for t in sft_scores}
            if base_scores else None
        ),
    }
    out_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
