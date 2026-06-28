#!/bin/bash
set -e

# 模型为本地路径，无需设置 HF_ENDPOINT；数据集走默认 huggingface.co

# --- 评测目标 ---
SFT_MODEL="../models/Qwen3-0.6B"       # SFT 后的模型路径
BASE_MODEL="" # base 模型，用于计算 delta；不需要对比时置空

# --- 任务说明 ---
# arc_easy / arc_challenge  常识推理，适合所有规模模型，有公开排行榜可对比
# hellaswag                 句子补全，常用通用能力基准
# ifeval                    指令遵循，SFT 最核心指标，直接反映微调效果
TASKS="arc_easy,arc_challenge,gsm8k"  # 本地已缓存；hellaswag/ifeval 未缓存需联网

# --- 评测参数 ---
# LIMIT: 空=全量；正整数=每任务样本数；0~1 小数=按比例采样（快速验证用）
LIMIT=50
# BATCH_SIZE: auto 自动寻找不 OOM 的最大值；显存紧张时手动设为 4 或 8
BATCH_SIZE=4
# DTYPE: A100/H100 → bfloat16；旧卡(V100/T4) → float16；Mac → float32
DTYPE="float32"
DEVICE="cpu"  # macOS 13 不支持新版 transformers 的 MPS，14+ 可改回 mps
OUTPUT_DIR="./eval_results"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

BASE_ARG=""
if [ -n "$BASE_MODEL" ]; then
    BASE_ARG="--base_model $BASE_MODEL"
fi

# 确保 lm-eval 已安装
if ! python -c "import lm_eval" 2>/dev/null; then
    echo "lm-eval 未安装，正在安装..."
    pip install lm-eval
fi

python eval.py \
    --model "$SFT_MODEL" \
    $BASE_ARG \
    --tasks "$TASKS" \
    $LIMIT_ARG \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"
