#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com

# --- 评测阶段 ---
# base：评测原始 base 模型（无 chat template，仅 MC 任务）
# sft ：评测 SFT 模型，与 base 对比
# dpo ：评测 DPO 模型，与 SFT 对比
STAGE="base"

# --- 模型路径 ---
SFT_MODEL="./output_mac_smollm3"
DPO_MODEL="./output_mac_dpo"
BASE_MODEL="HuggingFaceTB/SmolLM2-360M"

if [ "$STAGE" == "base" ]; then
    EVAL_MODEL="$BASE_MODEL"
    COMPARE_MODEL=""
elif [ "$STAGE" == "sft" ]; then
    EVAL_MODEL="$SFT_MODEL"
    COMPARE_MODEL="$BASE_MODEL"
elif [ "$STAGE" == "dpo" ]; then
    EVAL_MODEL="$DPO_MODEL"
    COMPARE_MODEL="$SFT_MODEL"
else
    echo "Unknown STAGE: $STAGE (use base / sft / dpo)"
    exit 1
fi

# --- 评测任务 ---
# base 阶段只跑 MC 任务（likelihood 打分），生成型任务（ifeval/gsm8k_cot）对 base 无意义
# sft/dpo 阶段全跑；可额外加 mmlu / gpqa_diamond / gsm_plus
if [ "$STAGE" == "base" ]; then
    TASKS="arc_challenge,hellaswag,truthfulqa_mc2"
else
    TASKS="arc_challenge,hellaswag,ifeval,gsm8k_cot,truthfulqa_mc2"
fi

# --- 评测参数 ---
# LIMIT: 空=全量；正整数=每任务条数（lighteval 只接受整数，不支持比例）
LIMIT="10"
BATCH_SIZE=""      # 空=使用默认值；或指定整数如 8
DTYPE="bfloat16"
DEVICE="auto"
OUTPUT_DIR="./eval_results"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

BATCH_ARG=""
if [ -n "$BATCH_SIZE" ]; then
    BATCH_ARG="--batch_size $BATCH_SIZE"
fi

COMPARE_ARG=""
if [ -n "$COMPARE_MODEL" ]; then
    COMPARE_ARG="--base_model $COMPARE_MODEL"
fi

# SFT/DPO 模型强制套 chat template，base 模型不套
CHAT_TEMPLATE_ARG=""
if [ "$STAGE" == "sft" ] || [ "$STAGE" == "dpo" ]; then
    CHAT_TEMPLATE_ARG="--apply_chat_template"
fi

if ! python -c "import lighteval" 2>/dev/null; then
    echo "lighteval 未安装，正在安装..."
    pip install lighteval
fi

echo "-------------------------------------------"
echo "Framework: lighteval"
echo "Stage:   $STAGE"
echo "Model:   $EVAL_MODEL"
echo "Compare: ${COMPARE_MODEL:-（无对比）}"
echo "Tasks:   $TASKS"
echo "-------------------------------------------"

python eval_lighteval.py \
    --model "$EVAL_MODEL" \
    $COMPARE_ARG \
    --stage "$STAGE" \
    --tasks "$TASKS" \
    $LIMIT_ARG \
    $BATCH_ARG \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    $CHAT_TEMPLATE_ARG
