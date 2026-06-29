#!/bin/bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=your_token_here   # 有限速问题时取消注释
# --- 评测阶段 ---
# sft：评测 SFT 模型，与 base 对比
# dpo：评测 DPO 模型，与 SFT 对比
STAGE="sft"

# --- 模型路径（根据 STAGE 自动切换对比基准）---
SFT_MODEL="./output_mac_smollm3"
DPO_MODEL="./output_mac_dpo"
BASE_MODEL="HuggingFaceTB/SmolLM2-360M"   # 训练起点，用于 SFT 阶段对比

if [ "$STAGE" == "sft" ]; then
    EVAL_MODEL="$SFT_MODEL"
    COMPARE_MODEL="$BASE_MODEL"
elif [ "$STAGE" == "dpo" ]; then
    EVAL_MODEL="$DPO_MODEL"
    COMPARE_MODEL="$SFT_MODEL"   # DPO 与 SFT 对比，看偏好对齐增益
else
    echo "Unknown STAGE: $STAGE (use sft or dpo)"
    exit 1
fi

# --- 评测任务（针对 smoltalk2 后训练）---
# ifeval          指令遵循（SFT 核心，smol_magpie_ultra 数据直接对应）
# gsm8k_cot       数学推理+思维链（OpenThoughts3 占 SFT 数据约 50%，最需验证）
# arc_challenge   常识推理，检查 SFT 后有无退化
# truthfulqa_mc2  抗幻觉（tulu_3 preference 对齐后应有提升）
# hellaswag       句子补全，通用能力保底
TASKS="arc_challenge,hellaswag,ifeval,gsm8k_cot,truthfulqa_mc2"

# --- 评测参数 ---
# LIMIT: 空=全量；正整数=每任务条数；小数=按比例采样
# 快速验证用 100~200，正式评测置空
LIMIT=""
BATCH_SIZE="auto"
# auto=自动（CPU/Mac→float32，CUDA→bfloat16）；或手动指定 float32/bfloat16/float16
DTYPE="auto"
DEVICE="auto"
OUTPUT_DIR="./eval_results"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

COMPARE_ARG=""
if [ -n "$COMPARE_MODEL" ]; then
    COMPARE_ARG="--base_model $COMPARE_MODEL"
fi

# 确保 lm-eval 已安装
if ! python -c "import lm_eval" 2>/dev/null; then
    echo "lm-eval 未安装，正在安装..."
    pip install lm-eval
fi

echo "-------------------------------------------"
echo "Stage: $STAGE"
echo "Model: $EVAL_MODEL"
echo "Compare: ${COMPARE_MODEL:-（无对比）}"
echo "Tasks: $TASKS"
echo "-------------------------------------------"

python eval.py \
    --model "$EVAL_MODEL" \
    $COMPARE_ARG \
    --tasks "$TASKS" \
    $LIMIT_ARG \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"
