#!/bin/bash
set -e

# --- 基础配置 ---
# export HF_ENDPOINT=https://hf-mirror.com/
# export HF_ENDPOINT=http://192.168.50.202:18090

MODEL_ID="HuggingFaceTB/SmolLM2-135M"
CHAT_TEMPLATE_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"  # 从 Instruct 版拷贝 chat template 到 base model
STRATEGY="mac" # 可选: deepspeed / fsdp / mac

# --- 数据集配置 ---
# 方案 A（默认）：smoltalk，有标准 train/test split，直接用
# DATASET="HuggingFaceTB/smoltalk"
# DATASET_NAME="all"          # 可选: smol-magpie-ultra / everyday-conversations 等子集
# DATASET_SPLIT=""            # 留空：自动取 train split

# 方案 B：smoltalk2（为 SmolLM3 重构，无 train split，需显式指定 split 名）
# smoltalk2 SFT 主要 split：
#   smoltalk_smollm3_smol_magpie_ultra_no_think   (406k，通用指令，推荐单 split)
#   OpenHermes_2.5_no_think                        (384k，多领域)
#   OpenThoughts3_1.2M_no_think_no_think           (435k，推理，无思维链)
#   OpenThoughts3_1.2M_think                       (1.1M，含思维链，需模板支持)
# 使用单个 split，取消注释以下三行：
# DATASET="HuggingFaceTB/smoltalk2"
# DATASET_NAME="SFT"
# DATASET_SPLIT="smoltalk_smollm3_smol_magpie_ultra_no_think"
#
# 使用全部 SFT 数据（~3.3M 条，streaming 推荐），取消注释以下三行：
DATASET="HuggingFaceTB/smoltalk2"
DATASET_NAME="SFT"
DATASET_SPLIT="ALL"

# 自动检测 GPU 数量，允许环境变量覆盖
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
[ "${NUM_GPUS:-0}" -eq 0 ] && NUM_GPUS=1

# --- 训练参数 ---
# LR=2e-5           学习率：SFT 通常用 1e-5~5e-5，过大破坏预训练权重，过小收敛慢
# BS_PER_GPU=8      每张卡的 batch size：越大梯度越稳定，受显存限制
# GRAD_ACCUM=4      梯度累积：等效 batch size = BS_PER_GPU × GRAD_ACCUM × NUM_GPUS
# STEPS=500         训练总步数：小数据集用步数比 epoch 更精确
# STREAMING=false   流式加载：数据集过大时开启，需同时设置 max_steps
#
# lr_scheduler_type cosine   cosine 衰减：比 linear 收尾更平滑，最终 loss 通常更低
# warmup_ratio 0.03          前 3% 步线性升温：防止训练初期大 LR 破坏预训练权重
# weight_decay 0.01          AdamW 权重衰减：抑制过拟合，SFT 小数据集尤为重要
# max_length 2048            超长样本在此截断；设太小丢信息，设太大显存不够（TRL 1.x 改名自 max_seq_length）
# bf16 True                  bfloat16：A100/H100 首选，比 fp16 数值范围更大更稳定
# gradient_checkpointing     用重计算换显存：显存不足时开启，速度约降 20%
# fsdp full_shard            参数/梯度/优化器状态全部分片到各卡，显存占用最省
#
# use_lora True              开启 LoRA：7B+ 模型推荐，可训练参数量从 100% 降到 ~1%
# lora_r 8                   LoRA rank：越大表达能力越强，显存也越多，通常 8~64
# lora_alpha 16              缩放系数，通常设为 2×lora_r
# lora_target_modules        目标模块：q_proj,v_proj（保守）或加 k_proj,o_proj,gate_proj 等
LR=2e-5
BS_PER_GPU=8
GRAD_ACCUM=4
STEPS=500
STREAMING=true
OUTPUT_DIR="./output_${STRATEGY}_smollm2"

# FSDP 需要指定模型的 decoder layer 类名，换模型时同步修改
# SmolLM2/Llama → LlamaDecoderLayer, Qwen2 → Qwen2DecoderLayer, Mistral → MistralDecoderLayer
FSDP_DECODER_LAYER="LlamaDecoderLayer"

STREAMING_ARG=""
if [ "$STREAMING" = "true" ] || [ "$STREAMING" = "True" ] || [ "$STREAMING" = "1" ]; then
    STREAMING_ARG="--streaming"
fi

SPLIT_ARG=""
if [ -n "$DATASET_SPLIT" ]; then
    SPLIT_ARG="--dataset_split $DATASET_SPLIT"
fi

echo "---------------------------------------"
echo "Starting SFT Training with $STRATEGY"
echo "GPUs: $NUM_GPUS, Learning Rate: $LR"
echo "---------------------------------------"

if [ "$STRATEGY" == "deepspeed" ]; then

    echo "Running DeepSpeed"

    torchrun --nproc_per_node="$NUM_GPUS" sft.py \
        --model_id "$MODEL_ID" \
        --chat_template_model "$CHAT_TEMPLATE_MODEL" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        $SPLIT_ARG \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --max_length 2048 \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --deepspeed ds_config.json \
        --bf16 True \
        --assistant_only_loss True \
        --per_device_train_batch_size "$BS_PER_GPU" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --gradient_checkpointing True

elif [ "$STRATEGY" == "fsdp" ]; then

    echo "Running FSDP"

    FSDP_CONFIG="{\"transformer_layer_cls_to_wrap\": \"$FSDP_DECODER_LAYER\"}"
    torchrun --nproc_per_node="$NUM_GPUS" sft.py \
        --model_id "$MODEL_ID" \
        --chat_template_model "$CHAT_TEMPLATE_MODEL" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        $SPLIT_ARG \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --max_length 2048 \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_config "$FSDP_CONFIG" \
        --bf16 True \
        --assistant_only_loss True \
        --per_device_train_batch_size "$BS_PER_GPU" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --gradient_checkpointing True

elif [ "$STRATEGY" == "mac" ]; then

    echo "Running Mac MPS training"

    export PYTORCH_ENABLE_MPS_FALLBACK=1

    python sft.py \
        --model_id "$MODEL_ID" \
        --chat_template_model "$CHAT_TEMPLATE_MODEL" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        $SPLIT_ARG \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --max_length 2048 \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --bf16 False \
        --fp16 False \
        --assistant_only_loss True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing True

else

    echo "Unknown strategy: $STRATEGY"
    exit 1

fi
