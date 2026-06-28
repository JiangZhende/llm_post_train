#!/bin/bash
set -e

# --- 基础配置 ---
# export HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=http://192.168.50.202:18090

MODEL_ID="HuggingFaceTB/SmolLM2-135M"
DATASET="trl-lib/ultrafeedback_binarized"
DATASET_NAME=""         # 无子集配置
DATASET_SPLIT="train"
STRATEGY="mac" # 可选: deepspeed / fsdp / mac

# 自动检测 GPU 数量，允许环境变量覆盖
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
[ "${NUM_GPUS:-0}" -eq 0 ] && NUM_GPUS=1

# --- 训练参数 ---
# LR=5e-7        DPO 学习率远小于 SFT，过大会破坏对齐效果
LR=5e-7
# BS_PER_GPU=4   DPO 每条样本包含 chosen+rejected 两条，显存占用约为 SFT 的 2 倍
BS_PER_GPU=4
GRAD_ACCUM=4
STEPS=500
STREAMING=false
OUTPUT_DIR="./output_${STRATEGY}_dpo"

# FSDP decoder layer 类名，换模型时同步修改
FSDP_DECODER_LAYER="LlamaDecoderLayer"

STREAMING_ARG=""
if [ "$STREAMING" = "true" ] || [ "$STREAMING" = "True" ] || [ "$STREAMING" = "1" ]; then
    STREAMING_ARG="--streaming"
fi

# 公共参数（三个策略共享）
# beta 0.1          DPO 温度：控制 policy 偏离 reference 的惩罚力度，越大越保守
# max_length 1024   prompt+response 总长度上限
# max_prompt_length 512  prompt 长度上限，超出则截断
# lr_scheduler cosine    收尾更平滑
# warmup_ratio 0.03      防初期破坏权重
# weight_decay 0.01      抑制过拟合

echo "---------------------------------------"
echo "Starting DPO Training with $STRATEGY"
echo "GPUs: $NUM_GPUS, Learning Rate: $LR"
echo "---------------------------------------"

if [ "$STRATEGY" == "deepspeed" ]; then

    echo "Running DeepSpeed"

    torchrun --nproc_per_node="$NUM_GPUS" dpo.py \
        --model_name_or_path "$MODEL_ID" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        --dataset_split "$DATASET_SPLIT" \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --beta 0.1 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --per_device_train_batch_size "$BS_PER_GPU" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --bf16 True \
        --gradient_checkpointing True \
        --deepspeed ds_config.json

elif [ "$STRATEGY" == "fsdp" ]; then

    echo "Running FSDP"

    FSDP_CONFIG="{\"transformer_layer_cls_to_wrap\": \"$FSDP_DECODER_LAYER\"}"
    torchrun --nproc_per_node="$NUM_GPUS" dpo.py \
        --model_name_or_path "$MODEL_ID" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        --dataset_split "$DATASET_SPLIT" \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --beta 0.1 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --per_device_train_batch_size "$BS_PER_GPU" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --bf16 True \
        --gradient_checkpointing True \
        --fsdp "full_shard auto_wrap" \
        --fsdp_config "$FSDP_CONFIG"

elif [ "$STRATEGY" == "mac" ]; then

    echo "Running Mac MPS training"

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export ACCELERATE_USE_FSDP=false

    # MPS 显存有限，batch size=1，grad accum 补偿等效 batch size
    python dpo.py \
        --model_name_or_path "$MODEL_ID" \
        --dataset_path "$DATASET" \
        --dataset_name "$DATASET_NAME" \
        --dataset_split "$DATASET_SPLIT" \
        $STREAMING_ARG \
        --output_dir "$OUTPUT_DIR" \
        --beta 0.1 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_steps "$STEPS" \
        --logging_steps 10 \
        --save_steps 100 \
        --eval_strategy steps \
        --eval_steps 100 \
        --bf16 False \
        --fp16 False \
        --gradient_checkpointing True

else

    echo "Unknown strategy: $STRATEGY"
    exit 1

fi
