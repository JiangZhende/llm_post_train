# LLM Post-Training：SFT → DPO → Eval

基于 [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) 和 [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) 的完整后训练流程。

```
Base Model
    │
    ▼  sft.sh / sft.py
SFT Model   ←  smoltalk2 (sft subset, 3.4M 对话)
    │
    ▼  dpo.sh / dpo.py
DPO Model   ←  smoltalk2 (preference subset, 447k chosen/rejected 对)
    │
    ▼  eval.sh / eval.py
评测结果 (arc / hellaswag / ifeval / truthfulqa)
```

## 依赖安装

```bash
pip install torch transformers trl datasets peft accelerate lm-eval
```

DeepSpeed / FSDP 额外需要：

```bash
pip install deepspeed
```

## 数据集

| 用途 | 数据集 | 子集 | 规模 | 格式 |
|------|--------|------|------|------|
| SFT  | `HuggingFaceTB/smoltalk2` | `sft` | 3.4M | `messages`（list of dicts） |
| DPO  | `HuggingFaceTB/smoltalk2` | `preference` | 447k | `prompt`（str）+ `chosen`/`rejected`（list of dicts） |

国内可通过镜像加速下载，在 sh 文件中切换：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## SFT

**文件：** `sft.sh` / `sft.py`

### 快速启动

```bash
# Mac（MPS，默认）
bash sft.sh

# 多卡 DeepSpeed
STRATEGY=deepspeed bash sft.sh

# 多卡 FSDP
STRATEGY=fsdp bash sft.sh
```

### 关键参数（修改 sft.sh 顶部变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_ID` | `HuggingFaceTB/SmolLM2-135M` | 模型 ID 或本地路径 |
| `STRATEGY` | `mac` | 并行策略：`mac` / `deepspeed` / `fsdp` |
| `LR` | `2e-5` | 学习率，SFT 推荐 1e-5 ~ 5e-5 |
| `BS_PER_GPU` | `8` | 每卡 batch size（mac 策略固定为 1） |
| `GRAD_ACCUM` | `4` | 梯度累积，等效 batch = BS × ACCUM × GPU 数 |
| `STEPS` | `500` | 训练步数 |
| `STREAMING` | `false` | 数据集过大时开启流式加载 |
| `OUTPUT_DIR` | `./output_mac_smollm2` | 模型输出目录 |

### LoRA（可选，7B+ 模型推荐）

```bash
python sft.py \
    --model_id meta-llama/Llama-3.1-8B \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,v_proj,k_proj,o_proj \
    ... 其他参数
```

### 换模型时注意

FSDP 策略需同步修改 decoder layer 类名：

```bash
# sft.sh 中
FSDP_DECODER_LAYER="LlamaDecoderLayer"    # SmolLM2 / Llama
# FSDP_DECODER_LAYER="Qwen2DecoderLayer"  # Qwen2
# FSDP_DECODER_LAYER="MistralDecoderLayer" # Mistral
```

---

## DPO

**文件：** `dpo.sh` / `dpo.py`

DPO 在 SFT 模型基础上做偏好对齐，将 `MODEL_ID` 改为 SFT 输出路径后运行。

### 快速启动

```bash
# 先改 dpo.sh 中 MODEL_ID 为 SFT 输出路径
MODEL_ID="./output_mac_smollm2"

# Mac
bash dpo.sh

# 多卡 DeepSpeed
STRATEGY=deepspeed bash dpo.sh
```

### 关键参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_ID` | `HuggingFaceTB/SmolLM2-135M` | **建议改为 SFT 输出路径** |
| `LR` | `5e-7` | DPO 学习率远小于 SFT，过大破坏对齐 |
| `BS_PER_GPU` | `4` | DPO 每条样本含 chosen+rejected，显存约 SFT 2 倍 |
| `BETA` | `0.1` | DPO 温度，控制 policy 偏离 reference 的惩罚力度 |
| `MAX_LENGTH` | `1024` | prompt+response 总长上限 |
| `MAX_PROMPT_LENGTH` | `512` | prompt 长度上限 |

### LoRA DPO（推荐用于大模型）

使用 LoRA 时无需显式加载 reference model（DPOTrainer 自动从 base 推断），显存减半：

```bash
python dpo.py \
    --model_name_or_path ./output_mac_smollm2 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    ... 其他参数
```

---

## 评测

**文件：** `eval.sh` / `eval.py`

### 快速启动

```bash
# 修改 eval.sh 中 SFT_MODEL 路径后运行
bash eval.sh

# 同时对比 base 模型
BASE_MODEL="HuggingFaceTB/SmolLM2-135M" bash eval.sh
```

### 直接调用 eval.py

```bash
# 评测 SFT 模型（自动套 chat template）
python eval.py --model ./output_mac_smollm2

# 与 base 模型对比
python eval.py \
    --model ./output_mac_dpo \
    --base_model HuggingFaceTB/SmolLM2-135M \
    --tasks arc_challenge,hellaswag,ifeval,truthfulqa_mc2

# 快速验证（每任务 50 条）
python eval.py \
    --model ./output_mac_smollm2 \
    --limit 50 \
    --device mps \
    --dtype float32

# 不套 chat template（评测 base 模型时）
python eval.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --no_apply_chat_template
```

### 评测任务说明

| 任务 | 指标 | 用途 |
|------|------|------|
| `ifeval` | prompt-level acc | SFT 主指标，直接衡量指令遵循能力 |
| `arc_challenge` | acc_norm | 常识推理，检验 SFT 后有无退化 |
| `hellaswag` | acc_norm | 句子补全，通用能力保底 |
| `truthfulqa_mc2` | acc | 检测 SFT/DPO 后幻觉是否增加 |

输出示例：

```
===================================================
Task                             SFT      Base     Delta
---------------------------------------------------
arc_challenge                  42.15%   38.91%   +3.24%
hellaswag                      61.30%   59.87%   +1.43%
ifeval                         55.20%   12.30%  +42.90%
truthfulqa_mc2                 48.60%   45.10%   +3.50%
===================================================
```

结果自动保存至 `eval_results/eval_<timestamp>.json`。

---

## 完整流程示例

```bash
# 1. SFT
bash sft.sh
# → 输出: ./output_mac_smollm2/

# 2. 评测 SFT 效果
python eval.py \
    --model ./output_mac_smollm2 \
    --base_model HuggingFaceTB/SmolLM2-135M \
    --limit 200

# 3. DPO（在 SFT 基础上）
# 先在 dpo.sh 中将 MODEL_ID 改为 ./output_mac_smollm2
bash dpo.sh
# → 输出: ./output_mac_dpo/

# 4. 评测 DPO 效果
python eval.py \
    --model ./output_mac_dpo \
    --base_model ./output_mac_smollm2 \
    --limit 200
```
