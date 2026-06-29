# LLM Post-Training：SFT → DPO → Eval

基于 [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)（训练）和 [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)（数据）的完整后训练流程。

```
Base Model（SmolLM2-135M）
    │
    ▼  sft.sh / sft.py
SFT Model   ←  smoltalk2 SFT（~3.3M 对话，含推理/指令/工具调用）
    │
    ▼  dpo.sh / dpo.py
DPO Model   ←  smoltalk2 Preference（~447k chosen/rejected 对）
    │
    ▼  eval.sh / eval.py
评测结果（ifeval / gsm8k_cot / arc / hellaswag / truthfulqa）
```

## 依赖安装

```bash
pip install torch transformers trl datasets peft accelerate lm-eval tensorboard
```

DeepSpeed / FSDP 额外需要：

```bash
pip install deepspeed
```

## 数据集

| 用途 | 数据集 | 配置/Split | 规模 | 格式 |
|------|--------|-----------|------|------|
| SFT  | `HuggingFaceTB/smoltalk2` | `SFT` / `ALL` | ~3.3M | `messages`（list of dicts）+ `chat_template_kwargs` |
| DPO  | `HuggingFaceTB/smoltalk2` | `Preference` / 指定 split | ~447k | `prompt`（str）+ `chosen`/`rejected`（完整对话 list of dicts） |

### smoltalk2 SFT 数据构成

smoltalk2 无标准 `train` split，每个数据源是独立的 split，通过 `--dataset_split ALL` 全部合并使用。

| Split | 规模 | 内容 |
|-------|------|------|
| `smoltalk_smollm3_smol_magpie_ultra_no_think` | 407k | 通用指令遵循 |
| `OpenHermes_2.5_no_think` | 385k | 多领域知识 |
| `OpenThoughts3_1.2M_no_think_no_think` | 435k | 数学/推理（无思维链） |
| `OpenThoughts3_1.2M_think` | 1.13M | 数学/推理（含 `<think>` 思维链） |
| `smoltalk_multilingual8_Qwen3_32B_think` | 245k | 多语言 |
| 其余 19 个 split | ~700k | 工具调用、科学、总结等 |

国内下载可在 sh 文件中切换镜像：

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
| `MODEL_ID` | `SmolLM2-135M` | 模型 ID 或本地路径 |
| `DATASET` | `HuggingFaceTB/smoltalk2` | 数据集 |
| `DATASET_NAME` | `SFT` | smoltalk2 的配置名 |
| `DATASET_SPLIT` | `ALL` | `ALL`=全部合并；单个 split 名；逗号分隔多个 |
| `STRATEGY` | `mac` | 并行策略：`mac` / `deepspeed` / `fsdp` |
| `LR` | `2e-5` | 学习率，SFT 推荐 1e-5 ~ 5e-5 |
| `STEPS` | `500` | 训练步数（streaming 必须设置） |
| `STREAMING` | `true` | smoltalk2 数据量大，默认开启 |

### 使用单个 split（快速实验）

```bash
# 在 sft.sh 中修改：
DATASET_SPLIT="smoltalk_smollm3_smol_magpie_ultra_no_think"   # 407k，指令遵循
# 或
DATASET_SPLIT="OpenThoughts3_1.2M_no_think_no_think"           # 435k，数学推理
```

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
# 先在 dpo.sh 中将 MODEL_ID 改为 SFT 输出路径：
# MODEL_ID="./output_mac_smollm3"

# Mac
bash dpo.sh

# 多卡 DeepSpeed
STRATEGY=deepspeed bash dpo.sh
```

### 关键参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_ID` | `SmolLM2-135M` | **建议改为 SFT 输出路径** |
| `DATASET` | `HuggingFaceTB/smoltalk2` | 偏好数据集 |
| `DATASET_NAME` | `Preference` | smoltalk2 的偏好配置 |
| `DATASET_SPLIT` | `llama_3.1_tulu_3_8b_preference_mixture_no_think` | 无 train split，需显式指定；`ALL` 合并两个 split（~447k） |
| `LR` | `5e-7` | DPO 学习率远小于 SFT，过大破坏对齐 |
| `BS_PER_GPU` | `4` | DPO 每条样本含 chosen+rejected，显存约 SFT 2 倍 |
| `BETA` | `0.1` | DPO 温度，控制 policy 偏离 reference 的惩罚力度 |
| `MAX_LENGTH` | `1024` | prompt+response 总长上限 |
| `MAX_PROMPT_LENGTH` | `512` | prompt 长度上限 |

**Preference split 说明：**

| Split | 规模 | 内容 |
|-------|------|------|
| `llama_3.1_tulu_3_8b_preference_mixture_no_think` | 230k | tulu_3 偏好混合，无思维链，**推荐** |
| `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think` | 216k | 含思维链，需模板支持 |
| `ALL` | 447k | 两者合并 |

---

## 评测

**文件：** `eval.sh` / `eval.py`

### 评测任务（针对 smoltalk2 训练内容）

| 任务 | 指标 | 对应训练数据 | 期望变化 |
|------|------|------------|---------|
| `ifeval` | prompt-level acc | smol_magpie_ultra（指令遵循） | SFT 后大幅提升 |
| `gsm8k_cot` | exact_match | OpenThoughts3（占数据~50%） | SFT 后明显提升 |
| `arc_challenge` | acc_norm | 通用，退化检查 | 基本持平 |
| `hellaswag` | acc_norm | 通用能力保底 | 基本不变 |
| `truthfulqa_mc2` | acc | tulu_3 preference 对齐 | DPO 后提升 |

### 快速启动

```bash
# 评测 SFT 模型（与 base 对比）
STAGE=sft bash eval.sh

# 评测 DPO 模型（与 SFT 对比）
STAGE=dpo bash eval.sh
```

### 直接调用 eval.py

```bash
# SFT 效果对比 base
python eval.py \
    --model ./output_mac_smollm3 \
    --base_model HuggingFaceTB/SmolLM2-360M \
    --tasks arc_challenge,hellaswag,ifeval,gsm8k_cot,truthfulqa_mc2

# DPO 效果对比 SFT
python eval.py \
    --model ./output_mac_dpo \
    --base_model ./output_mac_smollm3 \
    --tasks ifeval,gsm8k_cot,truthfulqa_mc2

# 快速验证（每任务 100 条）
python eval.py \
    --model ./output_mac_smollm3 \
    --limit 100 \
    --dtype float32

# base 模型 baseline（不套 chat template）
python eval.py \
    --model HuggingFaceTB/SmolLM2-360M \
    --no_apply_chat_template
```

输出示例：

```
================================================================
Task                              SFT      Base     Delta
----------------------------------------------------------------
arc_challenge                   43.10%   38.91%   +4.19%
hellaswag                       62.40%   59.87%   +2.53%
ifeval                          58.30%   11.20%  +47.10%
gsm8k_cot                       45.20%   18.60%  +26.60%
truthfulqa_mc2                  49.80%   45.10%   +4.70%
================================================================
```

结果自动保存至 `eval_results/eval_<timestamp>.json`。

---

## 训练监控

训练日志写入 `./runs/`，使用 TensorBoard 查看：

```bash
tensorboard --logdir ./runs
# 浏览器打开 http://localhost:6006
```

关键指标：
- `train/loss`：应持续下降
- `train/grad_norm`：突然飙升说明训练不稳定
- DPO 专有：`train/rewards/margins`（chosen−rejected），应稳步上升

---

## 完整流程示例

```bash
# 1. SFT（smoltalk2 全量，streaming）
bash sft.sh
# → 输出: ./output_mac_smollm3/

# 2. 评测 SFT 效果
STAGE=sft bash eval.sh
# 或快速验证：
python eval.py --model ./output_mac_smollm3 --limit 100

# 3. DPO（在 SFT 基础上）
# 先在 dpo.sh 中将 MODEL_ID 改为 ./output_mac_smollm3
bash dpo.sh
# → 输出: ./output_mac_dpo/

# 4. 评测 DPO 效果（与 SFT 对比）
STAGE=dpo bash eval.sh
```
