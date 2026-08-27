# Conda 安装指南 (NVIDIA GPU)

Last updated: 08/26/2026

本文档基于 [`install.md`](install.md)，将官方的 `uv venv` 流程改写为 **conda** 环境管理流程。适用于 NVIDIA GPU 环境。

> Ascend NPU 环境的 conda 安装指南见 [`install_conda_npu.md`](install_conda_npu.md)。

## 环境要求

* **Python**: >= 3.10（本指南使用 3.12）
* **CUDA**: >= 12.8
* **NVIDIA 驱动**: 需支持所用 CUDA 版本
* **conda**: Miniconda 或 Anaconda（推荐 [Miniconda](https://docs.conda.io/projects/miniconcda/en/latest/)）

## 安装步骤

### 0. 克隆仓库

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

### 1. 创建 conda 环境

```bash
conda create -n verl-omni python=3.12 -y
conda activate verl-omni
```

> 不要在 conda 环境中通过 `conda install` 安装 PyTorch / vLLM，这些包由后续步骤的 `uv pip`（或 `pip`）统一解析，以避免依赖冲突。

### 2. 安装 uv（推荐）

官方安装流程依赖 `uv` 的 `--torch-backend=auto` 来自动选择匹配 CUDA 的 PyTorch wheel。在 conda 环境中安装 `uv` 即可复用同一套命令：

```bash
pip install uv
```

> 如果你无法使用 `uv`，可跳到 [附录：纯 pip 安装](#附录纯-pip-安装可选) 查看等效命令。

### 3. 安装平台后端 (GPU)

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
```

该命令会安装 `vllm==0.27.0`（CUDA 版 PyTorch 栈）与 `kernels==0.16.0`（FA3 后端）及 `liger-kernel`。

### 4. 安装 vLLM-Omni 与 VeRL-Omni

```bash
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train]"
```

第一步安装 `vllm-omni`（固定到 `.github/vllm_omni_pin.txt` 中的 commit），第二步安装 `verl` 与 `verl-omni` 本体（editable 模式）。

## 可选依赖

| 用途 | 命令 | 说明 |
| --- | --- | --- |
| OCR reward | `uv pip install -e ".[ocr]"` | 安装 `Levenshtein`，FlowGRPO OCR 奖励所需 |
| 开发工具 | `uv pip install -e ".[dev]"` | `pytest`、`pre-commit`、`Levenshtein` 等 |
| 多模态训练 | `pip install qwen-vl-utils math-verify` | 视觉语言训练（如 MMK12） |

### Flash Attention 3

`[gpu]` extra 已包含 `kernels==0.16.0`，用于 Diffusers actor 的 FA3 (`attn_backend=_flash_3_varlen_hub`)。默认 actor 与 rollout 共用 Hub kernel 后端：

```bash
actor_rollout_ref.model.attn_backend=_flash_3_varlen_hub
actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN_3_HUB
```

若运行时缺少 FA3 依赖，训练会自动回退到 native/SDPA。

## 可选引擎后端：VeOmni

VeRL-Omni 默认使用 **FSDP2** 作为训练引擎。扩散模型训练也可切换到 [VeOmni](https://github.com/ByteDance/VeOmni)。

VeOmni 0.1.11 的 `[gpu]` extra 固定 `torch==2.9.1+cu129`，与 `vllm==0.27.0` 拉入的 torch 版本冲突。需以 `--no-deps` 安装，避免破坏已有 torch/vllm 栈：

```bash
uv pip install veomni==0.1.11 --no-deps
uv pip install torchcodec librosa soundfile av audioread
```

验证导入：

```bash
python -c "import veomni; print('veomni', veomni.__version__)"
python -c "from veomni.distributed.offloading import load_model_to_gpu, load_optimizer, offload_model_to_cpu, offload_optimizer; print('VeOmni offloading helpers OK')"
```

## 安装后验证

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import vllm_omni; print('vllm-omni OK')"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
```

确认 `torch.cuda.is_available()` 返回 `True`：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

## conda 环境管理常用命令

```bash
# 查看环境列表
conda env list

# 激活 / 退出环境
conda activate verl-omni
conda deactivate

# 删除环境（如需重建）
conda env remove -n verl-omni

# 导出环境（仅记录，含 pip 包）
conda env export -n verl-omni > environment.yml
```

## 附录：纯 pip 安装（可选）

若不使用 `uv`，需手动指定 PyTorch 的 CUDA index，再安装其余依赖：

```bash
conda create -n verl-omni python=3.12 -y
conda activate verl-omni

# 1. 安装 CUDA 版 PyTorch（CUDA 12.8）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 安装 GPU 后端（vllm / kernels / liger-kernel）
pip install -e ".[gpu]"

# 3. 安装 vLLM-Omni 与 VeRL-Omni
pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
pip install -e ".[train]"
```

> 注意：纯 pip 方式下，`vllm==0.27.0` 可能尝试重新解析 torch 版本。若出现版本回退，优先按 `--torch-backend=auto`（即 uv 方式）安装。
