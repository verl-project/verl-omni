# Conda 安装指南 (Ascend NPU)

Last updated: 08/26/2026

本文档基于 [`install.md`](install.md)，将官方的 `uv venv` 流程改写为 **conda** 环境管理流程，适用于 **Ascend NPU**（Atlas A2 / 910B、Atlas A3）环境。版本对齐参考官方 NPU 镜像 [`docker/Dockerfile.a2.npu`](../../docker/Dockerfile.a2.npu) 与 [`docker/Dockerfile.a3.npu`](../../docker/Dockerfile.a3.npu)。

> NVIDIA GPU 环境请参考 [`install_conda.md`](install_conda.md)。

## 环境要求

* **Python**: >= 3.10（本指南使用 3.12，与官方 NPU 镜像一致）
* **CANN**: >= 8.5.0（官方 NPU 镜像使用 9.0.0，推荐）
* **Ascend 驱动**: 已安装驱动与固件，宿主机 `npu-smi info` 可正常执行
* **conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
* **操作系统**: Linux x86_64 / aarch64（Ubuntu 22.04 推荐）

## 关键版本

以下版本与官方 NPU Docker 镜像保持一致，安装时请勿随意改动：

| 组件 | 版本 / commit | 来源 |
| --- | --- | --- |
| PyTorch | `2.10.0+cpu` | [PyTorch CPU index](https://download.pytorch.org/whl/cpu/) |
| torchvision / torchaudio | `0.25.0+cpu` / `2.10.0+cpu` | 同上 |
| torch-npu | `2.10.0.post2` | [昇腾 pip 源](https://mirrors.huaweicloud.com/ascend/repos/pypi) |
| vLLM | `0.27.0` | PyPI |
| vllm-ascend | `d5e9816065ede613327d93908f87fee9f5c47128` | [`.github/vllm_ascend_pin.txt`](../../.github/vllm_ascend_pin.txt) |
| vllm-omni | `444485650b19b792403b12976f1b0eeb2ac1451c` | [`.github/vllm_omni_pin.txt`](../../.github/vllm_omni_pin.txt) |
| verl | `c4b389adadc58ce51cb2b63e70df497ca166d77f` | [`.github/verl_pin.txt`](../../.github/verl_pin.txt) |
| triton-ascend | `3.2.1` | 昇腾 pip 源 |
| numpy | `1.26.4` | PyPI |
| ray | `>= 2.56.1` | PyPI |

> 仓库更新后请以 `.github/` 下各 `*_pin.txt` 文件中的 commit 为准。

## 0. 准备系统依赖与 CANN

vllm-ascend 需要从源码编译，先安装构建工具链与运行库（Ubuntu 为例）：

```bash
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    gcc g++ cmake make build-essential ninja-build \
    numactl libnuma-dev libjemalloc2
```

确认宿主机已安装 Ascend 驱动与 CANN 工具包（含 `ascend-toolkit` 与 `nnal/atb`），并能加载环境：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

CANN 安装方法请参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/CANNCommercial/)。

## 1. 克隆仓库

```bash
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

## 2. 创建 conda 环境

```bash
conda create -n verl-omni-npu python=3.12 -y
conda activate verl-omni-npu
```

> 不要通过 `conda install` 安装 PyTorch / vLLM / torch-npu，这些包统一由 `pip` 安装，以避免依赖冲突。

可选：配置 pip 镜像加速（国内网络推荐）：

```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade pip setuptools wheel packaging
```

## 3. 安装基础运行时依赖

```bash
pip install modelscope 'ray>=2.56.1' 'protobuf>3.20.0'
```

> `ray>=2.56.1` 是硬性要求：该版本起，申请 0 张 NPU 的 actor（如纯 CPU 的 reward worker）不再清除 `ASCEND_RT_VISIBLE_DEVICES`。

## 4. 安装 PyTorch (CPU) 与 torch-npu

Ascend 上 torch-npu 叠加在 **CPU 版** PyTorch 之上（与官方 NPU 镜像一致）：

```bash
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu/

pip install torch-npu==2.10.0.post2 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

快速自检：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('torch', torch.__version__, '| NPU', torch.npu.is_available())"
```

## 5. 安装 vLLM

```bash
pip install vllm==0.27.0
pip uninstall -y triton || true
```

NPU 后端由 vllm-ascend 提供，标准 `triton` 需卸载，随后替换为 `triton-ascend`（见下一步）。

> 官方 NPU 镜像从源码构建 vLLM（`VLLM_TARGET_DEVICE="empty" pip install -e ".[audio]"`），以完全避免拉入 CUDA 依赖。如 PyPI wheel 安装后依赖解析异常，可改用源码方式。

## 6. 编译安装 vllm-ascend

```bash
mkdir -p ~/deps && cd ~/deps
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout d5e9816065ede613327d93908f87fee9f5c47128
git submodule update --init --recursive

# 加载 CANN 编译 / 运行环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 按硬件型号设置（影响算子编译目标）
export SOC_VERSION=ascend910b1        # Atlas A2 / 910B
# export SOC_VERSION=ascend910_9391   # Atlas A3
export COMPILE_CUSTOM_KERNELS=1
export VLLM_BATCH_INVARIANT=1

pip install -e . \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/

# 用 triton-ascend 替换标准 triton
pip uninstall -y triton triton-ascend || true
pip install triton-ascend==3.2.1 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 7. 安装 vllm-omni

```bash
VLLM_OMNI_TARGET_DEVICE=npu pip install \
    "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@444485650b19b792403b12976f1b0eeb2ac1451c" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/
```

> `VLLM_OMNI_TARGET_DEVICE=npu` 是 NPU 环境的必填项，用于选择 NPU 后端依赖。

## 8. 安装 verl 与 verl-omni

回到 verl-omni 仓库根目录：

```bash
cd /path/to/verl-omni
pip install -e ".[train]" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/
```

`[train]` extra 会安装固定到 [`.github/verl_pin.txt`](../../.github/verl_pin.txt) 的 `verl`，以及 editable 模式的 `verl-omni` 本体。

## 9. 版本对齐（防止依赖漂移）

上述步骤的依赖解析可能改动 torch / numpy 版本，最后统一对齐（与官方 NPU 镜像一致）：

```bash
pip install --force-reinstall \
    torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu/
pip install --force-reinstall torch-npu==2.10.0.post2 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --extra-index-url https://download.pytorch.org/whl/cpu/
pip install --force-reinstall numpy==1.26.4
```

## 10. 配置环境变量（conda activate 钩子）

将 CANN 环境与运行时推荐配置写入 conda 激活脚本，每次 `conda activate verl-omni-npu` 自动生效：

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/ascend_cann.sh << 'EOF'
# CANN / ATB 运行环境
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && . /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && . /usr/local/Ascend/nnal/atb/set_env.sh

# vLLM / 训练运行时推荐配置（与官方 NPU 镜像一致）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
EOF
```

## 安装后验证

```bash
conda activate verl-omni-npu

npu-smi info
python -c "import torch; import torch_npu; print('torch', torch.__version__, '| NPU', torch.npu.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import verl; print('verl', verl.__version__)"
python -c "import verl_omni; print('VeRL-Omni ready')"
python -c "import torch; import torch_npu; print('NPU device count:', torch.npu.device_count())"
```

预期 `torch.npu.is_available()` 返回 `True`，且 device count 与可见 NPU 卡数一致。

## 可选依赖

| 用途 | 命令 | 说明 |
| --- | --- | --- |
| OCR reward | `pip install -e ".[ocr]"` | 安装 `Levenshtein`，FlowGRPO OCR 奖励所需 |
| 音频 / 全模态 | `pip install -e ".[audio]"` | 安装 `qwen-omni-utils` |
| 多模态训练 | `pip install qwen-vl-utils math-verify` | 视觉语言训练（如 MMK12） |
| 开发工具 | `pip install -e ".[dev]"` | `pytest`、`pre-commit` 等 |

## 运行注意事项（NPU）

* NPU 训练必须覆盖 attention backend：

  ```bash
  actor_rollout_ref.model.attn_backend=_native_npu
  ```

* 优先使用 NPU 专用训练脚本，如 `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh`；手动拼命令时记得加上上面的 Hydra override。
* 完整的 NPU FlowGRPO 训练流程参见 [`flowgrpo_quickstart_npu.md`](flowgrpo_quickstart_npu.md)。

## 常见问题

| 现象 | 原因与解决 |
| --- | --- |
| `import torch_npu` 报错或找不到 `libascendcl.so` | CANN 环境未加载。先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，或确认已按第 10 步配置 activate 钩子 |
| torch 被升级为 CUDA 版本 | 某些依赖解析时拉入了默认 torch。按第 9 步 `--force-reinstall` 恢复 CPU 版 + torch-npu |
| numpy 相关报错 | numpy 被升级到 2.x。执行 `pip install --force-reinstall numpy==1.26.4` |
| triton 与 triton-ascend 冲突 | `pip uninstall -y triton triton-ascend` 后重装 `triton-ascend==3.2.1` |
| 纯 CPU reward worker 看不到 NPU 设备 | ray 版本过低。执行 `pip install -U 'ray>=2.56.1'` |
| vllm-ascend 编译失败 | 确认已 source CANN 环境、`SOC_VERSION` 与硬件匹配、`git submodule update --init --recursive` 已执行 |

## conda 环境管理常用命令

```bash
# 查看环境列表
conda env list

# 激活 / 退出环境
conda activate verl-omni-npu
conda deactivate

# 删除环境（如需重建）
conda env remove -n verl-omni-npu

# 导出环境（仅记录，含 pip 包）
conda env export -n verl-omni-npu > environment.yml
```
