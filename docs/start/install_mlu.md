# Installation (Cambricon MLU)

Last updated: 10/29/2026

For NVIDIA GPU, see the {doc}`GPU installation guide <install>`. For Ascend NPU, see the {doc}`NPU installation guide <install_npu>`. For AMD GPU, see the {doc}`ROCm installation guide <install_rocm>`. For Intel GPU, see the {doc}`Intel GPU installation guide <install_xpu>`.

## Requirements

* **Python**: Version >= 3.11
* **Container**: A compatible `cambricon_release` Docker image,
  available from Cambricon engineers.

## Check the container environment

Activate `pytorch_infer` and check the preinstalled components:

```bash
source /torch/venv3/pytorch_infer/bin/activate

python - <<'PY'
from importlib.metadata import version

import ray
import torch
import torch_mlu
import torchaudio
import torchvision
import vllm
import vllm_mlu
import vllm_omni
from vllm.platforms import current_platform
from vllm_omni.platforms import current_omni_platform

for name in ("torch", "torch-mlu", "torchvision", "torchaudio", "ray", "vllm", "vllm-mlu", "vllm-omni"):
    print(name, version(name))
print("MLU available:", torch.mlu.is_available(), "devices:", torch.mlu.device_count())
print("vLLM platform:", current_platform.device_type)
print("vLLM-Omni platform:", current_omni_platform.device_type)
PY
```

All imports must succeed. MLU availability should be `True` with a nonzero
device count, and both vLLM and vLLM-Omni must report the `mlu` platform.
Resolve any environment issues with Cambricon engineers before continuing
with installation.

## Install

Activate the container's existing `pytorch_infer` environment and clone the
training projects. The `python -m pip` commands below install directly into
this activated environment.

```bash
source /torch/venv3/pytorch_infer/bin/activate
git clone https://github.com/verl-project/verl.git
git clone https://github.com/verl-project/verl-hardware-plugin.git
git clone https://github.com/verl-project/verl-omni.git
cd verl-omni
```

1. Check out the verl revision pinned by VeRL-Omni and the hardware-plugin revision:

```bash
git -C ../verl checkout "$(tr -d '[:space:]' < .github/verl_pin.txt)"
git -C ../verl-hardware-plugin checkout 549091f934278582c354e3f31033ecf7edc68b47
```

2. Install the project dependencies while preserving the container's
preinstalled MLU packages:

```bash
export MLU_CONSTRAINTS="$(mktemp /tmp/verl-omni-mlu-constraints.XXXXXX)"
export MLU_REQUIREMENTS="$(mktemp /tmp/verl-omni-mlu-requirements.XXXXXX)"

python - <<'PY'
import os
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

keep = {
    "torch", "torch-mlu", "torch-mlu-ops", "torchvision", "torchaudio", "torchcodec",
    "triton", "ray", "vllm", "vllm-mlu", "vllm-omni", "compressed-tensors",
    "safetensors", "verl", "verl-hardware-plugin",
}
constraints = []
for dist in distributions():
    name = canonicalize_name(dist.metadata["Name"])
    if name in keep:
        constraints.append(f"{name}=={dist.version}")
Path(os.environ["MLU_CONSTRAINTS"]).write_text("\n".join(constraints) + "\n")

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
provided = {"verl", "vllm", "vllm-omni"}
requirements = [
    requirement for requirement in project["dependencies"]
    if canonicalize_name(Requirement(requirement).name) not in provided
]
Path(os.environ["MLU_REQUIREMENTS"]).write_text("\n".join(requirements) + "\n")
PY

python -m pip install -c "$MLU_CONSTRAINTS" -r "$MLU_REQUIREMENTS"
```

3. Install verl, the hardware plugin, and VeRL-Omni from the prepared sources:

```bash
python -m pip install --no-deps -e ../verl -e ../verl-hardware-plugin -e .
```

`--no-deps` preserves the prepared source packages and the container's vLLM
builds. Use `-c "$MLU_CONSTRAINTS"` for subsequent dependency installations.

## Post-installation verification

Check the installed training packages and platform selection:

```bash
export VERL_PLATFORM=cambricon
python - <<'PY'
from importlib.metadata import version

import diffusers
import torch
import torch_mlu
import transfer_queue
import verl
import verl_hardware_plugin
import verl_omni
import vllm_omni
from verl.utils.device import get_device_name, get_nccl_backend
from vllm.platforms import current_platform
from vllm_omni.platforms import current_omni_platform

for name in ("verl", "verl-hardware-plugin", "verl-omni", "diffusers", "TransferQueue", "vllm-omni"):
    print(name, version(name))
print("verl:", get_device_name(), get_nccl_backend())
print("vLLM platform:", current_platform.device_type)
print("vLLM-Omni platform:", current_omni_platform.device_type)
PY
```

All imports must succeed. verl should report `mlu cncl`, and both vLLM and
vLLM-Omni should report `mlu`.

Continue with the
{doc}`MiniMax-H3 FlowGRPO recipe <../examples/minimax_h3/flowgrpo_trainer_minimax_h3>`
for checkpoint preparation, data, reward dependencies, and training.
