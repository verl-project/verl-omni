# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl_omni/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "cachetools",
    "codetiming",
    "datasets",
    "diffusers",
    "dill",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers",
    "wandb",
    "packaging>=20.0",
    "tensorboard",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy", "pytest-asyncio", "pytest-rerunfailures"]
GPU_REQUIRES = ["flash-attn"]
VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.8.5,<=0.12.0"]

extras_require = {
    "test": TEST_REQUIRES,
    "gpu": GPU_REQUIRES,
    "vllm": VLLM_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl-omni",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where=".", include=["verl_omni", "verl_omni.*"]),
    url="https://github.com/verl-project/verl-omni",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="yhuangch@cse.ust.hk",
    description="verl-omni: Easy, fast, and stable RL training for diffusion and omni-modality models",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl_omni": [
            "trainer/config/*.yaml",
            "trainer/config/*/*.yaml",
            "trainer/config/*/*/*.yaml",
        ],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
