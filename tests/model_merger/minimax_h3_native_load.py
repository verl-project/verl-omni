# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Load a merged tiny MiniMax H3 artifact with the native vLLM-Omni DiT loader."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main() -> None:
    """Instantiate the native model and require every published tensor to load."""
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    config = json.loads((args.artifact / "transformer/config.json").read_text())
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
            local_rank=-1,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
        try:
            diffusion_config = OmniDiffusionConfig(
                model=str(args.artifact),
                tf_model_config=TransformerConfig.from_dict(config),
                dtype=torch.float32,
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=1),
            )
            # The CPU platform deliberately has no runtime attention backend.
            # This helper verifies native checkpoint loading, not backend dispatch.
            with patch.object(selector, "_cached_get_backend_cls", return_value=SDPABackend):
                model = MiniMaxH3DiTModel(diffusion_config)
            published = set()

            def weights():
                for path in sorted((args.artifact / "transformer").glob("*.safetensors")):
                    with safe_open(path, framework="pt", device="cpu") as archive:
                        for name in archive.keys():
                            published.add(name)
                            yield name, archive.get_tensor(name)

            loaded = model.load_weights(weights())
            if loaded != published:
                raise AssertionError(
                    f"Native H3 loader coverage mismatch: missing={sorted(published - loaded)}, "
                    f"unexpected={sorted(loaded - published)}"
                )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
