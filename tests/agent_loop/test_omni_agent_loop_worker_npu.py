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
"""Ascend port of the GPU OmniAgentLoopWorker image_gen tool smoke.

Reuses ``run_omni_agent_loop_image_gen`` from the GPU module so the manager
binding, hermes multi-turn tool loop, rollout stamps, agentic metrics and
trajectory dump assertions stay identical; only the checkpoint head dimension,
Ray runtime env and Ascend engine knobs are adapted.
"""

import os
from pathlib import Path

import pytest


def _npu_ray_runtime_env(requested: int) -> dict:
    """Ray runtime env mirroring the GPU hook plus Ascend visibility settings."""
    from tests.agent_loop.test_omni_agent_loop_worker_gpu import _register_vllm_omni_rollout_on_ray_worker

    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES") or ",".join(str(i) for i in range(requested))
    return {
        "worker_process_setup_hook": _register_vllm_omni_rollout_on_ray_worker,
        "env_vars": {
            "DEVICE_NAME": "npu",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_LOGGING_LEVEL": "INFO",
            "PYTHONPATH": os.pathsep.join(p for p in (os.path.abspath("."), os.environ.get("PYTHONPATH", "")) if p),
            "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
            "ASCEND_VISIBLE_DEVICES": os.getenv("ASCEND_VISIBLE_DEVICES") or visible_devices,
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_NPU_SOCKET_PORT_RANGE": os.getenv("HCCL_NPU_SOCKET_PORT_RANGE") or "auto",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        },
    }


@pytest.fixture
def npu_omni_config(tmp_path_factory):
    """Build the thinker-only Qwen3-Omni tool-loop config for Ascend."""
    import torch
    import torch_npu  # noqa: F401
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from tests.special_e2e.build_qwen3_omni_tiny_random import ensure_tiny_qwen3_omni_checkpoint

    if not torch.npu.is_available():
        pytest.fail("These smoke tests require Ascend NPUs and torch-npu.")
    available = torch.npu.device_count()
    try:
        requested = int(os.getenv("NUM_NPUS", str(available)))
    except ValueError:
        pytest.fail("NUM_NPUS must be an integer.")
    if requested < 2 or requested > available or requested % 2:
        pytest.fail(f"TP=2 requires an even 2 <= NUM_NPUS <= visible NPU count ({available}), got {requested}.")
    tp_size = min(2, requested)

    # Keep the Ascend head dimension separate from the GPU fixture's cache.
    model_path = ensure_tiny_qwen3_omni_checkpoint(
        os.path.expanduser(os.getenv("QWEN3_OMNI_MODEL_PATH", "~/.cache/verl-omni-npu-smoke/Qwen3-Omni-Tool")),
        skip_if_exists=True,
        head_dim=128,
    )
    run_root = Path(tmp_path_factory.mktemp("agentic_npu_smoke"))

    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="omni_trainer")

    with open_dict(config):
        config.trainer.device = "npu"
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.model.tokenizer_path = model_path
        config.actor_rollout_ref.model.trust_remote_code = True
        # Thinker-only strip mirrors GSPO smoke (talker/visual unused for tool text loop).
        config.actor_rollout_ref.model.exclude_modules = (
            ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"
        )

        config.actor_rollout_ref.rollout.name = "vllm_omni"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.35
        config.actor_rollout_ref.rollout.max_num_seqs = 4
        config.actor_rollout_ref.rollout.load_format = "safetensors"
        config.actor_rollout_ref.rollout.enable_prefix_caching = False
        config.actor_rollout_ref.rollout.prompt_length = 256
        config.actor_rollout_ref.rollout.response_length = 256
        config.actor_rollout_ref.rollout.max_model_len = 512
        config.actor_rollout_ref.rollout.temperature = 0.8
        config.actor_rollout_ref.rollout.calculate_log_probs = True
        config.actor_rollout_ref.rollout.enforce_eager = True

        if "engine_kwargs" not in config.actor_rollout_ref.rollout:
            config.actor_rollout_ref.rollout.engine_kwargs = {}
        if "vllm_omni" not in config.actor_rollout_ref.rollout.engine_kwargs:
            config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {}
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode = "ar"
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name = "qwen3_omni_moe"
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config = {"weight_nz_mode": 0}
        # Text-only tool loop never consumes the tiny model's mm towers.
        config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt = {
            "image": 0,
            "video": 0,
            "audio": 0,
        }

        config.actor_rollout_ref.rollout.multi_turn.enable = True
        config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 4
        config.actor_rollout_ref.rollout.multi_turn.max_user_turns = 2
        config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
        config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 512
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        # Leave function_tool_path unset — OmniAgentLoopWorker fills it under the gate.

        config.actor_rollout_ref.rollout.agent.num_workers = 1
        config.actor_rollout_ref.rollout.agent.default_agent_loop = "image_gen_tool_agent"
        # Standalone LLMServerManager.create() requires rollout.nnodes > 0 (not trainer.nnodes).
        config.actor_rollout_ref.rollout.nnodes = 1

        config.data.max_prompt_length = 256
        config.data.max_response_length = 256

        config.trainer.n_gpus_per_node = requested
        config.trainer.nnodes = 1
        config.trainer.logger = ["console"]
        config.trainer.project_name = "verl_omni_npu_smoke"
        config.trainer.experiment_name = "omni_agent_loop_worker"
        config.trainer.default_local_dir = str(run_root / "ckpt")

        # Reward manager is unused for this generate_sequences-only smoke, but keep naive
        # so accidental custom agentic_reward wiring does not hit VisualRewardManager.
        if hasattr(config, "reward") and hasattr(config.reward, "reward_manager"):
            config.reward.reward_manager.name = "naive"

        config.agentic_image_gen.vllm_omni_url = ""
        config.agentic_image_gen.qwen_image_url = ""
        config.agentic_image_gen.vllm_url = ""
        config.agentic_image_gen.force_first_generate = True
        config.agentic_image_gen.force_first_warmup_steps = 100
        config.agentic_image_gen.force_first_end_step = 200
        config.agentic_image_gen.force_reflection_after_judge = False
        config.agentic_image_gen.e2e_root = str(run_root)

    previous_device_name = os.environ.get("DEVICE_NAME")
    os.environ["DEVICE_NAME"] = "npu"
    try:
        yield config, run_root
    finally:
        if previous_device_name is None:
            os.environ.pop("DEVICE_NAME", None)
        else:
            os.environ["DEVICE_NAME"] = previous_device_name


def test_omni_agent_loop_worker_image_gen_generate_sequences_npu(npu_omni_config):
    """Preserve the GPU tool agent-loop wiring, stamp, metric and dump assertions."""
    from tests.agent_loop.test_omni_agent_loop_worker_gpu import run_omni_agent_loop_image_gen

    config, _run_root = npu_omni_config
    run_omni_agent_loop_image_gen(config, _npu_ray_runtime_env(config.trainer.n_gpus_per_node))
