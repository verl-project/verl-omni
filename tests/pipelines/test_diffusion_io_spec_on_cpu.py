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
"""Every registered diffusion pipeline must declare its rollout media outputs.

The diffusion rollout strategy and (later) the reward/trainer consumers read
the produced modality from the adapter-declared ``DiffusionIOSpec`` instead of
inferring it from tensor rank, so a missing declaration is a bug.
"""

import pytest

pytest.importorskip("verl_omni.pipelines.model_base")
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

# (adapter module, architecture, algorithm, primary modality)
_PRIMARY_MODALITY = [
    (
        "verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter",
        "OmniBagelForConditionalGeneration",
        "flow_grpo",
        "image",
    ),
    ("verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter", "BooguImagePipeline", "flow_grpo", "image"),
    ("verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter", "QwenImagePipeline", "flow_grpo", "image"),
    (
        "verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter",
        "QwenImagePipeline",
        "diffusion_nft",
        "image",
    ),
    ("verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter", "QwenImagePipeline", "dpo", "image"),
    # dual_grpo / mix_grpo subclass QwenImagePipelineWithLogProb and inherit the spec.
    ("verl_omni.pipelines.qwen_image_dual_grpo.vllm_omni_rollout_adapter", "QwenImagePipeline", "dual_grpo", "image"),
    ("verl_omni.pipelines.qwen_image_mix_grpo.vllm_omni_rollout_adapter", "QwenImagePipeline", "mix_grpo", "image"),
    (
        "verl_omni.pipelines.qwen_image_edit_flow_grpo.vllm_omni_rollout_adapter",
        "QwenImageEditPlusPipeline",
        "flow_grpo",
        "image",
    ),
    ("verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter", "StableDiffusion3Pipeline", "flow_grpo", "image"),
    ("verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter", "WanPipeline", "dance_grpo", "video"),
    ("verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter", "LTX2Pipeline", "flow_grpo", "video"),
    ("verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter", "MiniMaxH3Pipeline", "flow_grpo", "video"),
    (
        "verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter",
        "MiniMaxH3Pipeline",
        "diffusion_nft",
        "video",
    ),
]

# (architecture, algorithm, declared audio sample rate) for joint video/audio pipelines.
_JOINT_AUDIO_SAMPLE_RATE = [
    ("verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter", "MiniMaxH3Pipeline", "flow_grpo", 32000),
    (
        "verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter",
        "MiniMaxH3Pipeline",
        "diffusion_nft",
        32000,
    ),
    ("verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter", "LTX2Pipeline", "flow_grpo", 24000),
]


@pytest.mark.parametrize(("module", "architecture", "algorithm", "modality"), _PRIMARY_MODALITY)
def test_registered_pipeline_declares_primary_modality(module, architecture, algorithm, modality):
    pytest.importorskip(module)
    cls = VllmOmniPipelineBase.get_class(architecture, algorithm)
    assert cls is not None, f"{architecture}/{algorithm} is not registered"
    spec = getattr(cls, "diffusion_io_spec", None)
    assert spec is not None, f"{architecture}/{algorithm} must declare diffusion_io_spec"
    assert spec.primary.modality == modality


@pytest.mark.parametrize(("module", "architecture", "algorithm", "sample_rate"), _JOINT_AUDIO_SAMPLE_RATE)
def test_joint_pipeline_declares_audio_stream(module, architecture, algorithm, sample_rate):
    pytest.importorskip(module)
    cls = VllmOmniPipelineBase.get_class(architecture, algorithm)
    spec = getattr(cls, "diffusion_io_spec", None)
    assert spec is not None
    audio = next((stream for stream in spec.auxiliary if stream.modality == "audio"), None)
    assert audio is not None, f"{architecture}/{algorithm} must declare an auxiliary audio stream"
    assert audio.sample_rate == sample_rate
