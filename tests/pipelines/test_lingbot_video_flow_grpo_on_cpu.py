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

"""CPU contract tests for LingBot Dense T2V FlowGRPO helpers and adapter."""

import contextlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl_omni.pipelines.lingbot_video_flow_grpo.common import (
    DEFAULT_NEGATIVE_PROMPT,
    apply_cfg,
    apply_prompt_template,
    caption_to_json,
    guidance_scale,
    shifted_sigmas,
    validate_t2v_dimensions,
)
from verl_omni.pipelines.lingbot_video_flow_grpo.diffusers_training_adapter import LingBotVideoDenseFlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase

# The rollout adapter pulls in vLLM-Omni / diffusers video pieces at import time.
# Its pure helpers (_sample_sde_window, dtype/device coalescing) are CPU-testable,
# so import it defensively and skip only those tests when the stack is absent.
try:
    from verl_omni.pipelines.lingbot_video_flow_grpo import vllm_omni_rollout_adapter as rollout_adapter
except Exception:  # noqa: BLE001 - optional rollout stack (vllm-omni) may be missing
    rollout_adapter = None

requires_rollout_adapter = pytest.mark.skipif(
    rollout_adapter is None, reason="rollout adapter requires vllm-omni + diffusers video stack"
)


def test_caption_serialization_and_template_are_officially_compact():
    caption = {"subject": "猫", "motion": ["walks", "turns"]}
    encoded = caption_to_json(caption)
    assert encoded == '{"subject":"猫","motion":["walks","turns"]}'
    assert encoded in apply_prompt_template(encoded)
    with pytest.raises(ValueError, match="structured JSON"):
        caption_to_json("a plain text prompt")


@pytest.mark.parametrize("height,width,num_frames", [(480, 832, 81), (480, 832, 121), (16, 16, 1)])
def test_t2v_dimensions_accept_valid_shapes(height, width, num_frames):
    validate_t2v_dimensions(height, width, num_frames)


@pytest.mark.parametrize("height,width,num_frames", [(480, 832, 120), (481, 832, 121), (480, 831, 121)])
def test_t2v_dimensions_reject_invalid_shapes(height, width, num_frames):
    with pytest.raises(ValueError):
        validate_t2v_dimensions(height, width, num_frames)


def test_shifted_sigmas_match_flow_matching_shift_contract():
    sigmas = shifted_sigmas(4, 3.0)
    assert sigmas.shape == (4,)
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == pytest.approx(0.5)
    assert all(left > right for left, right in zip(sigmas[:-1], sigmas[1:], strict=True))


def test_cfg_formula_matches_official_lingbot_rule():
    positive = torch.tensor([3.0, -1.0])
    negative = torch.tensor([1.0, 2.0])
    assert torch.equal(apply_cfg(positive, negative, 3.0), torch.tensor([7.0, -7.0]))


def test_dense_adapter_is_registered_without_optional_lingbot_package():
    cfg = SimpleNamespace(architecture="LingBotVideoPipeline", algorithm="flow_grpo", external_lib=None)
    assert DiffusionModelBase.get_class(cfg) is LingBotVideoDenseFlowGRPO


def test_adapter_builds_lingbot_transformer_inputs_and_cfg_pair():
    model_config = SimpleNamespace(pipeline=SimpleNamespace(guidance_scale=3.0, true_cfg_scale=1.0))
    latents = torch.randn(2, 3, 16, 2, 4, 4)
    timesteps = torch.tensor([[1000.0, 500.0, 0.0], [1000.0, 500.0, 0.0]])
    embeds = torch.randn(2, 5, 2560)
    mask = torch.ones(2, 5, dtype=torch.long)
    model_inputs, negative_model_inputs = LingBotVideoDenseFlowGRPO.prepare_model_inputs(
        None,
        model_config,
        latents,
        timesteps,
        embeds,
        mask,
        embeds + 1,
        mask,
        None,
        step=1,
    )
    assert torch.equal(model_inputs["hidden_states"], latents[:, 1])
    assert torch.equal(model_inputs["timestep"], torch.tensor([500.0, 500.0]))
    assert negative_model_inputs is not None
    assert torch.equal(negative_model_inputs["encoder_hidden_states"], embeds + 1)


def test_manual_lora_applies_and_deactivates_standard_linear_layers():
    from verl_omni.pipelines.lingbot_video_flow_grpo.manual_lora import ManualLinearLoRAManager

    transformer = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        transformer[0].weight.zero_()
    manager = ManualLinearLoRAManager(transformer)
    request = SimpleNamespace(
        lora_int_id=7,
        peft_config={"lora_alpha": 4},
        lora_tensors={
            "base_model.model.transformer.0.lora_A.default.weight": torch.tensor([[1.0, 2.0, 3.0]]),
            "base_model.model.transformer.0.lora_B.default.weight": torch.tensor([[5.0], [7.0]]),
        },
    )

    manager.add_adapter(request)
    assert manager.list_adapters() == [7]

    x = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.equal(transformer(x), torch.zeros(1, 2))
    manager.set_active_adapter(request, lora_scale=0.5)
    # alpha / rank * external_scale = 4 / 1 * 0.5 = 2; A(x)=6; B(A(x))=[30,42].
    assert torch.equal(transformer(x), torch.tensor([[60.0, 84.0]]))

    manager.set_active_adapter(None)
    assert torch.equal(transformer(x), torch.zeros(1, 2))
    assert manager.remove_adapter(7) is True
    assert manager.list_adapters() == []


def test_manual_lora_rejects_unmatched_rollout_modules():
    from verl_omni.pipelines.lingbot_video_flow_grpo.manual_lora import ManualLinearLoRAManager

    manager = ManualLinearLoRAManager(torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)))
    request = SimpleNamespace(
        lora_int_id=8,
        peft_config={"lora_alpha": 1},
        lora_tensors={
            "transformer.missing.lora_A.default.weight": torch.zeros(1, 3),
            "transformer.missing.lora_B.default.weight": torch.zeros(2, 1),
        },
    )
    with pytest.raises(ValueError, match="No LingBot nn.Linear LoRA tensors matched"):
        manager.add_adapter(request)


def test_custom_pipeline_lora_proxy_routes_worker_lifecycle_calls():
    from verl_omni.workers.rollout.vllm_rollout.utils import (
        _PipelineLoRAProxy,
        _supports_pipeline_lora,
        vLLMOmniColocateWorkerExtension,
    )

    class Pipeline:
        def __init__(self):
            self.calls = []
            self.adapters = []

        def add_lora(self, request):
            self.calls.append(("add", request))
            self.adapters.append(request.lora_int_id)
            return True

        def remove_lora(self, adapter_id):
            self.calls.append(("remove", adapter_id))
            self.adapters.remove(adapter_id)
            return True

        def list_loras(self):
            return self.adapters

        def pin_lora(self, adapter_id):
            self.calls.append(("pin", adapter_id))
            return adapter_id in self.adapters

        def set_active_lora(self, request, scale):
            self.calls.append(("active", request, scale))

    pipeline = Pipeline()
    request = SimpleNamespace(lora_int_id=42)
    assert _supports_pipeline_lora(pipeline)
    proxy = _PipelineLoRAProxy(pipeline)
    assert proxy.add_adapter(request)
    assert proxy.list_adapters() == [42]
    assert proxy.pin_adapter(42)
    proxy.set_active_adapter(request, 0.5)
    assert proxy.remove_adapter(42)
    assert pipeline.calls == [
        ("add", request),
        ("pin", 42),
        ("active", request, 0.5),
        ("remove", 42),
    ]

    class Worker:
        def _get_custom_lora_pipeline(self):
            return pipeline

    worker = Worker()
    vLLMOmniColocateWorkerExtension.init_lora_manager(worker)
    assert isinstance(worker.lora_manager, _PipelineLoRAProxy)


def test_async_server_reads_trajectory_payload_from_multimodal_output():
    """The consumer must read payload["trajectory"] via multimodal_output.

    vLLM-Omni main removed ``DiffusionOutput.custom_output`` (#4922); the engine
    copies ``output["payload"]["trajectory"]`` into
    ``OmniRequestOutput.multimodal_output["trajectory"]`` and the formatter never
    repopulates ``custom_output``. This test drives ``_process_output`` with the
    post-#4922 shape and asserts the training-facing keys survive.
    """

    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

    video = torch.rand(5, 3, 8, 8)
    trajectory = {
        "all_latents": torch.randn(1, 3, 4, 2, 1, 1),
        "all_log_probs": torch.randn(1, 2),
        "all_timesteps": torch.tensor([[750.0, 500.0]]),
        "prompt_embeds": torch.randn(1, 7, 16),
        "prompt_embeds_mask": torch.ones(1, 7, dtype=torch.long),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        # Formatter-facing duplicates that the consumer must drop.
        "latents": torch.randn(1, 3, 4, 2, 1, 1),
        "log_probs": torch.randn(1, 2),
        "timesteps": torch.tensor([[750.0, 500.0]]),
    }
    final_res = SimpleNamespace(
        images=[video],
        custom_output={},
        multimodal_output={"trajectory": trajectory, "metadata": {"trajectory": {"type": "denoising"}}},
        request_output=None,
    )
    server = SimpleNamespace(
        _ar_mode=False,
        global_steps=3,
        _map_stop_reason=lambda self_reason: "stop",
        _to_tensor=None,
    )

    result = vLLMOmniHttpServer._process_output(server, final_res, params=None, sampling_params={"logprobs": True})

    assert torch.equal(result.diffusion_output, video)
    assert torch.equal(result.log_probs, trajectory["all_log_probs"][0])
    extra = result.extra_fields
    assert torch.equal(extra["all_latents"], trajectory["all_latents"][0])
    assert torch.equal(extra["all_timesteps"], trajectory["all_timesteps"][0])
    assert torch.equal(extra["prompt_embeds"], trajectory["prompt_embeds"][0])
    assert extra["negative_prompt_embeds"] is None
    assert extra["global_steps"] == 3
    # The formatter-facing duplicate keys must not leak into training data.
    assert "latents" not in extra and "log_probs" not in extra and "timesteps" not in extra


# --------------------------------------------------------------------------- #
# common.py — caption serialization, sigma schedule, guidance resolution
# --------------------------------------------------------------------------- #


def test_caption_to_json_accepts_list_and_json_string_and_rejects_scalars():
    assert caption_to_json([{"a": 1}, "b"]) == '[{"a":1},"b"]'
    # A JSON string is parsed then re-serialized in the official compact form.
    assert caption_to_json('{"a": 1}') == '{"a":1}'
    with pytest.raises(TypeError, match="mapping, list"):
        caption_to_json(5)
    with pytest.raises(ValueError, match="structured JSON"):
        caption_to_json("plain text, not json")


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_shifted_sigmas_rejects_nonpositive_steps(bad_steps):
    with pytest.raises(ValueError, match="num_inference_steps"):
        shifted_sigmas(bad_steps, 3.0)


@pytest.mark.parametrize("bad_shift", [0.0, -2.0])
def test_shifted_sigmas_rejects_nonpositive_shift(bad_shift):
    with pytest.raises(ValueError, match="shift"):
        shifted_sigmas(4, bad_shift)


def test_default_negative_prompt_is_valid_compact_structured_json():
    parsed = json.loads(DEFAULT_NEGATIVE_PROMPT)
    assert "universal_negative" in parsed
    # The bundled default must already be in the official compact serialization.
    assert caption_to_json(parsed) == DEFAULT_NEGATIVE_PROMPT


def test_guidance_scale_resolves_from_attr_dict_and_true_cfg_fallback():
    assert guidance_scale(SimpleNamespace(guidance_scale=3.0)) == 3.0
    assert guidance_scale({"guidance_scale": 2.5}) == 2.5
    # Falls back to true_cfg_scale when guidance_scale is absent or None.
    assert guidance_scale(SimpleNamespace(guidance_scale=None, true_cfg_scale=4.0)) == 4.0
    assert guidance_scale({"true_cfg_scale": 5.0}) == 5.0
    # Neither present -> official default of 1.0 (CFG disabled).
    assert guidance_scale(SimpleNamespace()) == 1.0


# --------------------------------------------------------------------------- #
# diffusers_training_adapter.py — CFG gating, train-mode guard, sigma schedule
# --------------------------------------------------------------------------- #


def test_prepare_model_inputs_disables_cfg_when_guidance_not_positive():
    model_config = SimpleNamespace(pipeline=SimpleNamespace(guidance_scale=1.0, true_cfg_scale=1.0))
    latents = torch.randn(2, 3, 16, 2, 4, 4)
    timesteps = torch.tensor([[1000.0, 500.0, 0.0], [1000.0, 500.0, 0.0]])
    embeds = torch.randn(2, 5, 2560)
    mask = torch.ones(2, 5, dtype=torch.long)
    model_inputs, negative_model_inputs = LingBotVideoDenseFlowGRPO.prepare_model_inputs(
        None, model_config, latents, timesteps, embeds, mask, None, None, None, step=0
    )
    assert negative_model_inputs is None
    assert torch.equal(model_inputs["hidden_states"], latents[:, 0])
    assert torch.equal(model_inputs["timestep"], torch.tensor([1000.0, 1000.0]))


def test_prepare_model_inputs_requires_negatives_under_cfg():
    model_config = SimpleNamespace(pipeline=SimpleNamespace(guidance_scale=3.0, true_cfg_scale=1.0))
    latents = torch.randn(1, 2, 16, 2, 4, 4)
    timesteps = torch.tensor([[1000.0, 500.0]])
    embeds = torch.randn(1, 5, 2560)
    mask = torch.ones(1, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="negative prompt embeddings"):
        LingBotVideoDenseFlowGRPO.prepare_model_inputs(
            None, model_config, latents, timesteps, embeds, mask, None, None, None, step=0
        )


def test_configure_train_mode_accepts_gradient_checkpointing_models():
    module = torch.nn.Linear(2, 2)
    module._supports_gradient_checkpointing = True

    LingBotVideoDenseFlowGRPO.configure_train_mode(module)


def test_patch_config_save_pretrained_writes_json(tmp_path):
    class _Config(dict):
        pass

    module = torch.nn.Module()
    module.config = _Config({"hidden_size": 4, "num_experts": 0})

    LingBotVideoDenseFlowGRPO._patch_config_save_pretrained(module)
    module.config.save_pretrained(tmp_path)

    assert (tmp_path / "config.json").read_text() == '{\n    "hidden_size": 4,\n    "num_experts": 0\n}'


def test_flash_attn_interface_compat_cpu_fallback(monkeypatch):
    from verl_omni.pipelines.lingbot_video_flow_grpo.common import install_flash_attn_interface_compat

    monkeypatch.delitem(sys.modules, "flash_attn_interface", raising=False)
    install_flash_attn_interface_compat(prefer_fa3=False)

    import flash_attn_interface

    q = torch.randn(5, 2, 4, requires_grad=True)
    k = torch.randn(5, 2, 4, requires_grad=True)
    v = torch.randn(5, 2, 4, requires_grad=True)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    out = flash_attn_interface.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        3,
        3,
        dropout_p=0.0,
        causal=False,
    )

    assert out.shape == q.shape
    out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


class _RecordingScheduler:
    def __init__(self):
        self.calls = {}

    def set_timesteps(self, num_steps, device, sigmas):
        self.calls = {"num_steps": num_steps, "device": device, "sigmas": sigmas}


def test_set_timesteps_uses_shifted_sigma_schedule_from_pipeline_config():
    scheduler = _RecordingScheduler()
    model_config = SimpleNamespace(pipeline=SimpleNamespace(num_inference_steps=4, shift=3.0))
    LingBotVideoDenseFlowGRPO.set_timesteps(scheduler, model_config, device="cpu")
    assert scheduler.calls["num_steps"] == 4
    np.testing.assert_allclose(scheduler.calls["sigmas"], shifted_sigmas(4, 3.0))


def test_set_timesteps_falls_back_to_official_defaults():
    scheduler = _RecordingScheduler()
    model_config = SimpleNamespace(pipeline=SimpleNamespace())
    LingBotVideoDenseFlowGRPO.set_timesteps(scheduler, model_config, device="cpu")
    # 40-step, shift-3 is the official baseline used when config omits them.
    assert scheduler.calls["num_steps"] == 40
    np.testing.assert_allclose(scheduler.calls["sigmas"], shifted_sigmas(40, 3.0))


def test_time_embedder_dtype_hook_casts_float_projection_to_weight_dtype():
    from diffusers.models.embeddings import TimestepEmbedding

    module = SimpleNamespace(time_embedder=TimestepEmbedding(2, 4).to(torch.bfloat16))
    LingBotVideoDenseFlowGRPO._patch_time_embedder_input_dtype(module)

    output = module.time_embedder(torch.ones(1, 2, dtype=torch.float32))

    assert output.dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# manual_lora.py — scaling rules, prefix normalization, error handling
# --------------------------------------------------------------------------- #


def _zeroed_linear_manager():
    from verl_omni.pipelines.lingbot_video_flow_grpo.manual_lora import ManualLinearLoRAManager

    transformer = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        transformer[0].weight.zero_()
    return ManualLinearLoRAManager(transformer), transformer


def _linear_lora_request(adapter_id, lora_a, lora_b, peft_config):
    return SimpleNamespace(
        lora_int_id=adapter_id,
        peft_config=peft_config,
        lora_tensors={
            "base_model.model.transformer.0.lora_A.default.weight": lora_a,
            "base_model.model.transformer.0.lora_B.default.weight": lora_b,
        },
    )


def test_manual_lora_rslora_scales_by_sqrt_rank():
    manager, transformer = _zeroed_linear_manager()
    # rank 2: A = I(2), B = ones(1, 2) so B(A(x)) = sum(x).
    request = _linear_lora_request(11, torch.eye(2), torch.ones(1, 2), {"lora_alpha": 4, "use_rslora": True})
    manager.add_adapter(request)
    manager.set_active_adapter(request)
    x = torch.tensor([[1.0, 1.0]])
    # A(x)=[1,1]; B(A(x))=2; rslora scale = alpha/sqrt(rank) = 4/sqrt(2).
    assert transformer(x).item() == pytest.approx(2.0 * (4.0 / math.sqrt(2)))


def test_manual_lora_alpha_pattern_overrides_lora_alpha():
    manager, transformer = _zeroed_linear_manager()
    request = _linear_lora_request(12, torch.eye(2), torch.ones(1, 2), {"lora_alpha": 4, "alpha_pattern": {"0": 6}})
    manager.add_adapter(request)
    manager.set_active_adapter(request)
    x = torch.tensor([[1.0, 1.0]])
    # module "0" matches alpha_pattern -> alpha 6, rank 2 -> scale 3; delta = 2 * 3.
    assert transformer(x).item() == pytest.approx(6.0)


def test_manual_lora_zero_scale_deactivates_and_set_active_auto_adds():
    manager, transformer = _zeroed_linear_manager()
    request = _linear_lora_request(13, torch.eye(2), torch.ones(1, 2), {"lora_alpha": 2})
    # set_active_adapter auto-adds an unknown request...
    manager.set_active_adapter(request)
    assert manager.list_adapters() == [13]
    assert manager.active_adapter_id == 13
    x = torch.tensor([[1.0, 1.0]])
    assert transformer(x).item() == pytest.approx(2.0)  # scale = 2/2 = 1; delta = 2
    # ...and a zero scale deactivates without removing the adapter.
    manager.set_active_adapter(request, lora_scale=0.0)
    assert manager.active_adapter_id is None
    assert manager.list_adapters() == [13]
    assert transformer(x).item() == pytest.approx(0.0)


def test_manual_lora_add_adapter_rejects_empty_incomplete_and_mismatched_tensors():
    manager, _ = _zeroed_linear_manager()
    with pytest.raises(ValueError, match="in-memory tensors"):
        manager.add_adapter(SimpleNamespace(lora_int_id=1, peft_config={}, lora_tensors={}))
    with pytest.raises(ValueError, match="Incomplete LoRA"):
        manager.add_adapter(
            SimpleNamespace(
                lora_int_id=2,
                peft_config={},
                lora_tensors={"transformer.0.lora_A.default.weight": torch.eye(2)},
            )
        )
    with pytest.raises(ValueError, match="do not match Linear"):
        manager.add_adapter(
            SimpleNamespace(
                lora_int_id=3,
                peft_config={},
                lora_tensors={
                    "transformer.0.lora_A.default.weight": torch.ones(2, 5),  # in=5 != 2
                    "transformer.0.lora_B.default.weight": torch.ones(1, 2),
                },
            )
        )


def test_manual_lora_normalizes_wrapper_prefixes_and_base_layer():
    manager, _ = _zeroed_linear_manager()
    # FSDP/PEFT wrapper prefixes plus a `.base_layer` segment must resolve to "0".
    request = SimpleNamespace(
        lora_int_id=21,
        peft_config={"lora_alpha": 2},
        lora_tensors={
            "_fsdp_wrapped_module.base_model.model.transformer.0.base_layer.lora_A.default.weight": torch.eye(2),
            "_fsdp_wrapped_module.base_model.model.transformer.0.base_layer.lora_B.default.weight": torch.ones(1, 2),
        },
    )
    assert manager.add_adapter(request)
    assert manager.list_adapters() == [21]


def test_manual_lora_pin_and_remove_report_membership():
    manager, _ = _zeroed_linear_manager()
    request = _linear_lora_request(31, torch.eye(2), torch.ones(1, 2), {"lora_alpha": 2})
    assert manager.pin_adapter(31) is False  # not added yet
    manager.add_adapter(request)
    assert manager.pin_adapter(31) is True
    assert manager.remove_adapter(31) is True
    assert manager.remove_adapter(31) is False  # already gone
    assert manager.list_adapters() == []


# --------------------------------------------------------------------------- #
# vllm_omni_rollout_adapter.py — SDE window sampling and dtype/device helpers
# --------------------------------------------------------------------------- #


@requires_rollout_adapter
def test_sample_sde_window_covers_full_range_and_bounded_windows():
    sample = rollout_adapter.LingBotVideoPipelineWithLogProb._sample_sde_window
    device = torch.device("cpu")
    # No window size -> capture the whole trajectory.
    assert sample(None, None, 10, None, device) == (0, 10)
    # A size-2 window constrained to [0, 2] can only start at 0.
    assert sample(2, [0, 2], 10, None, device) == (0, 2)
    gen = torch.Generator(device="cpu").manual_seed(0)
    start, end = sample(3, [1, 8], 10, gen, device)
    assert end - start == 3
    assert 1 <= start and end <= 8


@requires_rollout_adapter
@pytest.mark.parametrize("size", [0, 11])
def test_sample_sde_window_rejects_invalid_size(size):
    sample = rollout_adapter.LingBotVideoPipelineWithLogProb._sample_sde_window
    with pytest.raises(ValueError, match="window size"):
        sample(size, None, 10, None, torch.device("cpu"))


@requires_rollout_adapter
def test_sample_sde_window_rejects_bad_range():
    sample = rollout_adapter.LingBotVideoPipelineWithLogProb._sample_sde_window
    device = torch.device("cpu")
    with pytest.raises(ValueError, match=r"\[start, end\]"):
        sample(2, [1, 2, 3], 10, None, device)
    with pytest.raises(ValueError, match="cannot fit"):
        sample(5, [0, 4], 10, None, device)  # a size-5 window cannot fit in [0, 4]


@requires_rollout_adapter
def test_rollout_module_helpers_default_without_parameters():
    empty = torch.nn.Module()
    assert rollout_adapter._module_dtype(empty) == torch.float32
    assert rollout_adapter._module_device(empty) == torch.device("cpu")
    assert rollout_adapter._coalesce(None, 7) == 7
    assert rollout_adapter._coalesce(3, 7) == 3
    # CPU / fp32 path is a no-op context, never a CUDA autocast.
    assert isinstance(rollout_adapter._autocast(torch.device("cpu"), torch.float32), contextlib.nullcontext)
