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
"""Exercise H3 Image training transport, adjacent PDD states and the policy gradient."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minimax_h3_flow_grpo.common import configure_flow_scheduler
from verl_omni.pipelines.minimax_h3_image_flow_grpo import MiniMaxH3ImageFlowGRPO as Actor
from verl_omni.pipelines.minimax_h3_image_flow_grpo import MiniMaxH3ImageFlowGRPOPipeline as Rollout
from verl_omni.pipelines.minimax_h3_image_flow_grpo import vllm_omni_rollout_adapter as rollout_module
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss

GRID = [1.0, 0.96752, 0.88755, 0.75, 0.53277, 0.23201, 0.0]


@pytest.mark.parametrize("sde_type", ["cps", "sde"])
def test_rollout_replay_and_gradient(monkeypatch, sde_type):
    assert DiffusionModelBase.get_class_by_name("MiniMaxH3ImagePipeline", "flow_grpo") is Actor
    assert VllmOmniPipelineBase.get_class("MiniMaxH3ImagePipeline", "flow_grpo") is Rollout
    native = rollout_module.pipeline_minimax_h3_image
    # Stub only the backbone, native RoPE cache and image decoder.
    monkeypatch.setattr(native.MiniMaxH3DenoiseBranch, "prepare_rope_table", lambda *args: None)
    monkeypatch.setattr(
        native.MiniMaxH3DenoiseBranch,
        "forward_kwargs",
        lambda self, **kwargs: dict(kwargs, skip_mask_out_condition=False),
    )
    target_noise = torch.zeros(4, 96)

    def denoise(self, request):
        rows = self._denoise(
            target_noise=target_noise,
            hidden=torch.ones(3, 8),
            text_tags=torch.ones(3, dtype=torch.long),
            grids=[],
            anchors=target_noise[:0],
            latent_h=4,
            latent_w=4,
            num_steps=6,
        )
        from vllm_omni.diffusion.data import DiffusionOutput

        return DiffusionOutput(output=rows.unsqueeze(0))

    monkeypatch.setattr(native.MiniMaxH3ImagePipeline, "forward", denoise)
    p = object.__new__(Rollout)
    torch.nn.Module.__init__(p)
    p.device = torch.device("cpu")

    def transformer(pdd_step, video_rows, *, skip_mask_out_condition, **kwargs):
        assert skip_mask_out_condition is True
        return torch.full_like(video_rows, 0.01 * pdd_step), None

    p.transformer = transformer
    p.pdd_sigmas = GRID
    p._trajectory = p._sampling = p._final_rows = None
    sampling = SimpleNamespace(
        seed=53,
        output_type="latent",
        generator_device="cpu",
        max_sequence_length=8,
        extra_args={"sde_window_range": [1, 3], "sde_window_size": 2, "noise_level": 0.8, "sde_type": sde_type},
    )
    request = SimpleNamespace(
        sampling_params=sampling, prompts=["image"], request_id="test", is_dummy_run=lambda: False
    )
    result = p.forward(request)
    batch = dict(result.output["metadata"]["rl"], **result.output["metadata"]["prompt_embeddings"])
    batch["all_latents"] = result.trajectory_latents
    times = result.trajectory_timesteps
    assert batch["all_latents"].shape == (1, 3, 4, 96)
    assert batch["pdd_step_indices"].tolist() == [[1, 2]]
    assert batch["prompt_embeds_mask"].tolist() == [[1, 1, 1]]
    assert p._trajectory is None and p._sampling is None

    # Score exactly the actions sampled above, then change the policy and score them again.
    class TestActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, hidden_states, pdd_step_index, **kwargs):
            return (torch.ones_like(hidden_states) * (0.01 * pdd_step_index + self.offset),)

    model = TestActor()
    scheduler = FlowMatchSDEDiscreteScheduler()
    configure_flow_scheduler(scheduler, native.minimax_h3_image_run_sigmas(GRID), "cpu")
    cfg = SimpleNamespace(algo=SimpleNamespace(sde_type=sde_type, noise_level=0.8))
    for k in (0, 1):
        ins, neg = Actor.prepare_model_inputs(
            model,
            cfg,
            batch["all_latents"],
            times,
            batch["prompt_embeds"],
            batch["prompt_embeds_mask"],
            None,
            None,
            batch,
            k,
        )
        score, *_ = Actor.forward_and_sample_previous_step(model, scheduler, cfg, ins, neg, batch, k)
        old = result.trajectory_log_probs[:, k]
        torch.testing.assert_close(score, old)
        loss, metrics = FlowGRPOLoss.compute_loss(
            old_log_prob=old,
            log_prob=score,
            advantages=torch.ones_like(old),
            config=SimpleNamespace(diffusion_loss=SimpleNamespace(adv_clip_max=5, clip_ratio=0.2)),
        )
        assert all(math.isfinite(value) for value in metrics.values())
        loss.backward()
    assert torch.isfinite(model.offset.grad) and model.offset.grad.abs() > 0
    with torch.no_grad():
        model.offset.add_(0.15)
    changed, *_ = Actor.forward_and_sample_previous_step(model, scheduler, cfg, ins, neg, batch, k)
    assert not torch.allclose(changed, old)


def test_release_grid_needs_no_legacy_actor_heads(tmp_path):
    """Use the released sigma grid without loading or injecting an extra head file."""
    (tmp_path / "pdd_config.json").write_text(json.dumps({"sigmas": GRID, "num_inference_steps": 6}))
    cfg = SimpleNamespace(path=str(tmp_path), pipeline=SimpleNamespace(num_inference_steps=6))
    scheduler = FlowMatchSDEDiscreteScheduler()
    Actor.set_timesteps(scheduler, cfg, "cpu")
    torch.testing.assert_close(scheduler.sigmas, torch.tensor(GRID))
    cfg.pipeline.num_inference_steps = 5
    with pytest.raises(ValueError, match="released PDD grid"):
        Actor.set_timesteps(scheduler, cfg, "cpu")


def test_prompt_cache_keeps_native_token_ids_distinct(monkeypatch):
    """Cache the same IDs together without merging different pre-tokenized prompts."""
    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY

    cache = {}

    def encode(pipeline, prompt, references):
        if prompt not in cache:
            cache[prompt] = torch.tensor(pipeline.tokenizer(prompt)["input_ids"])
        return cache[prompt], None

    monkeypatch.setattr(rollout_module.MiniMaxH3ImagePipeline, "_encode_text", encode)
    pipeline = object.__new__(Rollout)
    torch.nn.Module.__init__(pipeline)
    tokenizer = lambda text: {"input_ids": [-1]}
    pipeline.tokenizer = tokenizer
    for ids in ([11, 12], [21, 22], [11, 12]):
        request = SimpleNamespace(
            prompts=[{"prompt_ids": ids}],
            sampling_params=SimpleNamespace(extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True}),
        )
        pipeline._ensure_prompt_text(request)
        result, _ = pipeline.encode_prompt(request.prompts[0]["prompt"], [])
        assert result.tolist() == ids
        assert pipeline.tokenizer is tokenizer
    assert len(cache) == 2
