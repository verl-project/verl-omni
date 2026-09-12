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
"""CPU parity test for the train-side diffusers rollouter prototype.

Validates ``sde_denoise_loop`` against an independently-written reference
loop that batches CFG the way production SD3 pipelines actually do (a single
concatenated forward instead of two separate forwards). Same
``FlowMatchSDEDiscreteScheduler``, same tiny offline SD3 checkpoint (see
``tests/special_e2e/build_sd3_tiny_random.py``), same seeds.

This is the CPU-runnable, CI-wired counterpart of the GPU parity check
posted as a comment on the PR that adds this file (a real T4 GPU run on
Modal). float32 throughout, since fp16 on CPU is not well supported by all
of the ops this loop touches and isn't the point of this test -- the fp16
run on GPU covers that regime separately.
"""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "special_e2e"))

diffusers = pytest.importorskip("diffusers")


def _stub_heavy_parent_packages() -> None:
    """Let ``verl_omni.workers.rollout.diffusers_train_rollout`` import without
    the full install (``verl``, ``vllm_omni``) this module doesn't need.

    ``verl_omni/__init__.py`` and ``verl_omni/workers/rollout/__init__.py``
    eagerly import ``verl``-dependent submodules at package-import time (see
    ``tests/utils/test_stable_diffusion_3_flops_on_cpu.py``-style PRs for the
    same pre-existing environment limitation, unrelated to this change).
    Pre-registering empty namespace packages -- pointed at the real source
    tree via ``__path__`` -- skips only those heavy ``__init__``s while still
    resolving submodule imports (this file, and
    ``verl_omni.pipelines.schedulers``) to the real, unmodified source.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for dotted_path in (
        "verl_omni",
        "verl_omni.pipelines",
        "verl_omni.workers",
        "verl_omni.workers.rollout",
    ):
        if dotted_path in sys.modules:
            continue
        module = types.ModuleType(dotted_path)
        module.__path__ = [os.path.join(repo_root, *dotted_path.split("."))]
        sys.modules[dotted_path] = module


_stub_heavy_parent_packages()


def _build_pipeline_and_scheduler():
    import build_sd3_tiny_random
    from diffusers import StableDiffusion3Pipeline

    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    # The repo's helper assumes a T5Tokenizer(vocab=...) API from a
    # transformers version/fork not pinned by this test; substitute a real
    # sentencepiece-backed T5Tokenizer trained on the same tiny corpus.
    def _real_t5_tokenizer(*, vocab_size: int = 32, model_max_length: int = 512):
        import tempfile

        import sentencepiece as spm
        from transformers import T5Tokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = os.path.join(tmp_dir, "corpus.txt")
            with open(corpus_path, "w") as f:
                f.write("a red circle on a white background\n")
                f.write("a blue square on a black background\n")
                f.write("a green triangle next to an orange rectangle\n")
            model_prefix = os.path.join(tmp_dir, "spm")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="unigram",
                pad_id=0,
                unk_id=2,
                eos_id=1,
                bos_id=-1,
                pad_piece="<pad>",
                unk_piece="<unk>",
                eos_piece="</s>",
            )
            return T5Tokenizer(vocab_file=model_prefix + ".model", model_max_length=model_max_length)

    build_sd3_tiny_random._build_tiny_t5_tokenizer = _real_t5_tokenizer

    components = build_sd3_tiny_random.get_dummy_sd3_components(hidden_size=8, seed=42)
    components.pop("image_encoder", None)
    components.pop("feature_extractor", None)
    pipe = StableDiffusion3Pipeline(**components).to(dtype=torch.float32)
    pipe.transformer.eval()
    pipe.set_progress_bar_config(disable=True)

    scheduler = FlowMatchSDEDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe, scheduler


@pytest.mark.parametrize("do_cfg", [True, False])
def test_sde_denoise_loop_matches_concatenated_cfg_reference(do_cfg: bool) -> None:
    from verl_omni.workers.rollout.diffusers_train_rollout import sde_denoise_loop

    pipe, base_scheduler = _build_pipeline_and_scheduler()
    device = "cpu"
    dtype = torch.float32

    prompt = ["a red circle on a white background"]
    negative_prompt = [""]
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=32,
    )

    num_channels = pipe.transformer.config.in_channels
    latent_h = latent_w = pipe.default_sample_size
    latent_shape = (1, num_channels, latent_h, latent_w)

    num_inference_steps = 4
    guidance_scale = 3.5
    noise_level = 0.7
    sde_window_range = (0, num_inference_steps)

    def fresh_scheduler():
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        s = FlowMatchSDEDiscreteScheduler.from_config(pipe.scheduler.config)
        s.set_timesteps(num_inference_steps, device=device)
        return s

    scheduler = fresh_scheduler()
    timesteps = scheduler.timesteps

    def make_initial_latents(seed: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(latent_shape, generator=g, device=device, dtype=torch.float32)

    latents_seed, sde_seed = 1234, 5678

    result = sde_denoise_loop(
        pipe.transformer,
        fresh_scheduler(),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds if do_cfg else None,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_cfg else None,
        latents=make_initial_latents(latents_seed),
        timesteps=timesteps,
        do_cfg=do_cfg,
        guidance_scale=guidance_scale,
        noise_level=noise_level,
        sde_window_range=sde_window_range,
        sde_type="sde",
        generator=torch.Generator(device=device).manual_seed(sde_seed),
        logprobs=True,
        model_dtype=dtype,
    )

    @torch.no_grad()
    def reference_loop(latents: torch.Tensor, scheduler, generator: torch.Generator):
        all_latents = [latents.detach().float().clone()]
        all_log_probs = []
        for t in timesteps:
            if do_cfg:
                latent_in = torch.cat([latents, latents], dim=0)
                timestep_in = t.expand(latent_in.shape[0]).to(device=device, dtype=dtype)
                embeds_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_in = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                latent_in = latents
                timestep_in = t.expand(latent_in.shape[0]).to(device=device, dtype=dtype)
                embeds_in = prompt_embeds
                pooled_in = pooled_prompt_embeds

            noise_pred = pipe.transformer(
                hidden_states=latent_in,
                timestep=timestep_in,
                encoder_hidden_states=embeds_in,
                pooled_projections=pooled_in,
                return_dict=False,
            )[0]
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents, log_prob, _, _ = scheduler.step(
                noise_pred.float(),
                t,
                latents,
                generator=generator,
                noise_level=noise_level,
                sde_type="sde",
                return_logprobs=True,
                return_dict=False,
            )
            all_latents.append(latents.detach().float().clone())
            all_log_probs.append(log_prob)
        return latents, torch.stack(all_latents, dim=0), torch.stack(all_log_probs, dim=0)

    final_ref, all_latents_ref, all_log_probs_ref = reference_loop(
        make_initial_latents(latents_seed),
        fresh_scheduler(),
        torch.Generator(device=device).manual_seed(sde_seed),
    )

    # float32: the two CFG-batching strategies should agree near machine precision.
    torch.testing.assert_close(result.all_latents, all_latents_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.final_latents, final_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.log_probs, all_log_probs_ref, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(result.final_latents).all()


def test_do_cfg_changes_output() -> None:
    """Guard against a no-op test: CFG must actually change the sampled trajectory."""
    from verl_omni.workers.rollout.diffusers_train_rollout import sde_denoise_loop

    pipe, _ = _build_pipeline_and_scheduler()
    device = "cpu"
    dtype = torch.float32

    prompt = ["a red circle on a white background"]
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=[""],
        negative_prompt_2=[""],
        negative_prompt_3=[""],
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=32,
    )

    num_channels = pipe.transformer.config.in_channels
    latent_h = latent_w = pipe.default_sample_size
    latent_shape = (1, num_channels, latent_h, latent_w)
    num_inference_steps = 4

    def fresh_scheduler():
        from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

        s = FlowMatchSDEDiscreteScheduler.from_config(pipe.scheduler.config)
        s.set_timesteps(num_inference_steps, device=device)
        return s

    timesteps = fresh_scheduler().timesteps

    def make_latents():
        g = torch.Generator(device=device).manual_seed(1234)
        return torch.randn(latent_shape, generator=g, device=device, dtype=torch.float32)

    common_kwargs = dict(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        latents=make_latents(),
        timesteps=timesteps,
        guidance_scale=3.5,
        noise_level=0.7,
        sde_window_range=(0, num_inference_steps),
        sde_type="sde",
        model_dtype=dtype,
    )

    result_cfg = sde_denoise_loop(
        pipe.transformer,
        fresh_scheduler(),
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        do_cfg=True,
        generator=torch.Generator(device=device).manual_seed(5678),
        logprobs=True,
        **common_kwargs,
    )
    result_no_cfg = sde_denoise_loop(
        pipe.transformer,
        fresh_scheduler(),
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        do_cfg=False,
        generator=torch.Generator(device=device).manual_seed(5678),
        logprobs=True,
        **common_kwargs,
    )

    assert (result_cfg.final_latents - result_no_cfg.final_latents).abs().max().item() > 1e-4
