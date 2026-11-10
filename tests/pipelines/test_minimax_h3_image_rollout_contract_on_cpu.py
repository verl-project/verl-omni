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
"""CPU contract of the MiniMax H3 image DiffusionNFT rollout adapter around a stubbed native pipeline."""

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import (
    MiniMaxH3ImagePipeline,
    minimax_h3_image_run_sigmas,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import parse_image_latent_meta, pdd_step_indices
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.vllm_omni_rollout_adapter import (
    MiniMaxH3ImageDiffusionNFTPipeline,
)
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

_PDD_SIGMAS = (1.0, 0.96752, 0.88755, 0.75, 0.53277, 0.23201, 0.0)
_LATENT_H, _LATENT_W, _TEXT_LEN, _TEXT_DIM = 4, 6, 5, 8
_ROWS = (_LATENT_H // 2) * (_LATENT_W // 2)
_HEIGHT, _WIDTH = _LATENT_H * 16, _LATENT_W * 16


def _run(monkeypatch, *, output_type="pt", extra_args=None, seed=7, grids=(), run_denoise=True, dummy=False):
    seen = {}
    target_rows = torch.arange(_ROWS * 96, dtype=torch.bfloat16).reshape(_ROWS, 96) / 64
    hidden = torch.randn(_TEXT_LEN, _TEXT_DIM).to(torch.bfloat16)
    anchor_rows = sum((height // 2) * (width // 2) for height, width in grids)
    anchors = torch.full((anchor_rows, 96), 0.5)

    def native_denoise(self, **kwargs):
        seen["denoise_kwargs"] = set(kwargs)
        return target_rows

    def native_forward(self, request):
        sampling = request.sampling_params
        seen["output_type"] = sampling.output_type
        seen["generator_device"] = sampling.generator_device
        if run_denoise:
            rows = self._denoise(
                hidden=hidden,
                text_tags=torch.ones(_TEXT_LEN, dtype=torch.long),
                grids=list(grids),
                anchors=anchors,
                latent_h=_LATENT_H,
                latent_w=_LATENT_W,
                num_steps=self.num_steps,
            )
            assert rows is target_rows
        if sampling.output_type == "latent":
            return DiffusionOutput(output=target_rows.unsqueeze(0))
        return DiffusionOutput(output=torch.full((1, _HEIGHT, _WIDTH, 3), 200, dtype=torch.uint8))

    monkeypatch.setattr(MiniMaxH3ImagePipeline, "_denoise", native_denoise)
    monkeypatch.setattr(MiniMaxH3ImagePipeline, "forward", native_forward)
    pipeline = object.__new__(MiniMaxH3ImageDiffusionNFTPipeline)
    pipeline.num_steps = 6
    pipeline.pdd_sigmas = _PDD_SIGMAS
    pipeline._h3_prompt_ids = None
    pipeline._nft_capture = None
    request = SimpleNamespace(
        request_id="h3-image-test",
        prompts=[{"prompt": "a red cube"}],
        sampling_params=OmniDiffusionSamplingParams(
            seed=seed, output_type=output_type, extra_args=extra_args or {}, num_outputs_per_prompt=1
        ),
        is_dummy_run=lambda: dummy,
    )
    result = pipeline.forward(request)
    return SimpleNamespace(
        result=result, seen=seen, request=request, pipeline=pipeline, target_rows=target_rows, hidden=hidden
    )


def _process(run, output_type="pt"):
    server = SimpleNamespace(
        global_steps=1,
        model_config=SimpleNamespace(architecture="MiniMaxH3ImagePipeline", algorithm="diffusion_nft"),
    )
    result = run.result
    final_res = OmniRequestOutput.from_diffusion(
        images=result.output["payload"]["image"],
        multimodal_output={"metadata": result.output["metadata"]},
        trajectory_latents=result.trajectory_latents,
        trajectory_timesteps=result.trajectory_timesteps,
        trajectory_log_probs=result.trajectory_log_probs,
        request_id="h3-image-test",
    )
    return DiffusionStrategy(server).process_output(final_res, None, {"output_type": output_type, "logprobs": False})


class TestTextToImage:
    def test_training_tensors_describe_the_six_step_rollout(self, monkeypatch):
        run = _run(monkeypatch)
        rl = run.result.output["metadata"]["rl"]

        assert rl["latents_clean"].dtype == torch.float32
        assert torch.equal(rl["latents_clean"], run.target_rows.float().reshape(1, _ROWS * 96))
        assert parse_image_latent_meta(rl["latent_meta"]) == (_ROWS, _LATENT_H, _LATENT_W)
        run_sigmas = torch.tensor(minimax_h3_image_run_sigmas(_PDD_SIGMAS)[:-1])
        assert torch.equal(rl["pdd_sigmas"], run_sigmas.unsqueeze(0))
        assert rl["train_timesteps"].shape == (1, 6)
        assert pdd_step_indices(rl["train_timesteps"][0].long(), rl["pdd_sigmas"][0]).tolist() == list(range(6))
        assert rl["condition_video_rows"].shape == (1, 0, 96)
        assert rl["condition_video_row_count"].tolist() == [[0]]
        assert rl["prompt_token_tags"].shape == (1, _TEXT_LEN)
        assert "ref_block_meta" not in rl and "ref_block_count" not in rl

        embeddings = run.result.output["metadata"]["prompt_embeddings"]
        assert torch.equal(embeddings["prompt_embeds"], run.hidden.unsqueeze(0))
        assert embeddings["prompt_embeds_mask"].tolist() == [[1] * _TEXT_LEN]
        assert run.pipeline._nft_capture is None and run.pipeline._h3_prompt_ids is None

    @pytest.mark.parametrize(
        "output_type, native_type, primary",
        [("pt", "pil", "image_preview"), ("latent", "latent", "image_latent")],
        ids=["decoded", "latent-only"],
    )
    def test_native_forward_runs_once_with_a_cpu_generator(self, monkeypatch, output_type, native_type, primary):
        run = _run(monkeypatch, output_type=output_type)
        assert run.seen["output_type"] == native_type
        assert run.seen["generator_device"] == "cpu"
        assert run.request.sampling_params.output_type == output_type

        processed = _process(run, output_type)
        assert processed.primary_artifact == primary
        latent = processed.artifacts["image_latent"].data
        assert latent.shape == (_ROWS, 96) and torch.equal(latent.float(), run.target_rows.float())
        if output_type == "pt":
            preview = processed.artifacts["image_preview"].data
            assert preview.shape == (3, _HEIGHT, _WIDTH) and preview.dtype == torch.uint8
            assert int(preview.min()) == int(preview.max()) == 200
        else:
            assert set(processed.artifacts) == {"image_latent"}
        assert processed.extra_fields["latents_clean"].shape == (_ROWS * 96,)
        assert processed.extra_fields["pdd_sigmas"].shape == (6,)

    def test_missing_seed_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="per-sample seed"):
            _run(monkeypatch, seed=None)

    def test_skipped_denoise_loop_is_rejected(self, monkeypatch):
        with pytest.raises(RuntimeError, match="full PDD denoise loop"):
            _run(monkeypatch, run_denoise=False)

    def test_dummy_run_passes_through_untouched(self, monkeypatch):
        run = _run(monkeypatch, seed=None, dummy=True)
        assert isinstance(run.result.output, torch.Tensor)


class TestEdit:
    def test_reference_blocks_and_condition_rows_reach_the_trainer(self, monkeypatch):
        grids = ((4, 6), (2, 4))
        run = _run(monkeypatch, grids=grids)
        rl = run.result.output["metadata"]["rl"]

        assert rl["condition_video_rows"].shape == (1, 6 + 2, 96)
        assert rl["condition_video_row_count"].tolist() == [[8]]
        assert rl["ref_block_count"].tolist() == [[2]]
        assert rl["ref_block_meta"][0, :2, 3:5].tolist() == [[4, 6], [2, 4]]
        assert not rl["ref_block_meta"][0, 2:].count_nonzero()


class TestPromptEncoding:
    def _pipeline(self, monkeypatch):
        calls = []

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(char) for char in text]}

        def native_encode(self, prompt, references):
            label = self.tokenizer("Picture 1: ", add_special_tokens=False)["input_ids"] if references else []
            ids = label + self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            calls.append(ids)
            return torch.zeros(len(ids), _TEXT_DIM), torch.ones(len(ids), dtype=torch.long)

        monkeypatch.setattr(MiniMaxH3ImagePipeline, "_encode_text", native_encode)
        pipeline = object.__new__(MiniMaxH3ImageDiffusionNFTPipeline)
        pipeline.tokenizer = Tokenizer()
        pipeline._h3_prompt_ids = None
        return pipeline, calls

    def test_agent_loop_token_ids_replace_only_the_prompt_segment(self, monkeypatch):
        pipeline, calls = self._pipeline(monkeypatch)
        tokenizer = pipeline.tokenizer
        pipeline._h3_prompt_ids = torch.tensor([11, 12, 13])

        hidden, tags = pipeline._encode_text("ab", ["reference"])

        assert calls == [[ord(char) for char in "Picture 1: "] + [11, 12, 13]]
        assert hidden.shape[0] == tags.shape[0] == len(calls[0])
        assert pipeline.tokenizer is tokenizer

    def test_plain_text_prompt_uses_the_native_tokenizer(self, monkeypatch):
        pipeline, calls = self._pipeline(monkeypatch)
        pipeline._encode_text("ab", [])
        assert calls == [[97, 98]]

    def test_token_ids_without_the_agent_loop_flag_are_rejected(self, monkeypatch):
        pipeline, _ = self._pipeline(monkeypatch)
        request = SimpleNamespace(
            prompts=[{"prompt": "ab", "prompt_token_ids": [1, 2]}],
            sampling_params=SimpleNamespace(extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: False}),
        )
        with pytest.raises(ValueError, match="minimax_h3_diffusion_single_turn_agent"):
            pipeline._ensure_prompt_text(request)
