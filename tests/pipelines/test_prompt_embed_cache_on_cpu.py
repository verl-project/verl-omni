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
"""CPU tests for prompt-embedding cache adapter boundaries."""

from types import MethodType, SimpleNamespace

import pytest
import torch


def _record_encode_prompt(calls):
    def encode_prompt(**kwargs):
        calls.append((kwargs["prompt_ids"], kwargs["attention_mask"]))
        return torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long)

    return encode_prompt


def test_prompt_collators_preserve_lists_only_when_requested():
    from verl_omni.pipelines.request_batch import collate_prompt_mask, collate_prompt_rows

    prompts = [
        {"prompt_token_ids": [1, 2], "prompt_mask": [1, 1]},
        {"prompt_token_ids": [3], "prompt_mask": [1]},
    ]
    tensor_rows, tensor_lengths = collate_prompt_rows(
        prompts,
        ("prompt_token_ids",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
    )
    list_rows, lengths = collate_prompt_rows(
        prompts,
        ("prompt_token_ids",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
        preserve_lists=True,
    )
    default_rows, default_lengths = collate_prompt_rows(
        prompts,
        ("prompt_token_ids",),
        [[1, 2], [3, 0]],
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
        preserve_lists=True,
    )
    list_mask = collate_prompt_mask(
        prompts,
        ("prompt_mask",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_mask",
        token_lengths=lengths,
        target_seq_len=2,
        preserve_lists=True,
    )
    tensor_mask = collate_prompt_mask(
        prompts,
        ("prompt_mask",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_mask",
        token_lengths=tensor_lengths,
        target_seq_len=2,
    )
    derived_tensor_mask = collate_prompt_mask(
        [{}, {}],
        ("prompt_mask",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_mask",
        token_lengths=tensor_lengths,
        target_seq_len=2,
    )

    assert isinstance(tensor_rows, torch.Tensor)
    assert torch.equal(tensor_rows, torch.tensor([[1, 2], [3, 0]], dtype=torch.long))
    assert tensor_rows.dtype == torch.long
    assert tensor_lengths == [2, 1]
    assert torch.equal(tensor_mask, torch.tensor([[True, True], [True, False]]))
    assert tensor_mask.dtype == torch.bool
    assert torch.equal(derived_tensor_mask, tensor_mask)
    assert list_rows == [[1, 2], [3, 0]]
    assert default_rows == [[1, 2], [3, 0]]
    assert default_lengths == [2, 2]
    assert list_mask == [[True, True], [True, False]]

    tensor_default_rows, tensor_default_lengths = collate_prompt_rows(
        prompts,
        ("prompt_token_ids",),
        torch.tensor([[1, 2], [3, 0]]),
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
        preserve_lists=True,
    )
    tensor_prompt_rows, tensor_prompt_lengths = collate_prompt_rows(
        [
            {"prompt_token_ids": torch.tensor([1, 2])},
            {"prompt_token_ids": torch.tensor([3])},
        ],
        ("prompt_token_ids",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_token_ids",
        preserve_lists=True,
    )
    tensor_prompt_mask = collate_prompt_mask(
        [
            {"prompt_mask": torch.tensor([True, True])},
            {"prompt_mask": torch.tensor([True])},
        ],
        ("prompt_mask",),
        None,
        device=torch.device("cpu"),
        field_name="prompt_mask",
        token_lengths=tensor_prompt_lengths,
        target_seq_len=2,
        preserve_lists=True,
    )

    assert torch.equal(tensor_default_rows, torch.tensor([[1, 2], [3, 0]]))
    assert tensor_default_lengths == [2, 2]
    assert torch.equal(tensor_prompt_rows, torch.tensor([[1, 2], [3, 0]]))
    assert tensor_prompt_lengths == [2, 1]
    assert torch.equal(tensor_prompt_mask, torch.tensor([[True, True], [True, False]]))


def test_qwen_encode_prompt_preserves_previous_tensor_inputs():
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter import (
        QwenImageDiffusionNFTPipeline,
    )
    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline
    from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    assert QwenImageDiffusionNFTPipeline.encode_prompt is QwenImageTokenIdPromptMixin.encode_prompt

    def make_get_prompt_embeds(encoded_inputs):
        def get_prompt_embeds(prompt_ids, attention_mask=None):
            encoded_inputs["prompt_ids"] = prompt_ids
            encoded_inputs["attention_mask"] = attention_mask
            return torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.long)

        return get_prompt_embeds

    for pipeline_class in (QwenImageTokenIdPromptMixin, QwenImagePipelineWithLogProb, QwenImageDPOPipeline):
        encoded_inputs = {}
        pipeline = SimpleNamespace(
            device=torch.device("cpu"), _get_qwen_prompt_embeds=make_get_prompt_embeds(encoded_inputs)
        )
        pipeline_class.encode_prompt(pipeline, prompt_ids=[[1, 2]], attention_mask=[[1, 1]])

        assert torch.equal(encoded_inputs["prompt_ids"], torch.tensor([[1, 2]], dtype=torch.long))
        assert torch.equal(encoded_inputs["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
        assert encoded_inputs["prompt_ids"].device.type == "cpu"
        assert encoded_inputs["attention_mask"].device.type == "cpu"

        pipeline_class.encode_prompt(
            pipeline,
            prompt_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[True, True]]),
        )

        assert torch.equal(encoded_inputs["prompt_ids"], torch.tensor([[1, 2]], dtype=torch.long))
        assert torch.equal(encoded_inputs["attention_mask"], torch.tensor([[True, True]], dtype=torch.bool))

        pipeline_class.encode_prompt(
            pipeline,
            prompt_ids=[[1, 2]],
            attention_mask=torch.tensor([[True, False]]),
        )

        assert torch.equal(encoded_inputs["prompt_ids"], torch.tensor([[1, 2]], dtype=torch.long))
        assert torch.equal(encoded_inputs["attention_mask"], torch.tensor([[True, False]], dtype=torch.bool))


def test_flow_encode_prompt_accepts_packed_lists_and_tensors():
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    encoded_inputs = {}

    def get_prompt_embeds(prompt_ids, attention_mask=None):
        encoded_inputs["prompt_ids"] = prompt_ids
        encoded_inputs["attention_mask"] = attention_mask
        return torch.zeros(2, 2, 4), torch.ones(2, 2, dtype=torch.long)

    pipeline = SimpleNamespace(device=torch.device("cpu"), _get_qwen_prompt_embeds=get_prompt_embeds)
    for prompt_ids, attention_mask in (
        ([[1, 2], [3, 0]], [[True, True], [True, False]]),
        (torch.tensor([[1, 2], [3, 0]]), torch.tensor([[True, True], [True, False]])),
    ):
        QwenImagePipelineWithLogProb.encode_prompt(
            pipeline,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
        )

        assert torch.equal(encoded_inputs["prompt_ids"], torch.tensor([[1, 2], [3, 0]], dtype=torch.long))
        assert torch.equal(
            encoded_inputs["attention_mask"], torch.tensor([[True, True], [True, False]], dtype=torch.bool)
        )
        assert encoded_inputs["prompt_ids"].device.type == "cpu"
        assert encoded_inputs["attention_mask"].device.type == "cpu"


@pytest.mark.parametrize("use_tensors", [False, True])
def test_qwen_encode_prompt_preserves_output_semantics_after_tensorization(use_tensors):
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline
    from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    base_embeds = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    base_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    for pipeline_class in (QwenImageTokenIdPromptMixin, QwenImagePipelineWithLogProb, QwenImageDPOPipeline):

        def get_prompt_embeds(prompt_ids, attention_mask=None):
            assert torch.equal(prompt_ids, torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long))
            assert torch.equal(attention_mask, torch.tensor([[True, True, False], [True, False, False]]))
            return base_embeds.clone(), base_mask.clone()

        pipeline = SimpleNamespace(device=torch.device("cpu"), _get_qwen_prompt_embeds=get_prompt_embeds)
        prompt_ids = [[1, 2, 0], [3, 0, 0]]
        attention_mask = [[True, True, False], [True, False, False]]
        if use_tensors:
            prompt_ids = torch.tensor(prompt_ids)
            attention_mask = torch.tensor(attention_mask)
        prompt_embeds, prompt_embeds_mask = pipeline_class.encode_prompt(
            pipeline,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            num_images_per_prompt=2,
            max_sequence_length=2,
        )

        expected_embeds = base_embeds[:, :2].repeat_interleave(2, dim=0)
        expected_mask = base_mask[:, :2].repeat_interleave(2, dim=0)
        torch.testing.assert_close(prompt_embeds, expected_embeds)
        torch.testing.assert_close(prompt_embeds_mask, expected_mask)
        assert prompt_embeds.shape == (4, 2, 4)
        assert prompt_embeds_mask.shape == (4, 2)
        assert prompt_embeds.dtype == base_embeds.dtype
        assert prompt_embeds_mask.dtype == base_mask.dtype


def test_qwen_encode_prompt_accepts_precomputed_embeds_without_prompt_ids():
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline
    from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    prompt_embeds = torch.zeros(1, 2, 4)
    prompt_embeds_mask = torch.ones(1, 2, dtype=torch.long)

    for pipeline_class in (QwenImageTokenIdPromptMixin, QwenImagePipelineWithLogProb, QwenImageDPOPipeline):
        output_embeds, output_mask = pipeline_class.encode_prompt(
            SimpleNamespace(device=torch.device("cpu")),
            prompt_ids=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        assert torch.equal(output_embeds, prompt_embeds)
        assert torch.equal(output_mask, prompt_embeds_mask)


def test_qwen_list_inputs_hit_vllm_prompt_cache():
    pytest.importorskip("vllm_omni")
    from vllm_omni.diffusion.cache.prompt_embed_cache import install_prompt_embed_cache

    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline
    from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    for pipeline_class in (QwenImageTokenIdPromptMixin, QwenImagePipelineWithLogProb, QwenImageDPOPipeline):
        encoder_calls = 0

        def get_prompt_embeds(prompt_ids, attention_mask=None):
            nonlocal encoder_calls
            encoder_calls += 1
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
            return torch.zeros(batch_size, 2, 4), torch.ones(batch_size, 2, dtype=torch.long)

        pipeline = SimpleNamespace(device=torch.device("cpu"), _get_qwen_prompt_embeds=get_prompt_embeds)
        pipeline.encode_prompt = MethodType(pipeline_class.encode_prompt, pipeline)
        cache = install_prompt_embed_cache(pipeline, enabled=True)
        assert cache is not None

        prompt_ids = [[1, 2], [3, 4]] if pipeline_class is QwenImagePipelineWithLogProb else [1, 2]
        attention_mask = [[1, 1], [1, 1]] if pipeline_class is QwenImagePipelineWithLogProb else [1, 1]
        for _ in range(2):
            pipeline.encode_prompt(prompt_ids=prompt_ids, attention_mask=attention_mask)

        changed_prompt_ids = [[1, 3], [3, 4]] if pipeline_class is QwenImagePipelineWithLogProb else [1, 3]
        changed_attention_mask = [[1, 0], [1, 1]] if pipeline_class is QwenImagePipelineWithLogProb else [1, 0]
        pipeline.encode_prompt(prompt_ids=changed_prompt_ids, attention_mask=attention_mask)
        pipeline.encode_prompt(prompt_ids=prompt_ids, attention_mask=changed_attention_mask)
        tensor_attention_mask = torch.tensor(attention_mask)
        pipeline.encode_prompt(prompt_ids=prompt_ids, attention_mask=tensor_attention_mask)

        assert encoder_calls == 4
        assert cache.stats() == {"size": 3, "max_size": 32, "hits": 1, "misses": 3, "bypassed": 1}


@pytest.mark.parametrize(
    "cache_enabled",
    [None, False, True],
    ids=["not-installed", "disabled", "enabled"],
)
def test_step_prompt_tokenizers_follow_prompt_cache_mode(cache_enabled):
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter import (
        QwenImageDiffusionNFTPipeline,
    )
    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    class Tokenizer:
        def __init__(self):
            self.calls = []
            self.to_calls = []

        def __call__(self, *_args, **kwargs):
            self.calls.append((_args, kwargs))
            if kwargs.get("return_tensors") == "pt":
                input_ids = torch.tensor([[1, 2], [3, 4]])
                attention_mask = torch.tensor([[1, 1], [1, 1]])
            else:
                input_ids = [[1, 2], [3, 4]]
                attention_mask = [[1, 1], [1, 1]]

            encoding = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

            def to(device):
                self.to_calls.append(device)
                return encoding

            encoding.to = to
            return encoding

    tokenizers = (
        QwenImageDiffusionNFTPipeline._tokenize_step_prompt,
        QwenImageDPOPipeline._tokenize_step_prompt,
        QwenImagePipelineWithLogProb._tokenize_text_prompt,
    )
    for tokenize_prompt in tokenizers:
        tokenizer = Tokenizer()
        pipeline = SimpleNamespace(
            prompt_template_encode="wrapped:{}",
            prompt_template_encode_start_idx=34,
            tokenizer_max_length=2,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            _prompt_embed_cache=None if cache_enabled is None else SimpleNamespace(enabled=cache_enabled),
        )
        prompt_ids, prompt_mask = tokenize_prompt(pipeline, ["first", "second"])
        if cache_enabled:
            assert prompt_ids == [[1, 2], [3, 4]]
            assert prompt_mask == [[1, 1], [1, 1]]
            assert tokenizer.to_calls == []
        else:
            torch.testing.assert_close(prompt_ids, torch.tensor([[1, 2], [3, 4]]))
            torch.testing.assert_close(prompt_mask, torch.tensor([[1, 1], [1, 1]]))
            assert tokenizer.to_calls == [torch.device("cpu")]
        assert tokenizer.calls == [
            (
                (["wrapped:first", "wrapped:second"],),
                {
                    "max_length": 36,
                    "padding": True,
                    "truncation": True,
                    **({} if cache_enabled else {"return_tensors": "pt"}),
                },
            )
        ]


@pytest.mark.parametrize("use_tensors", [False, True])
def test_dpo_prepare_encode_preserves_input_path_and_batch_size(use_tensors):
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline

    class StopAfterLatents(Exception):
        pass

    calls = []
    latent_batch_sizes = []

    def encode_prompt(**kwargs):
        calls.append((kwargs["prompt_ids"], kwargs["attention_mask"]))
        return torch.zeros(2, 2, 4), torch.ones(2, 2, dtype=torch.long)

    def prepare_latents(batch_size, *_args):
        latent_batch_sizes.append(batch_size)
        raise StopAfterLatents

    prompt_ids = torch.tensor([[1, 2], [3, 4]]) if use_tensors else [[1, 2], [3, 4]]
    prompt_mask = torch.tensor([[1, 1], [1, 1]]) if use_tensors else [[1, 1], [1, 1]]

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        default_sample_size=2,
        vae_scale_factor=8,
        transformer=SimpleNamespace(in_channels=4),
        _extract_step_prompt_ids=lambda _prompt: (prompt_ids, prompt_mask, None, None),
        encode_prompt=encode_prompt,
        prepare_latents=prepare_latents,
    )
    sampling = SimpleNamespace(
        height=None,
        width=None,
        num_inference_steps=None,
        num_outputs_per_prompt=1,
        true_cfg_scale=2.0,
        max_sequence_length=None,
        generator=None,
        seed=None,
    )
    state = SimpleNamespace(sampling=sampling, prompt={})

    with pytest.raises(StopAfterLatents):
        QwenImageDPOPipeline.prepare_encode(pipeline, state)

    if use_tensors:
        assert torch.equal(calls[0][0], prompt_ids)
        assert torch.equal(calls[0][1], prompt_mask)
    else:
        assert calls == [(prompt_ids, prompt_mask)]
    assert latent_batch_sizes == [2]


@pytest.mark.parametrize("use_tensors", [False, True])
def test_nft_prepare_context_preserves_input_path_and_batch_size(use_tensors):
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_diffusion_nft.vllm_omni_rollout_adapter import (
        QwenImageDiffusionNFTPipeline,
    )

    calls = []
    latent_batch_sizes = []

    def prepare_latents(batch_size, *_args):
        latent_batch_sizes.append(batch_size)
        return torch.zeros(batch_size, 1, 1)

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        vae_scale_factor=8,
        transformer=SimpleNamespace(in_channels=4, guidance_embeds=False),
        check_cfg_parallel_validity=lambda *_: None,
        encode_prompt=_record_encode_prompt(calls),
        prepare_latents=prepare_latents,
        prepare_timesteps=lambda *_: (torch.tensor([1]), 1),
    )
    prompt_ids = torch.tensor([[1, 2], [3, 4]]) if use_tensors else [[1, 2], [3, 4]]
    prompt_mask = torch.tensor([[1, 1], [1, 1]]) if use_tensors else [[1, 1], [1, 1]]
    negative_prompt_ids = torch.tensor([[5, 6], [7, 8]]) if use_tensors else [[5, 6], [7, 8]]
    negative_prompt_mask = torch.tensor([[1, 1], [1, 1]]) if use_tensors else [[1, 1], [1, 1]]
    QwenImageDiffusionNFTPipeline._prepare_token_id_generation_context(
        pipeline,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_mask=negative_prompt_mask,
        true_cfg_scale=2.0,
        height=16,
        width=16,
        num_inference_steps=1,
        sigmas=None,
        guidance_scale=1.0,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        attention_kwargs=None,
        max_sequence_length=2,
    )

    if use_tensors:
        assert torch.equal(calls[0][0], prompt_ids)
        assert torch.equal(calls[0][1], prompt_mask)
        assert torch.equal(calls[1][0], negative_prompt_ids)
        assert torch.equal(calls[1][1], negative_prompt_mask)
    else:
        assert calls == [(prompt_ids, prompt_mask), (negative_prompt_ids, negative_prompt_mask)]
    assert latent_batch_sizes == [2]


def test_dpo_forward_passes_list_inputs_to_encode_prompt():
    pytest.importorskip("vllm_omni")

    from verl_omni.pipelines.qwen_image_dpo.vllm_omni_rollout_adapter import QwenImageDPOPipeline

    class StopAfterPromptEncoding(Exception):
        pass

    def stop_after_prompt_encoding(*_):
        raise StopAfterPromptEncoding

    calls = []
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        default_sample_size=2,
        vae_scale_factor=8,
        transformer=SimpleNamespace(in_channels=4),
        encode_prompt=_record_encode_prompt(calls),
        prepare_latents=stop_after_prompt_encoding,
    )
    sampling_params = SimpleNamespace(
        height=None,
        width=None,
        num_inference_steps=None,
        max_sequence_length=None,
        generator=None,
        seed=None,
        true_cfg_scale=None,
        num_outputs_per_prompt=None,
    )
    request = SimpleNamespace(
        prompts=[
            {
                "prompt_token_ids": [1, 2],
                "prompt_mask": [1, 1],
                "negative_prompt_ids": [3, 4],
                "negative_prompt_mask": [1, 1],
            }
        ],
        sampling_params=sampling_params,
    )

    with pytest.raises(StopAfterPromptEncoding):
        QwenImageDPOPipeline.forward(pipeline, request, true_cfg_scale=2.0)

    assert calls == [([1, 2], [1, 1]), ([3, 4], [1, 1])]


@pytest.mark.parametrize("use_tensors", [False, True])
def test_flow_forward_preserves_input_path_through_encode_prompt(use_tensors):
    pytest.importorskip("vllm_omni")

    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    class StopAfterPromptEncoding(Exception):
        pass

    calls = []
    latent_batch_sizes = []

    def encode_prompt(**kwargs):
        prompt_ids = kwargs["prompt_ids"]
        attention_mask = kwargs["attention_mask"]
        calls.append((prompt_ids, attention_mask))
        batch_size = len(prompt_ids)
        return torch.zeros(batch_size, 2, 4), torch.ones(batch_size, 2, dtype=torch.long)

    def prepare_latents(batch_size, *_args):
        latent_batch_sizes.append(batch_size)
        raise StopAfterPromptEncoding

    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        _prompt_embed_cache=SimpleNamespace(enabled=True),
        default_sample_size=2,
        vae_scale_factor=8,
        transformer=SimpleNamespace(in_channels=4),
        encode_prompt=encode_prompt,
        prepare_latents=prepare_latents,
    )
    sampling = SimpleNamespace(
        height=None,
        width=None,
        num_inference_steps=None,
        sigmas=None,
        max_sequence_length=None,
        output_type=None,
        extra_args={},
        generator=None,
        seed=None,
        true_cfg_scale=2.0,
        guidance_scale_provided=False,
        num_outputs_per_prompt=1,
        latents=None,
    )

    def prompt_value(value, *, dtype=None):
        return torch.tensor(value, dtype=dtype) if use_tensors else value

    request_batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="request-0",
                prompt={
                    "prompt_token_ids": prompt_value([1, 2]),
                    "prompt_mask": prompt_value([1, 1]),
                    "negative_prompt_ids": prompt_value([]),
                    "negative_prompt_mask": prompt_value([], dtype=torch.bool),
                },
                sampling_params=sampling,
                kv_sender_info=None,
            ),
            SimpleNamespace(
                request_id="request-1",
                prompt={
                    "prompt_token_ids": prompt_value([3]),
                    "prompt_mask": prompt_value([1]),
                    "negative_prompt_ids": prompt_value([4]),
                    "negative_prompt_mask": prompt_value([1]),
                },
                sampling_params=sampling,
                kv_sender_info=None,
            ),
        ]
    )

    with pytest.raises(StopAfterPromptEncoding):
        QwenImagePipelineWithLogProb.forward(pipeline, request_batch)

    if use_tensors:
        assert torch.equal(calls[0][0], torch.tensor([[1, 2], [3, 0]]))
        assert torch.equal(calls[0][1], torch.tensor([[True, True], [True, False]]))
        assert torch.equal(calls[1][0], torch.tensor([[0], [4]]))
        assert torch.equal(calls[1][1], torch.tensor([[False], [True]]))
    else:
        assert calls == [
            ([[1, 2], [3, 0]], [[True, True], [True, False]]),
            ([[0], [4]], [[False], [True]]),
        ]
    assert latent_batch_sizes == [2]
