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
"""Contract tests for the typed rollout request (verl_omni/pipelines/rollout_request.py)."""

import pytest

from verl_omni.pipelines.rollout_request import (
    MediaInput,
    OmniRolloutRequest,
    PromptBundle,
    condition_images_from_payload,
)


class TestOmniRolloutRequest:
    def test_text_only_request_has_no_media(self):
        request = OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1, 2, 3])
        assert request.media == ()
        assert request.multi_modal_data() == {}
        assert request.prompt.token_ids == [1, 2, 3]

    def test_carries_prompt_bundle_fields(self):
        mask = object()
        request = OmniRolloutRequest.from_generate_kwargs(
            prompt_ids=[1, 2],
            prompt_mask=mask,
            negative_prompt_ids=[3, 4],
            extra_prompt_ids={"clip": [5]},
            negative_extra_prompt_ids={"clip": [6]},
            mm_processor_kwargs={"fps": 8},
        )
        prompt = request.prompt
        assert isinstance(prompt, PromptBundle)
        assert prompt.mask is mask
        assert prompt.negative_token_ids == [3, 4]
        assert prompt.extra_token_ids == {"clip": [5]}
        assert prompt.negative_extra_token_ids == {"clip": [6]}
        assert prompt.mm_processor_kwargs == {"fps": 8}

    def test_media_included_only_when_not_none(self):
        request = OmniRolloutRequest.from_generate_kwargs(
            prompt_ids=[1],
            image_data=["img"],
            video_data=None,
            audio_data=["aud"],
        )
        assert request.media == (MediaInput("image", ["img"]), MediaInput("audio", ["aud"]))
        assert request.multi_modal_data() == {"image": ["img"], "audio": ["aud"]}

    def test_empty_list_is_a_present_stream(self):
        # An empty list is a present-but-empty stream (matches legacy behavior),
        # while None means the stream is absent.
        request = OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1], image_data=[])
        assert request.multi_modal_data() == {"image": []}

    def test_all_three_modalities_in_declaration_order(self):
        request = OmniRolloutRequest.from_generate_kwargs(
            prompt_ids=[1],
            image_data=["i"],
            video_data=["v"],
            audio_data=["a"],
        )
        assert [m.modality for m in request.media] == ["image", "video", "audio"]

    def test_duplicate_media_modality_is_rejected(self):
        request = OmniRolloutRequest(
            prompt=PromptBundle(token_ids=[1]),
            media=(MediaInput("image", ["first"]), MediaInput("image", ["second"])),
        )

        with pytest.raises(ValueError, match="Duplicate media modality.*image"):
            request.multi_modal_data()

    @pytest.mark.parametrize(
        "media",
        [
            pytest.param(lambda: MediaInput("depth", ["value"]), id="unsupported-modality"),
            pytest.param(lambda: MediaInput("image", None), id="missing-data"),
        ],
    )
    def test_invalid_media_input_is_rejected(self, media):
        with pytest.raises(ValueError):
            media()


class TestConditionImagesFromPayload:
    def test_equivalent_aliases_are_accepted(self):
        image = object()
        payload = {
            "images": [image],
            "image": image,
            "multi_modal_data": {"image": [image]},
            "extra_args": {"multi_modal_data": {"image": [image]}},
            "additional_information": {"condition_images": (image,)},
        }
        assert condition_images_from_payload(payload) == [image]

    def test_conflicting_aliases_are_rejected(self):
        payload = {
            "images": ["top-level"],
            "multi_modal_data": {"image": ["different"]},
        }

        with pytest.raises(
            ValueError,
            match=r"Conflicting condition-image aliases.*images.*multi_modal_data\.image",
        ):
            condition_images_from_payload(payload)

    def test_single_image_key_is_wrapped(self):
        assert condition_images_from_payload({"image": "img"}) == ["img"]

    def test_multimodal_data_image(self):
        assert condition_images_from_payload({"multi_modal_data": {"image": ["raw"]}}) == ["raw"]

    def test_extra_args_multimodal_tuple_becomes_list(self):
        payload = {"extra_args": {"multi_modal_data": {"image": ("a", "b")}}}
        assert condition_images_from_payload(payload) == ["a", "b"]

    def test_additional_information_condition_images_fallback(self):
        assert condition_images_from_payload({"additional_information": {"condition_images": "img"}}) == ["img"]

    def test_missing_returns_empty(self):
        assert condition_images_from_payload({"prompt_token_ids": [1, 2]}) == []

    @pytest.mark.parametrize("key", ["multi_modal_data", "extra_args", "additional_information"])
    def test_malformed_alias_container_is_rejected(self, key):
        with pytest.raises(TypeError, match=key):
            condition_images_from_payload({key: "not-a-mapping"})


def test_prompt_bundle_defaults():
    bundle = PromptBundle(token_ids=[1])
    assert bundle.mask is None
    assert bundle.negative_token_ids is None
    assert bundle.extra_token_ids is None
    assert bundle.negative_extra_token_ids is None
    assert bundle.mm_processor_kwargs is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
