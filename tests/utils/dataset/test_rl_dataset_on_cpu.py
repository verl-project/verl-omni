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

from io import BytesIO

import pytest
from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset as UpstreamRLHFDataset

from verl_omni.utils.dataset.rl_dataset import RLHFDataset


@pytest.fixture
def dataset():
    dataset = object.__new__(RLHFDataset)
    dataset.processor = None
    dataset.prompt_key = "prompt"
    dataset.negative_prompt_key = "negative_prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.serialize_dataset = False
    return dataset


@pytest.mark.parametrize("with_processor", [False, True])
@pytest.mark.parametrize("image_format", ["pil", "bytes", "path"])
def test_image_messages_preserve_processor_and_upstream_formats(dataset, tmp_path, with_processor, image_format):
    processor = object() if with_processor else None
    dataset.processor = processor
    image = Image.new("RGB", (56, 56), "red")
    if image_format == "bytes":
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = {"bytes": buffer.getvalue()}
    elif image_format == "path":
        payload = tmp_path / "frame.png"
        image.save(payload)
    else:
        payload = image

    messages = dataset._build_messages(
        {"prompt": [{"role": "user", "content": "<image>Animate this frame."}], "images": [payload]},
        key="prompt",
    )

    content = messages[0]["content"]
    assert content[0]["type"] == "image"
    if image_format == "path":
        assert content[0]["image"] == str(payload)
    else:
        assert content[0]["image"].tobytes() == image.tobytes()
    assert content[1] == {"type": "text", "text": "Animate this frame."}
    assert dataset.processor is processor


@pytest.mark.parametrize("prompt", ["<image><image>Animate.", "Animate."])
def test_no_processor_still_rejects_image_placeholder_count_mismatch(dataset, prompt):
    with pytest.raises(AssertionError, match="image_offset"):
        dataset._build_messages(
            {"prompt": [{"role": "user", "content": prompt}], "images": [Image.new("RGB", (56, 56))]},
            key="prompt",
        )
    assert dataset.processor is None


def test_no_processor_still_rejects_invalid_image_type(dataset):
    with pytest.raises(TypeError, match="unsupported image type"):
        dataset._build_messages(
            {"prompt": [{"role": "user", "content": "<image>Animate."}], "images": [123]},
            key="prompt",
        )
    assert dataset.processor is None


def test_text_only_messages_are_unchanged(dataset):
    prompt = [{"role": "user", "content": "A fox in snow."}]
    assert dataset._build_messages({"prompt": prompt}, key="prompt") == prompt
    assert dataset.processor is None


def test_processor_free_media_parsing_does_not_call_upstream(dataset, monkeypatch):
    def reject_delegation(*args, **kwargs):
        raise AssertionError("media parsing must not bypass the upstream processor contract")

    monkeypatch.setattr(UpstreamRLHFDataset, "_build_messages", reject_delegation)
    messages = dataset._build_messages(
        {"prompt": [{"role": "user", "content": "<image>Animate."}], "images": [Image.new("RGB", (56, 56))]},
        key="prompt",
    )
    assert messages[0]["content"][0]["type"] == "image"
    assert dataset.processor is None


@pytest.mark.parametrize("with_processor", [False, True])
@pytest.mark.parametrize("video_format", ["path", "frames", "dict"])
@pytest.mark.parametrize("audio_format", ["path", "dict", "url", "nested"])
def test_media_transport_preserves_video_audio_payloads(dataset, tmp_path, with_processor, video_format, audio_format):
    dataset.processor = object() if with_processor else None
    video_path = tmp_path / "video.mp4"
    video = {"path": video_path, "frames": [tmp_path / "frame.png"], "dict": {"video": str(video_path), "fps": 2}}[
        video_format
    ]
    audio = {
        "path": "audio.wav",
        "dict": {"audio": "audio.wav", "sample_rate": 24000},
        "url": {"audio_url": "https://example.invalid/audio.wav"},
        "nested": {"array": [0.0], "sampling_rate": 24000},
    }[audio_format]
    content = dataset._build_messages(
        {"prompt": [{"role": "user", "content": "<video><audio>Describe."}], "videos": [video], "audios": [audio]},
        key="prompt",
    )[0]["content"]
    expected_video = (
        {"type": "video", **video}
        if isinstance(video, dict)
        else {"type": "video", "video": [str(frame) for frame in video] if isinstance(video, list) else str(video)}
    )
    expected_audio = (
        {"type": "audio", **audio} if audio_format in ("dict", "url") else {"type": "audio", "audio": audio}
    )
    assert content == [expected_video, expected_audio, {"type": "text", "text": "Describe."}]


@pytest.mark.parametrize("negative_content", ["", "blurry", "<image>blurry"])
def test_no_processor_negative_prompt_only_uses_images_when_referenced(dataset, negative_content):
    messages = dataset._build_messages(
        {
            "prompt": [{"role": "user", "content": "<image>Animate this frame."}],
            "negative_prompt": [{"role": "user", "content": negative_content}],
            "images": [Image.new("RGB", (56, 56))],
        },
        key="negative_prompt",
    )

    if negative_content.startswith("<image>"):
        assert messages[0]["content"][0]["type"] == "image"
        assert messages[0]["content"][1] == {"type": "text", "text": "blurry"}
    else:
        assert messages == [{"role": "user", "content": negative_content}]
    assert dataset.processor is None
