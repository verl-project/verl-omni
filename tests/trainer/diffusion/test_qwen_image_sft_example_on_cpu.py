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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.qwen_image_sft_trainer.qwen_image_sft import (
    CoRTAtomicDataset,
    collate_atomic_entries,
    compute_batch_loss,
    normalize_entry_type,
    should_use_edit_pipeline,
    validate_pipeline_entry_types,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def _create_dummy_data(tmp_path: Path) -> dict[str, Path]:
    image_dir = tmp_path / "atomic_images"
    image_dir.mkdir(parents=True)
    _save_image(image_dir / "t2i.png", (30, 180, 90))
    _save_image(image_dir / "edit_src.png", (180, 180, 30))
    _save_image(image_dir / "edit_tgt.png", (30, 180, 180))

    atomic_train = tmp_path / "atomic_train.jsonl"
    _write_jsonl(
        atomic_train,
        [
            {
                "entry_type": "t2i",
                "sample_id": "atomic_t2i_0",
                "prompt": "a green square",
                "target_image": str(image_dir / "t2i.png"),
            },
            {
                "entry_type": "edit",
                "sample_id": "atomic_edit_0",
                "prompt": "make the square cyan",
                "source_image": str(image_dir / "edit_src.png"),
                "target_image": str(image_dir / "edit_tgt.png"),
                "reflection": "<problem>the square is yellow</problem>\n<fix>make the square cyan</fix>",
            },
        ],
    )

    sample_dir = tmp_path / "intermediate" / "echo4o" / "bagel" / "echo4o_dummy_0"
    sample_dir.mkdir(parents=True)
    _save_image(sample_dir / "img0.png", (220, 40, 40))
    _save_image(sample_dir / "img1.png", (40, 80, 220))
    (sample_dir / "meta.json").write_text("{}", encoding="utf-8")
    cort_train = tmp_path / "cort_train.jsonl"
    _write_jsonl(
        cort_train,
        [
            {
                "sample_id": "echo4o_dummy_0",
                "source": "echo4o",
                "generator_model": "bagel",
                "prompt": "make a blue square",
                "num_turns": 1,
                "reflection1": "<problem>the square is red</problem>\n<fix>make the square blue</fix>",
                "gt_img": "img1",
            }
        ],
    )
    return {"atomic_train": atomic_train, "cort_train": cort_train}


def _dataset_kwargs(tmp_path: Path):
    return {
        "cort_intermediate_dirs": [str(tmp_path / "intermediate")],
        "train_entry_types": {"t2i", "edit"},
        "cort_t2i_target": "final",
        "edit_prompt_mode": "prompt_fix",
        "edit_as_t2i": False,
    }


def test_dummy_atomic_and_cort_rows_expand_to_trainable_entries(tmp_path):
    manifest = _create_dummy_data(tmp_path)

    atomic_dataset = CoRTAtomicDataset([str(manifest["atomic_train"])], **_dataset_kwargs(tmp_path))
    assert [entry.entry_type for entry in atomic_dataset] == ["t2i", "edit"]
    assert all(Path(entry.target_image).exists() for entry in atomic_dataset)

    cort_dataset = CoRTAtomicDataset([str(manifest["cort_train"])], **_dataset_kwargs(tmp_path))
    assert [entry.entry_type for entry in cort_dataset] == ["t2i", "edit"]
    assert Path(cort_dataset[0].target_image).name == "img1.png"
    assert Path(cort_dataset[1].source_image).name == "img0.png"
    assert Path(cort_dataset[1].target_image).name == "img1.png"


def test_entry_type_aliases_are_normalized_and_loaded(tmp_path):
    source = tmp_path / "src.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(source)
    Image.new("RGB", (16, 16), (0, 0, 255)).save(target)

    data_file = tmp_path / "aliases.jsonl"
    row = {
        "entry_type": "gen_edit",
        "sample_id": "alias_0",
        "prompt": "make it blue",
        "source_image": str(source),
        "target_image": str(target),
        "reflection": {"problem": "it is red", "fix": "make it blue"},
    }
    data_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = CoRTAtomicDataset([str(data_file)], **_dataset_kwargs(tmp_path))
    assert normalize_entry_type("gen_edit") == "edit"
    assert len(dataset) == 1
    assert dataset[0].entry_type == "edit"
    assert "Edit instruction: make it blue" in dataset[0].prompt


class _FakeLatentDist:
    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents


class _FakeEncoderOutput:
    def __init__(self, latents):
        self.latent_dist = _FakeLatentDist(latents)


class _FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0])

    def encode(self, pixel_values):
        batch_size = pixel_values.shape[0]
        height = pixel_values.shape[-2] // 8
        width = pixel_values.shape[-1] // 8
        latents = torch.zeros(
            (batch_size, 1, 1, height, width),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        return _FakeEncoderOutput(latents)


class _FakeTransformer:
    def __init__(self):
        self.config = SimpleNamespace(guidance_embeds=False)
        self.last_kwargs = {}

    def __call__(self, hidden_states, **kwargs):
        self.last_kwargs = kwargs
        return (torch.zeros_like(hidden_states),)


class _FakePipe:
    def __init__(self):
        self.vae = _FakeVAE()
        self.transformer = _FakeTransformer()
        self.vae_scale_factor = 8
        self.processor = object()

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents[:, :, 0]
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    def encode_prompt(self, prompt, device, num_images_per_prompt, max_sequence_length, image=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        seq_len = 3 if image is not None else 2
        embeds = torch.ones((batch_size, seq_len, 4), device=device)
        mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        return embeds, mask


def test_compute_batch_loss_handles_mixed_t2i_and_edit_entries_with_fake_pipe(tmp_path):
    manifest = _create_dummy_data(tmp_path)
    dataset = CoRTAtomicDataset([str(manifest["atomic_train"])], **_dataset_kwargs(tmp_path))
    batch = collate_atomic_entries([dataset[0], dataset[1]])
    args = SimpleNamespace(
        height=16,
        width=16,
        max_sequence_length=8,
        edit_as_t2i=False,
    )

    pipe = _FakePipe()
    loss = compute_batch_loss(pipe, batch, args, torch.device("cpu"), torch.float32)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert pipe.transformer.last_kwargs["encoder_hidden_states_mask"] is not None


def test_pipeline_selection_rejects_plain_t2i_on_edit_pipeline():
    args = SimpleNamespace(pipeline_class="auto", edit_as_t2i=False, train_entry_types="edit,t2i")
    assert should_use_edit_pipeline(args, {"edit", "t2i"})
    with pytest.raises(ValueError, match="cannot train plain t2i rows"):
        validate_pipeline_entry_types(args, {"edit", "t2i"})

    args = SimpleNamespace(pipeline_class="auto", edit_as_t2i=True, train_entry_types="edit,t2i")
    assert not should_use_edit_pipeline(args, {"edit", "t2i"})
    validate_pipeline_entry_types(args, {"edit", "t2i"})
