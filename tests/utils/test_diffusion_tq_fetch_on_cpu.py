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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import NonTensorStack, TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import BatchData
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.rollout_correction import compute_rollout_corr_metrics_from_batch
from verl_omni.trainer.diffusion.v1 import tq_utils, trainer_base
from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_KEYS, DiffusionTrajectoryFieldsConfig
from verl_omni.workers.diffusion_tq import materialize_diffusion_trajectory
from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

FIELDS = DiffusionTrajectoryFieldsConfig(
    required=["all_latents", "prompt_embeds", "pooled_prompt_embeds"],
    optional=["prompt_embeds_mask", "negative_prompt_embeds"],
)


@pytest.fixture
def storage(monkeypatch):
    payload = TensorDict(
        {
            "prompts": torch.zeros(2, 4, dtype=torch.long),
            "uid": NonTensorStack("a", "b"),
            "rm_scores": torch.tensor([[1.0], [3.0]]),
            "responses": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
            "media_artifact__image_preview": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
            "extra_fields": NonTensorStack(
                *({"reward_extra_info": {"brightness": float(index)}} for index in range(2))
            ),
            "all_latents": torch.arange(96, dtype=torch.float32).reshape(2, 3, 1, 4, 4),
            "all_timesteps": torch.tensor([[900.0, 500.0], [900.0, 500.0]]),
            "prompt_embeds": torch.arange(64, dtype=torch.float32).reshape(2, 4, 8),
            "prompt_embeds_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "pooled_prompt_embeds": torch.ones(2, 4),
            "rollout_log_probs": torch.zeros(2, 2),
        },
        batch_size=[2],
    )
    state = SimpleNamespace(payload=payload, reads=[], missing_keys=set(), last_read=None)

    def get(*, keys, partition_id, select_fields):
        assert partition_id == "train"
        state.reads.append((list(keys), select_fields))
        indices = [0 if key.startswith("a_") else 1 for key in keys if key not in state.missing_keys]
        selected = state.payload
        if select_fields is not None:
            selected = selected.select(*(field for field in select_fields if field in selected.keys()))
        nested = [key for key, value in selected.items() if isinstance(value, torch.Tensor) and value.is_nested]
        result = selected.exclude(*nested)[indices]
        for field in nested:
            rows = selected[field].unbind()
            result[field] = torch.nested.as_nested_tensor([rows[index] for index in indices], layout=torch.jagged)
        state.last_read = result
        return result

    monkeypatch.setattr(tq_utils.tq, "kv_batch_get", get)
    return state


def read_training_batch(projected=True):
    trainer = SimpleNamespace(worker_side_trajectory=projected, tokenizer=SimpleNamespace(pad_token_id=0))
    meta = KVBatchMeta(keys=["b_0_0", "a_0_0"], tags=[{}, {}], partition_id="train")
    return trainer_base.PolicyGradientDiffusionTrainerV1._read_training_batch(trainer, meta)


def compose_trainer(overrides):
    with initialize_config_dir(config_dir=str(Path(verl_omni.__file__).parent / "trainer/config"), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


def test_projection_carries_only_row_keys_and_preserves_driver_diagnostics(storage):
    projected = read_training_batch()
    fields = storage.reads[0][1]
    assert set(fields) == {"prompts", "uid", "rm_scores", "all_timesteps", "extra_fields", "rollout_log_probs"}
    assert projected.meta_info == {}
    assert projected.non_tensor_batch[DIFFUSION_TQ_KEYS].tolist() == ["a_0_0", "b_0_0"]
    assert [info["brightness"] for info in projected.non_tensor_batch["reward_extra_info"]] == [0.0, 1.0]
    full = read_training_batch(projected=False)
    assert DIFFUSION_TQ_KEYS not in full.non_tensor_batch
    for data in (projected, full):
        data.batch["old_log_probs"] = torch.tensor([[0.1, -0.1], [0.2, -0.2]])
    before = compute_rollout_corr_metrics_from_batch(full, bypass_mode=False)
    after = compute_rollout_corr_metrics_from_batch(projected, bypass_mode=False)
    assert any("/ts_" in key for key in before)
    assert after == before


def test_configured_worker_fields_and_training_artifact_follow_reordered_rows(storage):
    config = compose_trainer(
        [
            "actor_rollout_ref.model.trajectory_fields.required=[denoise_states,media_artifact__source_latent]",
            "actor_rollout_ref.model.trajectory_fields.optional=[unused_field,advantages]",
        ]
    )
    fields = omega_conf_to_dataclass(config.actor_rollout_ref.model.trajectory_fields)
    assert isinstance(fields, DiffusionTrajectoryFieldsConfig)
    states = storage.payload["all_latents"]
    storage.payload["denoise_states"] = states
    storage.payload["media_artifact__source_latent"] = states[:, 0]
    storage.payload["advantages"] = torch.zeros(2, 2)
    data = read_training_batch().select_idxs([1, 0, 1]).to_tensordict()
    data["advantages"] = torch.ones(3, 2)

    result = materialize_diffusion_trajectory(data, fields)
    assert storage.reads[1] == (["b_0_0", "a_0_0"], fields.required + fields.optional)
    assert len(storage.reads) == 2
    assert DIFFUSION_TQ_KEYS not in result
    assert "media_artifact__image_preview" not in result
    assert "unused_field" not in result
    torch.testing.assert_close(result["denoise_states"], states[[1, 0, 1]])
    torch.testing.assert_close(result["media_artifact__source_latent"], states[[1, 0, 1], 0])
    torch.testing.assert_close(result["advantages"], data["advantages"])


@pytest.mark.parametrize("stage,field", [("driver", "rm_scores"), ("worker", "all_latents")])
def test_missing_required_fields_fail_without_full_read(storage, stage, field):
    storage.payload = storage.payload.exclude(field)
    with pytest.raises(ValueError, match=rf"(?i)missing required .*fields.*{field}"):
        data = read_training_batch()
        if stage == "worker":
            materialize_diffusion_trajectory(data.to_tensordict(), FIELDS)
    assert len(storage.reads) == (1 if stage == "driver" else 2)
    assert all(selected is not None for _, selected in storage.reads)


def test_worker_reports_row_dropped_by_tq(storage):
    data = read_training_batch().to_tensordict()
    storage.missing_keys.add("b_0_0")
    with pytest.raises(ValueError, match="row count mismatch: expected 2, got 1"):
        materialize_diffusion_trajectory(data, FIELDS)


def test_data_parallel_dispatch_keeps_keys_aligned(storage):
    data = read_training_batch().to_tensordict()
    for index, chunk in enumerate(BatchData(data).chunk(chunks=2)):
        result = materialize_diffusion_trajectory(chunk, FIELDS)
        torch.testing.assert_close(result["all_latents"], storage.payload["all_latents"][index : index + 1])


def test_worker_reuses_fetched_latents_for_unique_rows(storage):
    data = read_training_batch().select_idxs([1, 0]).to_tensordict()
    result = materialize_diffusion_trajectory(data, FIELDS)
    torch.testing.assert_close(result["all_latents"], storage.payload["all_latents"][[1, 0]])
    assert result["all_latents"].data_ptr() == storage.last_read["all_latents"].data_ptr()


@pytest.mark.parametrize("method", ["infer_actor_batch", "infer_ref_batch", "update_actor"])
def test_worker_entrypoints_keep_tensor_dict_rpc_contract(storage, method):
    data = read_training_batch().to_tensordict()
    expected = TensorDict({"log_probs": torch.ones(2, 2)}, batch_size=[2])

    def compute(data):
        torch.testing.assert_close(data["all_latents"], storage.payload["all_latents"])
        assert DIFFUSION_TQ_KEYS not in data
        assert tu.get_non_tensor_data(data, "enable_routing_replay", None) == (method != "infer_ref_batch")
        return expected

    engine = SimpleNamespace(infer_batch=compute, train_mini_batch=compute)
    worker = SimpleNamespace(
        actor=engine,
        ref=engine,
        _trajectory_fields=FIELDS,
        enable_routing_replay=True,
        config=SimpleNamespace(actor={}),
    )
    result = getattr(ActorRolloutRefWorker, method)(worker, data)
    torch.testing.assert_close(result["log_probs"], expected["log_probs"])
    assert len(storage.reads) == 2


@pytest.fixture
def ragged_storage(storage):
    storage.payload["prompt_embeds"] = torch.nested.as_nested_tensor(
        [torch.arange(24).reshape(3, 8).float(), torch.arange(16).reshape(2, 8).float()], layout=torch.jagged
    )
    storage.payload["prompt_embeds_mask"] = torch.nested.as_nested_tensor(
        [torch.ones(3, dtype=torch.long), torch.ones(2, dtype=torch.long)], layout=torch.jagged
    )
    return storage


@pytest.mark.parametrize(
    ("mask", "order"),
    [
        pytest.param(None, [1, 0, 1], id="nested"),
        pytest.param([[1, 1, 0], [1, 1, 0]], [1, 0, 1], id="dense-subset"),
        pytest.param([[1, 1, 0], [1, 1, 0]], [1], id="short-shard"),
    ],
)
def test_ragged_embeds_match_full_read(ragged_storage, mask, order):
    if mask is not None:
        ragged_storage.payload["prompt_embeds_mask"] = torch.tensor(mask, dtype=torch.long)
    full = read_training_batch(projected=False).select_idxs(order)
    projected = read_training_batch().select_idxs(order)
    expected = embeds_padding_2_no_padding(full.to_tensordict())
    actual = materialize_diffusion_trajectory(projected.to_tensordict(), FIELDS)
    for key in ("prompt_embeds", "prompt_embeds_mask"):
        for expected_row, actual_row in zip(expected[key].unbind(), actual[key].unbind(), strict=True):
            torch.testing.assert_close(expected_row, actual_row, rtol=0, atol=0)


@pytest.mark.parametrize("mask", [None, [[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1], [1, 0, 1]]])
def test_invalid_mask_on_ragged_embeds_fails_closed(ragged_storage, mask):
    if mask is None:
        ragged_storage.payload = ragged_storage.payload.exclude("prompt_embeds_mask")
    else:
        ragged_storage.payload["prompt_embeds_mask"] = torch.tensor(mask, dtype=torch.long)
    with pytest.raises(ValueError, match="requires 'prompt_embeds_mask'|incompatible 'prompt_embeds_mask'"):
        materialize_diffusion_trajectory(read_training_batch().to_tensordict(), FIELDS)


@pytest.mark.parametrize("optional", [["all_latents"], ["all_timesteps"], [DIFFUSION_TQ_KEYS], [""]])
def test_invalid_field_groups_rejected_during_config_construction(optional):
    with pytest.raises(ValueError, match="distinct non-empty names|driver-owned"):
        DiffusionTrajectoryFieldsConfig(required=["all_latents"], optional=optional)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["trainer.v1.worker_side_trajectory=bad"], "must be a boolean"),
        (["trainer.v1.worker_side_trajectory=true", "trainer.v1.trainer_mode=separate_async"], "trainer_mode=sync"),
        (["trainer.v1.worker_side_trajectory=true", "reward.reward_model.enable=true"], "rollout-side rewards"),
        (["trainer.v1.worker_side_trajectory=true", "distillation.enabled=true"], "no teacher"),
        (["trainer.v1.worker_side_trajectory=true"], "trajectory_fields.required"),
        (
            [
                "trainer.v1.worker_side_trajectory=true",
                "algorithm.trainer_type=direct_preference",
                "actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo",
            ],
            "direct-preference",
        ),
    ],
)
def test_unsupported_enabled_modes_fail_at_trainer_construction(overrides, message):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    with pytest.raises(ValueError, match=message):
        PolicyGradientDiffusionTrainerV1Sync(compose_trainer(overrides))
