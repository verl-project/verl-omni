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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict
from transfer_queue import BatchMeta, KVBatchMeta
from transfer_queue.controller import FieldMeta
from transfer_queue.metadata import extract_field_schema
from verl.protocol import BatchData
from verl.utils import tensordict_utils as tu

import verl_omni
from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import StableDiffusion3FlowGRPO
from verl_omni.pipelines.wan22_dance_grpo.diffusers_training_adapter import Wan22DanceGRPO
from verl_omni.trainer.diffusion.rollout_correction import compute_rollout_corr_metrics_from_batch
from verl_omni.trainer.diffusion.v1 import tq_utils, trainer_base
from verl_omni.utils import diffusion_tq
from verl_omni.workers.engine_workers import ActorRolloutRefWorker, _with_routing_replay_flag
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

REQUIRED = {"all_latents", "prompt_embeds", "pooled_prompt_embeds"}
OPTIONAL = {"prompt_embeds_mask", "negative_prompt_embeds"}


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
        },
        batch_size=[2],
    )
    state = SimpleNamespace(
        payload=payload, core_reads=[], worker_reads=[], ready=True, drop_fetched=None, schema_overrides={}
    )

    def retrieve(*, keys, partition_id, create=False):
        indices = [0 if key.startswith("a_") else 1 for key in keys]
        schema = extract_field_schema(state.payload)
        schema.update(state.schema_overrides)
        return BatchMeta(
            global_indexes=indices,
            partition_ids=[partition_id] * len(keys),
            field_schema=schema,
            production_status=np.full(len(keys), int(state.ready), dtype=np.int8),
        )

    def select_payload(indices, fields=None):
        payload = state.payload if fields is None else state.payload.select(*fields)
        nested_fields = [key for key, value in payload.items() if isinstance(value, torch.Tensor) and value.is_nested]
        result = payload.exclude(*nested_fields)[indices]
        for field in nested_fields:
            rows = payload[field].unbind()
            result[field] = torch.nested.as_nested_tensor([rows[index] for index in indices], layout=torch.jagged)
        return result

    def core_read(*, keys, partition_id, select_fields):
        state.core_reads.append(select_fields)
        indices = [0 if key.startswith("a_") else 1 for key in keys]
        return select_payload(indices, select_fields)

    def bridge(meta):
        state.worker_reads.append(meta)
        return retrieve(keys=meta.keys, partition_id=meta.partition_id).select_fields(meta.fields)

    def worker_read(meta):
        result = select_payload(meta.global_indexes, meta.field_names)
        if state.drop_fetched is not None:
            result = result.exclude(state.drop_fetched)
        return result

    client = SimpleNamespace(kv_retrieve_meta=retrieve, get_data=worker_read)
    monkeypatch.setattr(tq_utils.tq, "get_client", lambda: client)
    monkeypatch.setattr(tq_utils.tq, "kv_batch_get", core_read)
    monkeypatch.setattr(diffusion_tq, "kv_batch_meta2batch_meta", bridge)
    return state


def read_core():
    return tq_utils.diffusion_tq_batch_to_dataproto(
        KVBatchMeta(keys=["b_0_0", "a_0_0"], tags=[{}, {}], partition_id="train"),
        trajectory_fields=(REQUIRED, OPTIONAL),
    )


def test_core_read_uses_explicit_allowlist_and_records_actual_sde_window(storage):
    data = read_core()
    assert REQUIRED.isdisjoint(data.batch.keys())
    assert set(storage.core_reads[0]) == {"prompts", "uid", "rm_scores", "all_timesteps", "extra_fields"}
    assert "responses" not in data.batch and "media_artifact__image_preview" not in data.batch
    assert [item["brightness"] for item in data.non_tensor_batch["reward_extra_info"]] == [0.0, 1.0]
    assert data.non_tensor_batch[diffusion_tq.TRAJECTORY_KEYS].tolist() == ["a_0_0", "b_0_0"]
    assert data.meta_info[diffusion_tq.TRAJECTORY_META]["num_timesteps"] == 2
    assert "negative_prompt_embeds" not in data.meta_info[diffusion_tq.TRAJECTORY_META]["fields"]


def test_trainer_read_wires_enabled_field_groups(storage):
    trainer = SimpleNamespace(
        worker_side_trajectory=True,
        _trajectory_field_groups=(REQUIRED, OPTIONAL),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    data = trainer_base.PolicyGradientDiffusionTrainerV1._read_training_batch(
        trainer, KVBatchMeta(keys=["a_0_0", "b_0_0"], tags=[{}, {}], partition_id="train")
    )
    assert "all_latents" not in data.batch.keys()
    assert diffusion_tq.TRAJECTORY_META in data.meta_info


def test_non_tq_worker_batches_keep_their_existing_contract():
    batch = TensorDict({"input_ids": torch.ones(2, 4, dtype=torch.long)}, batch_size=[2])
    assert diffusion_tq.materialize_diffusion_trajectory(batch) is batch


def test_worker_fetch_preserves_reordered_padded_rows_and_driver_overlays(storage):
    data = read_core().select_idxs([1, 0, 1, 1])
    data.batch["advantages"] = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    sent = data.to_tensordict()
    output = diffusion_tq.materialize_diffusion_trajectory(sent)
    torch.testing.assert_close(output["all_latents"], storage.payload["all_latents"][[1, 0, 1, 1]])
    torch.testing.assert_close(output["advantages"], data.batch["advantages"])
    assert storage.worker_reads[0].keys == ["b_0_0", "a_0_0"]
    assert set(storage.worker_reads[0].fields) == REQUIRED | {"prompt_embeds_mask"}
    assert [len(row) for row in output["prompt_embeds"].unbind()] == [2, 3, 2, 2]
    assert diffusion_tq.TRAJECTORY_KEYS not in output.keys()
    assert diffusion_tq.TRAJECTORY_KEYS in sent.keys()
    assert "all_latents" not in data.batch.keys()


def test_real_dp_dispatch_preserves_single_row_key_handles(storage):
    chunks = BatchData(read_core().to_tensordict()).chunk(chunks=2)
    for index, chunk in enumerate(chunks):
        result = diffusion_tq.materialize_diffusion_trajectory(chunk)
        torch.testing.assert_close(result["all_latents"], storage.payload["all_latents"][index : index + 1])
    assert [meta.keys for meta in storage.worker_reads] == [["a_0_0"], ["b_0_0"]]


def test_driver_rejects_misaligned_transition_shape_before_read(storage):
    storage.payload["all_latents"] = torch.zeros(2, 4, 1, 4, 4)
    with pytest.raises(ValueError, match="S\\+1 latents"):
        read_core()
    assert storage.core_reads == []


@pytest.mark.parametrize("missing", sorted(REQUIRED | {"prompts", "uid", "rm_scores", "all_timesteps"}))
def test_required_field_miss_never_falls_back_to_full_read(storage, missing):
    storage.payload = storage.payload.exclude(missing)
    with pytest.raises(ValueError, match="Missing training fields"):
        read_core()
    assert storage.core_reads == []


def test_worker_rejects_field_disappearing_after_core_read(storage):
    data = read_core().to_tensordict()
    storage.payload = storage.payload.exclude("all_latents")
    with pytest.raises(ValueError, match="missing fields=.*all_latents"):
        diffusion_tq.materialize_diffusion_trajectory(data)


def test_worker_rejects_missing_returned_data(storage):
    data = read_core().to_tensordict()
    storage.drop_fetched = "prompt_embeds"
    with pytest.raises(ValueError, match="Missing diffusion trajectory data"):
        diffusion_tq.materialize_diffusion_trajectory(data)


@pytest.mark.parametrize("stage", ["driver", "worker"])
def test_unready_rows_fail_closed(storage, stage):
    data = read_core() if stage == "worker" else None
    storage.ready = False
    with pytest.raises(ValueError, match="not ready|Invalid diffusion trajectory metadata"):
        read_core() if data is None else diffusion_tq.materialize_diffusion_trajectory(data.to_tensordict())


def test_worker_rejects_stale_trajectory_overriding_driver_data(storage):
    data = read_core().to_tensordict()
    data["all_latents"] = storage.payload["all_latents"]
    with pytest.raises(ValueError, match="conflicts with driver-owned"):
        diffusion_tq.materialize_diffusion_trajectory(data)


@pytest.mark.parametrize("field", [diffusion_tq.TRAJECTORY_KEYS, diffusion_tq.TRAJECTORY_META])
def test_incomplete_handle_raises(storage, field):
    data = read_core().to_tensordict().exclude(field)
    with pytest.raises(ValueError, match="Incomplete"):
        diffusion_tq.materialize_diffusion_trajectory(data)


@pytest.mark.parametrize("method", ["infer_actor_batch", "infer_ref_batch", "update_actor"])
def test_actual_worker_entrypoints_fetch_and_preserve_return_contract(storage, method):
    data = read_core().to_tensordict()
    expected = TensorDict({"log_probs": torch.ones(2, 2)}, batch_size=[2])

    def compute(data):
        torch.testing.assert_close(data["all_latents"], storage.payload["all_latents"])
        assert diffusion_tq.TRAJECTORY_KEYS not in data.keys()
        assert tu.get_non_tensor_data(data, "enable_routing_replay", None) == (method != "infer_ref_batch")
        return expected

    engine = SimpleNamespace(infer_batch=compute, train_mini_batch=compute)
    worker = SimpleNamespace(actor=engine, ref=engine, enable_routing_replay=True, config=SimpleNamespace(actor={}))
    result = getattr(ActorRolloutRefWorker, method)(worker, data)
    torch.testing.assert_close(result["log_probs"], expected["log_probs"])
    assert len(storage.worker_reads) == 1
    assert diffusion_tq.TRAJECTORY_KEYS in data.keys()


def test_routing_wrapper_does_not_fetch_trajectory(storage):
    data = read_core().to_tensordict()

    @_with_routing_replay_flag(enabled=False)
    def infer(self, batch):
        return batch

    assert infer(SimpleNamespace(enable_routing_replay=False), data) is data
    assert storage.worker_reads == []


def test_advantage_uses_core_timesteps_without_latent_read(storage, monkeypatch):
    data = read_core()
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {"algorithm": {"norm_adv_by_std_in_grpo": True, "adv_estimator": "grpo", "global_std": False}}
        )
    )
    monkeypatch.setattr(trainer_base, "compute_advantage", lambda data, **kwargs: data)
    result = trainer_base.PolicyGradientDiffusionTrainerV1._compute_advantage(trainer, data)
    assert result.batch["sample_level_rewards"].shape == (2, 2)
    torch.testing.assert_close(result.batch["sample_level_rewards"], torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    assert len(storage.core_reads) == 1 and storage.worker_reads == []


def test_projection_preserves_per_timestep_rollout_diagnostics(storage):
    storage.payload["all_timesteps"][1] = torch.tensor([850.0, 450.0])
    storage.payload["rollout_log_probs"] = torch.zeros(2, 2)
    meta = KVBatchMeta(keys=["a_0_0", "b_0_0"], tags=[{}, {}], partition_id="train")
    full = tq_utils.diffusion_tq_batch_to_dataproto(meta)
    projected = read_core()
    timestep_metrics = []
    for data in (full, projected):
        data.batch["old_log_probs"] = torch.ones(2, 2)
        metrics = compute_rollout_corr_metrics_from_batch(data, bypass_mode=False)
        timestep_metrics.append({key: value for key, value in metrics.items() if "/ts_" in key})
    assert len(timestep_metrics[0]) == 4
    assert timestep_metrics[0] == timestep_metrics[1]
    assert storage.worker_reads == []


@pytest.mark.parametrize("field", ["all_latents", "all_timesteps"])
@pytest.mark.parametrize("select_other_shape", [False, True])
def test_partition_shapes_are_checked_only_for_selected_rows(storage, field, select_other_shape):
    schema = extract_field_schema(storage.payload)[field]
    partition = FieldMeta(global_indexes={0, 1}, **schema)
    other_shape = (3,) if field == "all_timesteps" else (3, 1, 8, 8)
    partition.update({**schema, "shape": other_shape}, [2])
    indexes = [0, 2] if select_other_shape else [0, 1]
    storage.schema_overrides[field] = partition.to_batch_schema(indexes)
    if select_other_shape:
        with pytest.raises(ValueError, match=f"uniform '{field}' shapes"):
            read_core()
        assert storage.core_reads == []
    else:
        result = diffusion_tq.materialize_diffusion_trajectory(read_core().to_tensordict())
        torch.testing.assert_close(result["all_latents"], storage.payload["all_latents"])
        torch.testing.assert_close(result["all_timesteps"], storage.payload["all_timesteps"])


@pytest.mark.parametrize("field", ["all_latents", "all_timesteps"])
def test_missing_per_row_shape_metadata_fails_before_core_read(storage, field):
    storage.schema_overrides[field] = {"is_nested": True, "shape": None}
    with pytest.raises(ValueError, match="Missing per-row shapes"):
        read_core()
    assert storage.core_reads == []


def test_masked_ragged_embeds_match_full_read_after_reorder_and_duplication(storage):
    storage.payload["prompt_embeds"] = torch.nested.as_nested_tensor(
        [torch.arange(24).reshape(3, 8).float(), torch.arange(16).reshape(2, 8).float()], layout=torch.jagged
    )
    storage.payload["prompt_embeds_mask"] = torch.nested.as_nested_tensor(
        [torch.ones(3, dtype=torch.long), torch.ones(2, dtype=torch.long)], layout=torch.jagged
    )
    meta = KVBatchMeta(keys=["a_0_0", "b_0_0"], tags=[{}, {}], partition_id="train")
    full = tq_utils.diffusion_tq_batch_to_dataproto(meta).select_idxs([1, 0, 1])
    expected = embeds_padding_2_no_padding(full.to_tensordict())
    actual = diffusion_tq.materialize_diffusion_trajectory(read_core().select_idxs([1, 0, 1]).to_tensordict())
    for field in ("prompt_embeds", "prompt_embeds_mask"):
        for full_row, projected_row in zip(expected[field].unbind(), actual[field].unbind(), strict=True):
            torch.testing.assert_close(full_row, projected_row, rtol=0, atol=0)


def test_unmasked_ragged_embeds_fail_closed(storage):
    storage.payload["prompt_embeds"] = torch.nested.as_nested_tensor(
        [torch.ones(3, 8), torch.ones(2, 8)], layout=torch.jagged
    )
    storage.payload = storage.payload.exclude("prompt_embeds_mask")
    with pytest.raises(ValueError, match="requires 'prompt_embeds_mask'"):
        diffusion_tq.materialize_diffusion_trajectory(read_core().to_tensordict())


def test_ragged_embeds_reject_mismatched_mask(storage):
    storage.payload["prompt_embeds"] = torch.nested.as_nested_tensor(
        [torch.ones(3, 8), torch.ones(2, 8)], layout=torch.jagged
    )
    storage.payload["prompt_embeds_mask"] = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="incompatible 'prompt_embeds_mask'"):
        diffusion_tq.materialize_diffusion_trajectory(read_core().to_tensordict())


def test_timestep_fields_cannot_be_deferred(storage):
    meta = KVBatchMeta(keys=["a_0_0", "b_0_0"], tags=[{}, {}], partition_id="train")
    with pytest.raises(ValueError, match="all_timesteps must remain driver-owned"):
        tq_utils.diffusion_tq_batch_to_dataproto(meta, trajectory_fields=(REQUIRED | {"all_timesteps"}, OPTIONAL))
    assert storage.core_reads == []


def test_worker_rejects_missing_core_timesteps_before_fetch(storage):
    data = read_core().to_tensordict().exclude("all_timesteps")
    with pytest.raises(ValueError, match="driver-owned timesteps"):
        diffusion_tq.materialize_diffusion_trajectory(data)
    assert storage.worker_reads == []


@pytest.mark.parametrize("cfg", [1.0, 5.0])
def test_pipeline_contracts_require_cfg_fields_only_when_enabled(cfg):
    config = SimpleNamespace(pipeline=OmegaConf.create({"guidance_scale": cfg, "true_cfg_scale": cfg}))
    sd3_required, _ = StableDiffusion3FlowGRPO.tq_trajectory_fields(config)
    wan_required, _ = Wan22DanceGRPO.tq_trajectory_fields(config)
    assert ("negative_pooled_prompt_embeds" in sd3_required) == (cfg > 1)
    assert ("negative_prompt_embeds" in wan_required) == (cfg > 1)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["trainer.v1.worker_side_trajectory=bad"], "must be a boolean"),
        (
            ["trainer.v1.worker_side_trajectory=true", "trainer.v1.trainer_mode=separate_async"],
            "trainer_mode=sync",
        ),
        (["trainer.v1.worker_side_trajectory=true", "reward.reward_model.enable=true"], "rollout-side rewards"),
        (["trainer.v1.worker_side_trajectory=true", "distillation.enabled=true"], "no teacher"),
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
    from pathlib import Path

    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    with initialize_config_dir(config_dir=str(Path(verl_omni.__file__).parent / "trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=overrides)
    with pytest.raises(ValueError, match=message):
        PolicyGradientDiffusionTrainerV1Sync(config)
