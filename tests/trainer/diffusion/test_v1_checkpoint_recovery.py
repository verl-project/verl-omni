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
"""Real TransferQueue recovery and named-media integration checks for L2 smoke."""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from PIL import Image
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu

from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ
from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, artifact_fields, previews_from_batch
from verl_omni.reward_loop.reward_manager.visual import _reward_extra_info, _validate_visual_response
from verl_omni.trainer.diffusion.v1 import tq_utils
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync


@pytest.fixture(scope="module")
def initialized_tq():
    """Own a bounded GPU-free Ray cluster and close its queue after the module."""
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_gpus=0, num_cpus=4, object_store_memory=128 * 1024**2, include_dashboard=False)
    tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 2}}}))
    try:
        yield
    finally:
        tq.close()
        if owns_ray:
            ray.shutdown()


@pytest.mark.parametrize("trainer_mode", ["separate_async", "colocate_async"])
@pytest.mark.parametrize("preview_name", ["image_preview", None])
def test_transfer_queue_checkpoint_roundtrip_and_reissue(
    initialized_tq, tmp_path, monkeypatch, trainer_mode, preview_name
):
    """Round-trip named rows, then check reissue cleanup, reward and sorted preview export."""
    partition_id = f"test-{uuid.uuid4().hex}"
    pending, finished = uuid.uuid4().hex, uuid.uuid4().hex
    prompt_batch = tu.get_tensordict(
        {
            "uid": [pending, finished],
            "raw_prompt": ["pending prompt", "finished prompt"],
            "index": torch.tensor([0, 1]),
            "rollout_seed_index": torch.tensor([4, 5]),
        }
    )
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.trainer_mode = trainer_mode
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2, "gen_batch_size": 1},
            "trainer": {"v1": {"sampler": {"drop_incomplete_groups": False}}},
            "actor_rollout_ref": {"rollout": {"seed": 42}},
        }
    )
    chunks = iter(prompt_batch.split(1))
    trainer._rollout_seed_step = None
    trainer._rollout_seed_next_index = 0
    trainer._fetch_one_gen_batch = lambda: next(chunks)
    prompt_batch = trainer._next_train_batch()
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: None)
    real_put = tq.kv_batch_put
    with monkeypatch.context() as patch:
        patch.setattr(tq, "kv_batch_put", lambda **kwargs: real_put(**{**kwargs, "partition_id": partition_id}))
        assert trainer._submit_batch_to_rollout(prompt_batch) == 2
    tq.kv_batch_put(
        keys=[pending, finished],
        partition_id=partition_id,
        tags=[
            {"is_prompt": True, "status": "running", "global_steps": 3},
            {"is_prompt": True, "status": "finished", "global_steps": 3},
        ],
    )
    pending_trajectory = f"{pending}_0_0"
    finished_trajectory = f"{finished}_0_0"
    consumed_trajectory = f"{uuid.uuid4().hex}_0_0"
    for key in (pending_trajectory, consumed_trajectory):
        tq.kv_put(
            key=key,
            partition_id=partition_id,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "global_steps": 3},
        )

    worker = object.__new__(DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class)
    outputs = []
    for session, value in enumerate((41, 179)):
        artifacts = {
            "image_latent": MediaArtifact(
                MediaSpec("image", "latent", "CHW"),
                torch.full((16, 2, 2), float(value)),
                context=f"request={finished}, session={session}",
            ),
            "image_preview": MediaArtifact(
                MediaSpec("image", "decoded", "CHW"), torch.full((3, 4, 4), value, dtype=torch.uint8)
            ),
            "aux_preview": MediaArtifact(
                MediaSpec("image", "decoded", "CHW"), torch.full((3, 4, 4), 255 - value, dtype=torch.uint8)
            ),
        }
        fields = artifact_fields(artifacts, "image_latent", preview_name)
        fields.update(global_steps=3 + session, reward_extra_info={"ocr": value / 255})
        outputs.append(
            (
                session,
                SimpleNamespace(
                    prompt_ids=torch.tensor([[value]]),
                    response_diffusion_output=artifacts["image_latent"].data.unsqueeze(0),
                    response_logprobs=None,
                    reward_score=float(value),
                    num_turns=1,
                    extra_fields={k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in fields.items()},
                ),
            )
        )
    real_async_put = tq.async_kv_batch_put

    async def put_in_partition(**kwargs):
        await real_async_put(**{**kwargs, "partition_id": partition_id})

    with monkeypatch.context() as patch:
        patch.setattr(tq, "async_kv_batch_put", put_in_partition)
        asyncio.run(
            worker._write_trajectories_to_tq(
                outputs,
                uid=finished,
                trajectory={"step": 3},
                validate=False,
                reward_model={"ground_truth": "ground truth"},
                data_source="named-media",
            )
        )

    checkpoint_dir = tmp_path / "transfer_queue"
    tq.save_checkpoint(checkpoint_dir, metadata={"global_steps": 3})
    tq.kv_clear(keys=list(tq.kv_list(partition_id)[partition_id]), partition_id=partition_id)
    tq.load_checkpoint(checkpoint_dir)

    submitted = []
    trainer.global_steps = 4
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"seed": 7}}, "trainer": {}})
    trainer._rollout_seed_step = None
    trainer._rollout_seed_next_index = 0
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=submitted.append)
    try:
        assert trainer._reissue_inflight_prompts(partition_id) == 1
        items = tq.kv_list(partition_id)[partition_id]
        assert items[pending]["status"] == "pending"
        assert items[finished]["status"] == "finished"
        assert pending_trajectory not in items
        assert consumed_trajectory not in items
        assert finished_trajectory in items
        assert list(submitted[0]["uid"]) == [pending]
        assert list(submitted[0]["raw_prompt"]) == ["pending prompt"]
        assert int(submitted[0]["global_steps"]) == 4
        assert int(submitted[0]["rollout_seed"][0]) == 44
        assert int(submitted[0]["rollout_seed_index"][0]) == 0
        persisted_prompt = tq.kv_batch_get(keys=[pending], partition_id=partition_id)
        assert list(persisted_prompt["uid"]) == [pending]
        assert list(persisted_prompt["raw_prompt"]) == ["pending prompt"]
        assert int(persisted_prompt["index"][0]) == 0
        assert int(persisted_prompt["rollout_seed"][0]) == 44
        assert int(persisted_prompt["rollout_seed_index"][0]) == 0

        # Read in reverse session order; tensors, declarations and row metadata
        # must stay aligned through the real TQ checkpoint and retry cleanup.
        keys = [f"{finished}_1_0", finished_trajectory]
        meta = KVBatchMeta(partition_id=partition_id, keys=keys, tags=[items[k] for k in keys])
        data = tq_utils.diffusion_tq_batch_to_dataproto(meta)
        assert data.non_tensor_batch["preview_artifact"].tolist() == [preview_name] * 2
        assert data.non_tensor_batch["min_global_steps"].tolist() == [4, 3]
        assert data.non_tensor_batch["max_global_steps"].tolist() == [4, 3]
        for row, value, session in zip(data, (179, 41), (1, 0), strict=True):
            extra = _reward_extra_info(row)
            assert extra["media_artifact_context"] == f"request={finished}, session={session}"
            assert row.non_tensor_batch["reward_extra_info"]["ocr"] == pytest.approx(value / 255)
            assert "preview_artifact" in extra and extra["preview_artifact"] == preview_name
            _validate_visual_response(row.batch["responses"], None, is_validate=False, extra_info=extra)
            assert set(extra["media_artifacts"]) == {"image_latent", "image_preview", "aux_preview"}
            assert torch.all(extra["media_artifacts"]["image_latent"].data == value)
            assert torch.all(extra["media_artifacts"]["image_preview"].data == value)
            assert torch.all(extra["media_artifacts"]["aux_preview"].data == 255 - value)
            assert torch.all(row.batch["responses"] == value)
        previews = previews_from_batch(data)
        if preview_name is None:
            assert previews == [None, None]
        else:
            assert [int(preview.data[0, 0, 0]) for preview in previews] == [179, 41]

        trainer.tokenizer = SimpleNamespace(
            pad_token_id=0, batch_decode=lambda ids, **kwargs: [str(int(row[0])) for row in ids]
        )
        trainer._init_dump_executor()
        try:
            trainer._log_rollout_data(meta, {}, str(tmp_path / "dump"))
        finally:
            trainer._shutdown_dump_executor()
        dump_file = tmp_path / "dump/4.jsonl"
        if preview_name is None:
            assert not dump_file.exists()
        else:
            rows = [json.loads(line) for line in dump_file.read_text().splitlines()]
            assert [row["uid"] for row in rows] == [finished_trajectory, keys[0]]
            assert [row["input"] for row in rows] == ["41", "179"]
            assert [row["score"] for row in rows] == [41, 179]
            assert [row["gts"] for row in rows] == ["ground truth"] * 2
            for row, value in zip(rows, (41, 179), strict=True):
                with Image.open(row["output"]) as image:
                    assert image.getpixel((0, 0)) == (value,) * 3
    finally:
        tq.kv_clear(keys=list(tq.kv_list(partition_id)[partition_id]), partition_id=partition_id)
