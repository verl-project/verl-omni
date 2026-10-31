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
"""Streaming admission, result and accepted-work ownership contracts."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.reward_loop.streaming import create_async_reward_client, merge_reward_outputs
from verl_omni.utils.config import validate_config
from verl_omni.workers.config.reward import (
    StreamingRewardConfig,
    get_streaming_reward_config,
    streaming_reward_enabled,
)


def _sample(index=0):
    return DataProto.from_dict(tensors={"responses": torch.tensor([[index]])})


@pytest.mark.parametrize("keys", ["audio", [""], ["audio", "audio"], [1], [["audio"]]])
def test_streaming_validates_extra_field_selection(keys):
    with pytest.raises(ValueError, match="extra_fields"):
        StreamingRewardConfig(extra_fields=keys)


@pytest.mark.asyncio
@pytest.mark.parametrize("keys", [None, [], ["keep", "optional"]])
async def test_named_payload_projection_preserves_inputs_and_other_metadata(keys):
    fields = {"keep": torch.tensor([2]), "trajectory": torch.ones(128), "img_shapes": [(1, 32, 48)]}
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1]])},
        non_tensors={"tool_extra_fields": np.array([fields], dtype=object), "extra_info": np.array([{"index": 7}])},
        meta_info={"validate": True},
    )
    seen = []

    async def score(projected):
        seen.append(projected.non_tensor_batch["tool_extra_fields"][0])
        assert projected.non_tensor_batch["extra_info"][0] == {"index": 7}
        assert projected.meta_info["validate"] is True
        torch.testing.assert_close(projected.batch["responses"], data.batch["responses"])
        return {"reward_score": 1.0}

    config = get_streaming_reward_config(_config(extra_fields=keys))
    client = create_async_reward_client({"a": (_worker(score),), "b": (_worker(score),)}, config)
    try:
        result = await _score_sample(client, data)
        assert result["reward_score"] == 2.0
        expected = set(fields) if keys is None else set(fields).intersection([*keys, "img_shapes"])
        assert [set(value) for value in seen] == [expected, expected]
        assert all(value["img_shapes"] == fields["img_shapes"] for value in seen)
        assert set(data.non_tensor_batch["tool_extra_fields"][0]) == {"keep", "trajectory", "img_shapes"}
        assert data.non_tensor_batch["tool_extra_fields"][0] is fields
    finally:
        await client.close()


def _output(key, score):
    return {"reward_score": score, "reward_extra_info": {f"reward/{key}": score, "reward/combined": score}}


def _worker(fn):
    async def batch(data):
        return [await fn(data)]

    return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=lambda data: asyncio.create_task(batch(data))))


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_fields", [None, []])
@pytest.mark.parametrize("primary", ["image_preview", "image_latent"])
async def test_named_media_streaming_matches_batch_scoring(extra_fields, primary):
    artifacts = {
        "image_preview": MediaArtifact(
            MediaSpec("image", "decoded", "CHW"), torch.arange(48, dtype=torch.uint8).reshape(3, 4, 4)
        ),
        "image_latent": MediaArtifact(MediaSpec("image", "latent", "CHW"), torch.ones(16, 2, 2)),
    }
    fields = artifact_fields(artifacts, primary, "image_preview")
    fields["trajectory"] = torch.ones(4, 16, 2, 2)
    data = DataProto.from_dict(
        tensors={"responses": artifacts[primary].data.unsqueeze(0)},
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "prompt"}],
            "tool_extra_fields": [fields],
        },
    )
    manager = _jpeg_manager()
    expected = await manager.run_single(data)

    async def score(projected):
        received = projected.non_tensor_batch["tool_extra_fields"][0]
        assert set(received) == set(fields) - ({"trajectory"} if extra_fields == [] else set())
        return await manager.run_single(projected)

    client = create_async_reward_client({"jpeg": (_worker(score),)}, StreamingRewardConfig(extra_fields=extra_fields))
    try:
        assert await _score_sample(client, data) == expected
        assert data.non_tensor_batch["tool_extra_fields"][0] is fields
        assert "trajectory" in fields
    finally:
        await client.close()


def _jpeg_manager():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"pipeline": {"output_type": "pt"}}},
            "reward": {
                "reward_model": {"rollout": {}},
                "reward_functions": {
                    "jpeg": {
                        "path": "pkg://verl_omni.utils.reward_score.jpeg_compressibility",
                        "name": "compute_score",
                        "weight": 0.3,
                        "required": False,
                    }
                },
            },
        }
    )
    return MultiVisualRewardManager(config, tokenizer=None, compute_score=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_fields", [None, []])
@pytest.mark.parametrize("invalid", ["declarations", "tensor", "primary", "preview", "projection"])
async def test_streaming_preserves_fatal_media_errors_for_optional_reward(extra_fields, invalid):
    image = torch.zeros(3, 4, 4, dtype=torch.uint8)
    fields = artifact_fields({"image": MediaArtifact(MediaSpec("image", "decoded", "CHW"), image)}, "image", "image")
    if invalid == "declarations":
        del fields["media_artifact_specs"]
    elif invalid == "tensor":
        del fields["media_artifact__image"]
    elif invalid in {"primary", "preview"}:
        fields[f"{invalid}_artifact"] = "absent"
    data = DataProto.from_dict(
        tensors={"responses": (image + 1 if invalid == "projection" else image).unsqueeze(0)},
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "prompt"}],
            "tool_extra_fields": [fields],
        },
    )
    manager = _jpeg_manager()
    with pytest.raises(ValueError) as batch_error:
        await manager.run_single(data)
    client = create_async_reward_client(
        {"jpeg": (_worker(manager.run_single),)}, StreamingRewardConfig(extra_fields=extra_fields)
    )
    try:
        with pytest.raises(type(batch_error.value)) as streaming_error:
            await _score_sample(client, data)
        assert str(streaming_error.value) == str(batch_error.value)
    finally:
        await client.close()


async def _score_sample(client, data):
    async with client.sample_slot():
        return await client.compute_score(data)


def _config(models=None, **streaming):
    return OmegaConf.create(
        {
            "trainer": {"resume_mode": "disable", "use_v1": True, "v1": {"trainer_mode": "sync"}},
            "reward": {
                "streaming": {"_target_": "verl_omni.workers.config.reward.StreamingRewardConfig", **streaming},
                "models": models or {},
                "reward_model": {"enable": False, "enable_resource_pool": True},
            },
        }
    )


def test_named_streaming_requires_opt_in():
    config = _config({"quality": {"backend": "native", "offload": False}})
    assert not streaming_reward_enabled(config)
    config.reward.streaming.enabled = True
    assert streaming_reward_enabled(config)


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trainer.use_v1", False, "V1 synchronous"),
        ("trainer.v1.trainer_mode", "separate_async", "V1 synchronous"),
        ("trainer.v1.trainer_mode", "colocate_async", "V1 synchronous"),
        ("trainer.v1.trainer_mode", "omni_sync", "V1 synchronous"),
        ("algorithm.trainer_type", "direct_preference", "trainer_type=policy_gradient"),
        ("reward.aggregation", "preserve_components", "aggregation=weighted_sum"),
        ("reward.models.quality.backend", "engine", "only native"),
        ("reward.models.quality.placement.resource", "cpu", "accelerator placement"),
        ("reward.reward_model.enable_resource_pool", False, "enable_resource_pool=true"),
        ("reward.models.quality.offload", True, "offload=false"),
    ],
)
def test_entrypoint_validation_rejects_unsupported_streaming(path, value, message, as_dict):
    config = _config({"quality": {"backend": "native", "offload": False}}, enabled=True)
    OmegaConf.update(config, path, value)
    if as_dict:
        config = OmegaConf.to_container(config)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize("entrypoint", ["v1", "legacy", "omni"])
def test_launchers_reject_unsupported_streaming_before_ray(monkeypatch, entrypoint):
    from verl_omni.trainer import main_diffusion, main_diffusion_v1, main_omni

    config = _config({"quality": {"backend": "engine", "offload": False}}, enabled=True)
    monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", lambda: pytest.fail("Ray must not start"))
    if entrypoint == "legacy":
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="V1 synchronous"):
            main_diffusion.run_diffusion(config)
    else:
        launch = main_diffusion_v1.run_diffusion_v1 if entrypoint == "v1" else main_omni.run_omni
        with pytest.raises(ValueError, match="only native"):
            launch(config)


def test_omni_entrypoint_rejects_diffusion_streaming_mode_before_ray(monkeypatch):
    from verl_omni.trainer import main_omni

    config = _config({"quality": {"backend": "native", "offload": False}}, enabled=True)
    monkeypatch.setattr(main_omni.ray, "is_initialized", lambda: pytest.fail("Ray must not start"))
    with pytest.raises(ValueError, match="Omni entrypoint does not support named reward streaming"):
        main_omni.run_omni(config)


@pytest.mark.parametrize("missing", ["diffusion_base", "create", "raise_if_reward_failed"])
def test_incompatible_streaming_manager_rejected_before_ray(monkeypatch, missing):
    """Reject incompatible custom managers before any distributed initialization."""
    from verl.trainer.ppo.v1 import AgentLoopManagerTQ

    from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopManagerTQ
    from verl_omni.trainer import main_diffusion_v1

    class CustomManager(DiffusionAgentLoopManagerTQ):
        pass

    manager_cls = AgentLoopManagerTQ if missing == "diffusion_base" else CustomManager
    if missing != "diffusion_base":
        setattr(manager_cls, missing, None)
    config = _config({"quality": {"backend": "native", "offload": False}}, enabled=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.agent.agent_loop_manager_class", "custom.Manager")
    monkeypatch.setattr(main_diffusion_v1, "load_class_from_fqn", lambda *_: manager_cls)
    monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", lambda: pytest.fail("Ray must not start"))
    with pytest.raises(ValueError, match="DiffusionAgentLoopManagerTQ.*callable"):
        main_diffusion_v1.run_diffusion_v1(config)


@pytest.mark.parametrize(("enabled", "named"), [(True, True), (False, True), (False, False)])
def test_streaming_manager_check_preserves_supported_customization(monkeypatch, enabled, named):
    """Accept diffusion subclasses and leave non-streaming manager selection unrestricted."""
    from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopManagerTQ
    from verl_omni.trainer import main_diffusion_v1

    class CustomManager(DiffusionAgentLoopManagerTQ):
        pass

    config = _config({"quality": {"backend": "native", "offload": False}}, enabled=enabled)
    if not named:
        config.reward.models = {}
    config.transfer_queue = {"enable": True}
    OmegaConf.update(config, "actor_rollout_ref.rollout.agent.agent_loop_manager_class", "custom.Manager")
    monkeypatch.setattr(main_diffusion_v1, "load_class_from_fqn", lambda *_: CustomManager if enabled else object)
    monkeypatch.setattr(main_diffusion_v1, "enable_rl_insight", lambda _: None)

    def reached_ray():
        raise RuntimeError("manager validation passed")

    monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", reached_ray)
    with pytest.raises(RuntimeError, match="manager validation passed"):
        main_diffusion_v1.run_diffusion_v1(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "true"},
        {"max_inflight": 0},
        {"max_inflight": True},
        {"max_inflight": 1.5},
        {"timeout": 0},
        {"timeout": True},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
    ],
)
def test_streaming_rejects_invalid_external_settings(kwargs):
    with pytest.raises(ValueError):
        StreamingRewardConfig(**kwargs)


def test_named_streaming_requires_named_models():
    with pytest.raises(ValueError, match="named reward.models"):
        streaming_reward_enabled(_config(enabled=True))


def test_merge_preserves_weighted_values_and_diagnostics():
    outputs = [_output("quality", 1.5), _output("ocr", -0.5)]
    outputs[1]["reward_extra_info"]["reward/ocr/errors"] = 1
    result = merge_reward_outputs(outputs)
    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"] == {
        "reward/quality": 1.5,
        "reward/ocr": -0.5,
        "reward/ocr/errors": 1,
        "reward/combined": 1.0,
    }
    assert outputs[0]["reward_extra_info"]["reward/combined"] == 1.5
    with pytest.raises(ValueError, match="Duplicate"):
        merge_reward_outputs([outputs[0], outputs[0]])


@pytest.mark.asyncio
@pytest.mark.parametrize("include_extra_info", [False, True])
async def test_streaming_fans_out_once_and_keeps_sample_identity(include_extra_info):
    calls = {"a": [], "b": []}

    async def score(key, data):
        index = data.batch["responses"].item()
        assert data.non_tensor_batch["uid"].item() == f"sample-{index}"
        assert data.meta_info == {"validate": True}
        calls[key].append(index)
        await asyncio.sleep(0)
        output = _output(key, index + (1 if key == "a" else 2))
        return output if include_extra_info else {"reward_score": output["reward_score"]}

    client = create_async_reward_client(
        {key: tuple(_worker(lambda data, key=key: score(key, data)) for _ in range(3)) for key in calls},
        StreamingRewardConfig(max_inflight=2),
    )
    samples = [_sample(i) for i in range(5)]
    for i, data in enumerate(samples):
        data.non_tensor_batch["uid"] = np.array([f"sample-{i}"])
        data.meta_info["validate"] = True
    results = await asyncio.gather(*(_score_sample(client, data) for data in samples))
    assert [result["reward_score"] for result in results] == [3, 5, 7, 9, 11]
    assert [result["reward_extra_info"]["reward/combined"] for result in results] == [3, 5, 7, 9, 11]
    assert calls == {"a": list(range(5)), "b": list(range(5))}


@pytest.mark.asyncio
async def test_legacy_adapter_keeps_results_concurrency_and_cancellation():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    cancelled = []
    result = {"reward_score": 3.0}

    async def score(data):
        index = data.batch["responses"].item()
        calls.append(index)
        if len(calls) == 2:
            entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        return result

    handle = SimpleNamespace(compute_score=SimpleNamespace(remote=lambda data: asyncio.create_task(score(data))))
    client = create_async_reward_client([handle], StreamingRewardConfig(max_inflight=1, timeout=0.001))
    first, second = [asyncio.create_task(_score_sample(client, _sample(i))) for i in range(2)]
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            assert not client.requires_drain
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert cancelled == [0]
            release.set()
            assert await second is result
            assert calls == [0, 1]
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await client.close()


def test_no_handles_do_not_enable_async_scoring():
    assert create_async_reward_client(None, StreamingRewardConfig()) is None


@pytest.mark.asyncio
async def test_named_adapter_rejects_more_than_one_sample_before_submission():
    calls = []

    async def score(data):
        calls.append(data)
        return _output("a", 1)

    client = create_async_reward_client({"a": (_worker(score),)}, StreamingRewardConfig())
    with pytest.raises(ValueError, match="require one sample"):
        await _score_sample(client, DataProto.concat([_sample(), _sample()]))
    assert not calls


@pytest.mark.asyncio
async def test_backpressure_covers_accepted_work_and_cancelled_waiters():
    release = asyncio.Event()
    admitted = asyncio.Event()
    calls = []

    async def score(data):
        calls.append(data.batch["responses"].item())
        admitted.set()
        await release.wait()
        return _output("a", 1)

    client = create_async_reward_client({"a": (_worker(score),)}, StreamingRewardConfig(max_inflight=1))
    first = asyncio.create_task(_score_sample(client, _sample(1)))
    await admitted.wait()
    second = asyncio.create_task(_score_sample(client, _sample(2)))
    await asyncio.sleep(0)
    assert calls == [1]
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert calls == [1]
    assert (await _score_sample(client, _sample(3)))["reward_score"] == 1
    assert calls == [1, 3]


@pytest.mark.asyncio
async def test_deadline_discards_result_but_waits_for_accepted_work():
    finished = asyncio.Event()

    async def score(data):
        await asyncio.sleep(0.03)
        finished.set()
        return _output("a", 1)

    client = create_async_reward_client({"a": (_worker(score),)}, StreamingRewardConfig(timeout=0.005))
    with pytest.raises(TimeoutError):
        await _score_sample(client, _sample())
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("submission_failure", [False, True])
async def test_group_failure_and_submission_failure_drain_other_groups(submission_failure):
    release = asyncio.Event()
    entered = asyncio.Event()
    finished = asyncio.Event()

    async def slow(data):
        entered.set()
        await release.wait()
        finished.set()
        return _output("slow", 1)

    async def failure(data):
        raise RuntimeError("required scorer failed")

    failing_worker = _worker(failure)
    if submission_failure:

        def reject(data):
            raise RuntimeError("submission failed")

        failing_worker.compute_score_batch.remote = reject
    client = create_async_reward_client(
        {"slow": (_worker(slow),), "failed": (failing_worker,)},
        StreamingRewardConfig(),
    )
    task = asyncio.create_task(_score_sample(client, _sample()))
    await entered.wait()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(RuntimeError, match="failed"):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_close_fences_queued_samples_and_drains_before_return():
    release = asyncio.Event()
    entered = asyncio.Event()
    calls = []

    async def score(data):
        calls.append(data.batch["responses"].item())
        entered.set()
        await release.wait()
        return _output("a", 1)

    client = create_async_reward_client({"a": (_worker(score),)}, StreamingRewardConfig(max_inflight=1))
    first = asyncio.create_task(_score_sample(client, _sample(1)))
    await entered.wait()
    second = asyncio.create_task(_score_sample(client, _sample(2)))
    close = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await first
    await close
    with pytest.raises(RuntimeError, match="admission is closed"):
        await second
    await client.close()
    with pytest.raises(RuntimeError, match="admission is closed"):
        await _score_sample(client, _sample(3))
    assert calls == [1]


@pytest.mark.asyncio
async def test_close_during_generation_prevents_late_scoring():
    calls = []

    async def score(data):
        calls.append(data)
        return _output("a", 1)

    client = create_async_reward_client({"a": (_worker(score),)}, StreamingRewardConfig())
    async with client.sample_slot():
        await client.close()
        with pytest.raises(RuntimeError, match="admission is closed"):
            await client.compute_score(_sample())
    assert not calls
