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

import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto

from verl_omni.reward_loop.streaming import StreamingRewardClient, merge_reward_outputs
from verl_omni.utils.config import validate_config
from verl_omni.workers.config.reward import StreamingRewardConfig, streaming_reward_enabled


def _sample(index=0):
    return DataProto.from_dict(tensors={"responses": torch.tensor([[index]])})


def _output(key, score):
    return {"reward_score": score, "reward_extra_info": {f"reward/{key}": score, "reward/combined": score}}


def _worker(fn):
    return SimpleNamespace(compute_score=SimpleNamespace(remote=lambda data: asyncio.create_task(fn(data))))


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
        ("reward.models.quality.backend", "engine", "only native"),
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
        calls[key].append(index)
        await asyncio.sleep(0)
        output = _output(key, index + (1 if key == "a" else 2))
        return output if include_extra_info else {"reward_score": output["reward_score"]}

    client = StreamingRewardClient(
        {key: (_worker(lambda data, key=key: score(key, data)),) for key in calls},
        StreamingRewardConfig(max_inflight=2),
    )
    results = await asyncio.gather(*(client.compute_score(_sample(i)) for i in range(5)))
    assert [result["reward_score"] for result in results] == [3, 5, 7, 9, 11]
    assert [result["reward_extra_info"]["reward/combined"] for result in results] == [3, 5, 7, 9, 11]
    assert calls == {"a": list(range(5)), "b": list(range(5))}


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

    client = StreamingRewardClient({"a": (_worker(score),)}, StreamingRewardConfig(max_inflight=1))
    first = asyncio.create_task(client.compute_score(_sample(1)))
    await admitted.wait()
    second = asyncio.create_task(client.compute_score(_sample(2)))
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
    assert (await client.compute_score(_sample(3)))["reward_score"] == 1
    assert calls == [1, 3]


@pytest.mark.asyncio
async def test_deadline_discards_result_but_waits_for_accepted_work():
    finished = asyncio.Event()

    async def score(data):
        await asyncio.sleep(0.03)
        finished.set()
        return _output("a", 1)

    client = StreamingRewardClient({"a": (_worker(score),)}, StreamingRewardConfig(timeout=0.005))
    with pytest.raises(TimeoutError):
        await client.compute_score(_sample())
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

        failing_worker.compute_score.remote = reject
    client = StreamingRewardClient(
        {"slow": (_worker(slow),), "failed": (failing_worker,)},
        StreamingRewardConfig(),
    )
    task = asyncio.create_task(client.compute_score(_sample()))
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

    client = StreamingRewardClient({"a": (_worker(score),)}, StreamingRewardConfig(max_inflight=1))
    first = asyncio.create_task(client.compute_score(_sample(1)))
    await entered.wait()
    second = asyncio.create_task(client.compute_score(_sample(2)))
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
        await client.compute_score(_sample(3))
    assert calls == [1]


@pytest.mark.asyncio
async def test_close_during_generation_prevents_late_scoring():
    calls = []

    async def score(data):
        calls.append(data)
        return _output("a", 1)

    client = StreamingRewardClient({"a": (_worker(score),)}, StreamingRewardConfig())
    async with client.sample_slot():
        await client.close()
        with pytest.raises(RuntimeError, match="admission is closed"):
            await client.compute_admitted_score(_sample())
    assert not calls
