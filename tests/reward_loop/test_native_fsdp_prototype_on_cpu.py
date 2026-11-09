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
"""CPU contract tests; actual sharding is checked by validate_native_fsdp.py."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import ray
import torch
from transformers import CLIPConfig, CLIPModel

from verl_omni.reward_loop.native_fsdp import (
    FSDPRewardReplica,
    NativeFSDPExecutor,
    NativeFSDPGroupError,
    PickScoreForward,
    build_replica,
    terminate_ranks,
)
from verl_omni.utils.reward_score.image_preference_utils import clip_feature_tensor
from verl_omni.utils.reward_score.pickscore_reward import _pairwise_pickscore


def test_forward_boundary_preserves_real_clip_features_and_scores():
    torch.manual_seed(42)
    config = CLIPConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 8,
            "vocab_size": 16,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
        projection_dim=8,
    )
    clip = CLIPModel(config).eval()
    image_inputs = {"pixel_values": torch.randn(3, 3, 8, 8)}
    text_inputs = {"input_ids": torch.tensor([[1, 4, 2], [1, 5, 2], [1, 6, 2]])}
    with torch.no_grad():
        expected = {
            "image_embeddings": clip_feature_tensor(clip.get_image_features(**image_inputs)),
            "text_embeddings": clip_feature_tensor(clip.get_text_features(**text_inputs)),
            "logit_scale": clip.logit_scale.exp(),
        }
        actual = PickScoreForward(clip)(image_inputs, text_inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(_pairwise_pickscore(**actual), _pairwise_pickscore(**expected), rtol=0, atol=0)


@pytest.fixture
def replica(monkeypatch):
    def processor(**kwargs):
        return {"input": torch.ones(1)}

    actors = [SimpleNamespace(name=name, status=SimpleNamespace(remote=lambda: "dead")) for name in ("rank0", "rank1")]
    group = SimpleNamespace(world_size=2, workers=actors)
    calls = []
    for method in ("wake_up", "sleep", "close"):
        setattr(group, method, lambda method=method: calls.append(method) or [None, None])
    group.infer = lambda *args: [{"score": torch.ones(1)}, None]

    def get(refs, **kwargs):
        if refs == "dead" if isinstance(refs, str) else False:
            raise ray.exceptions.ActorDiedError()
        return refs

    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", get)
    monkeypatch.setattr(
        "verl_omni.reward_loop.native_fsdp.ray.kill", lambda worker, **kwargs: calls.append(worker.name)
    )
    return FSDPRewardReplica(group, processor), calls


@pytest.mark.parametrize("method", ["sleep", "close"])
def test_lifecycle_waits_for_accepted_scoring_on_all_ranks(replica, method):
    model, calls = replica
    entered, release = threading.Event(), threading.Event()
    waiting = threading.Event()
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if lock.locked():
                waiting.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    model._lock = ObservedLock()

    def infer(*args):
        entered.set()
        assert release.wait(5)
        calls.append("infer_finished")
        return [{"score": torch.ones(1)}, None]

    model.worker_group.infer = infer
    model.wake_up()
    with ThreadPoolExecutor(2) as executor:
        forward = executor.submit(model.infer, ["prompt"], [object()])
        assert entered.wait(5)
        sleeping = executor.submit(getattr(model, method))
        assert waiting.wait(5)
        assert method not in calls
        release.set()
        forward.result(5)
        sleeping.result(5)
    assert calls[:3] == ["wake_up", "infer_finished", method]
    with pytest.raises(RuntimeError, match="not awake"):
        model.infer(["prompt"], [object()])
    if method == "sleep":
        model.wake_up()
        assert model.infer(["prompt"], [object()])["score"].item() == 1
    else:
        model.close()


@pytest.mark.parametrize("failure", [RuntimeError("rank failed"), TimeoutError("rank hung"), KeyboardInterrupt()])
def test_partial_submission_and_rank_failure_terminate_whole_group(replica, failure):
    model, calls = replica
    model.wake_up()

    def partial(*args):
        calls.append("rank0_accepted")
        raise failure

    model.worker_group.infer = partial
    expected = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else NativeFSDPGroupError
    with pytest.raises(expected):
        model.infer(["prompt"], [object()])
    assert calls[-3:] == ["rank0_accepted", "rank0", "rank1"]
    with pytest.raises(RuntimeError, match="closed"):
        model.wake_up()
    model.close()


def test_preprocessing_failure_does_not_enter_collectives(replica):
    model, calls = replica
    model.wake_up()
    with pytest.raises(ValueError, match="equally sized"):
        model.infer(["prompt"], [])
    assert calls == ["wake_up"]
    assert model.infer(["prompt"], [object()])["score"].item() == 1


def test_duplicate_rank_outputs_are_rejected(replica):
    model, calls = replica
    model.wake_up()
    model.worker_group.infer = lambda *args: [{"score": 1}, {"score": 1}]
    with pytest.raises(RuntimeError, match="one result"):
        model.infer(["prompt"], [object()])
    assert calls[-2:] == ["rank0", "rank1"]


def test_cleanup_attempts_every_rank_when_one_termination_fails(replica, monkeypatch):
    model, calls = replica

    def kill(worker, **kwargs):
        calls.append(worker.name)
        if worker.name == "rank0":
            raise RuntimeError("termination failed")

    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.kill", kill)
    with pytest.raises(BaseExceptionGroup, match="job termination is required"):
        model.close()
    assert calls == ["close", "rank0", "rank1"]
    with pytest.raises(BaseExceptionGroup, match="job termination is required"):
        model.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "120"])
def test_invalid_timeout_never_allocates_workers(timeout, monkeypatch):
    from verl_omni.reward_loop import native_fsdp

    def unexpected(*args, **kwargs):
        pytest.fail("allocation/preprocessing happened before timeout validation")

    monkeypatch.setattr(native_fsdp, "_RankWorkerGroup", unexpected)
    monkeypatch.setattr(native_fsdp.CLIPProcessor, "from_pretrained", unexpected)
    with pytest.raises(ValueError, match="finite positive"):
        build_replica(object(), "unused", "unused", timeout=timeout)


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2], [True, 1]])
def test_invalid_rank_placement_never_allocates(indices, monkeypatch):
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.is_cuda_available", True)
    with pytest.raises(ValueError, match="two distinct"):
        build_replica(
            SimpleNamespace(world_size=2, use_gpu=True, store=[2]), "unused", "unused", bundle_indices=indices
        )


def test_unconfirmed_rank_death_is_an_error_and_other_ranks_are_attempted(monkeypatch):
    calls = []
    actors = [SimpleNamespace(status=SimpleNamespace(remote=lambda: "pending")) for _ in range(2)]
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.kill", lambda actor, **kwargs: calls.append(actor))

    def timeout(*args, **kwargs):
        raise TimeoutError("rank did not exit")

    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", timeout)
    with pytest.raises(BaseExceptionGroup, match="job termination") as error:
        terminate_ranks(actors, 0.1)
    assert calls == actors and len(error.value.exceptions) == 2


def test_validation_cleanup_continues_and_writes_failed_report(tmp_path, monkeypatch):
    from tests.reward_loop import validate_native_fsdp as validation

    calls = []

    def fail_close():
        calls.append("close")
        raise RuntimeError("close failed")

    def remove(pg):
        calls.append(pg)
        if pg == "pg0":
            raise RuntimeError("remove failed")

    monkeypatch.setattr(validation, "remove_placement_group", remove)
    monkeypatch.setattr(validation.ray, "shutdown", lambda: calls.append("shutdown"))
    with pytest.raises(BaseExceptionGroup) as error:
        validation.finish_validation(
            SimpleNamespace(close=fail_close),
            SimpleNamespace(pgs=["pg0", "pg1"]),
            tmp_path,
            {"passed": True},
            ValueError("primary"),
        )
    assert calls == ["close", "pg0", "pg1", "shutdown"]
    report = json.loads((tmp_path / "result.json").read_text())
    assert not report["passed"] and len(report["errors"]) == len(error.value.exceptions) == 3


@pytest.mark.asyncio
async def test_cancelled_native_handle_retains_accepted_work_until_settlement():
    entered, release = threading.Event(), threading.Event()
    finished = threading.Event()

    def infer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {
            "image_embeddings": torch.ones(1, 2),
            "text_embeddings": torch.ones(1, 2),
            "logit_scale": torch.ones(()),
        }

    handle = NativeFSDPExecutor(SimpleNamespace(infer=infer))
    task = asyncio.create_task(handle.infer(["prompt"], [object()]))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert finished.is_set()
