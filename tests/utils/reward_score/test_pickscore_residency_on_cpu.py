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
"""CPU-only fakes for the PickScore CUDA residency contract."""

import asyncio
import importlib.util
import threading
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/pickscore_reward.py"
    spec = importlib.util.spec_from_file_location("pickscore_residency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value, error", [(1, TypeError), ("true", TypeError), (True, ValueError)])
def test_rejects_invalid_opt_in_before_loading(monkeypatch, value, error):
    module = _load_module()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: pytest.fail("loaded model"))
    with pytest.raises(error):
        module.PickScoreNativeModel(device="cpu", retain_weights_on_cpu=value)


@pytest.mark.asyncio
async def test_opt_in_moves_same_model_between_cpu_and_explicit_cuda(monkeypatch):
    module = _load_module()

    class Weights:
        def __init__(self):
            self.targets = []
            self.dtype = torch.float16

        def to(self, target):
            self.targets.append(torch.device(target))
            return self

    weights = Weights()
    inferencer = type("Inferencer", (), {"model": weights, "processor": object(), "device": torch.device("cuda:2")})()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: inferencer)
    model = module.PickScoreNativeModel(device="cuda:2", retain_weights_on_cpu=True)
    assert model.supports_cpu_offload
    original_processor = inferencer.processor
    await model.sleep()
    assert weights.targets == [torch.device("cpu")]
    assert inferencer.device == torch.device("cpu")
    with pytest.raises(RuntimeError, match="asleep"):
        await model.infer([], [])
    await model.wake_up()
    assert weights.targets[-1] == torch.device("cuda:2")
    assert model._inferencer is inferencer and inferencer.processor is original_processor
    assert weights.dtype == torch.float16
    await model.close()
    with pytest.raises(RuntimeError, match="closed"):
        await model.infer([], [])


@pytest.mark.asyncio
async def test_sleep_drains_consumer_thread_before_cpu_transfer(monkeypatch):
    module = _load_module()
    entered = threading.Event()
    release = threading.Event()

    class Weights:
        def __init__(self):
            self.moves = []

        def to(self, target):
            assert release.is_set()
            self.moves.append(torch.device(target))
            return self

    class Inferencer:
        def __init__(self):
            self.model = Weights()
            self.device = torch.device("cuda:1")
            self.processor = object()

        def infer(self, prompts, images):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release consumer")
            values = torch.ones((len(prompts), 2))
            return {"text_embeddings": values, "image_embeddings": values, "logit_scale": torch.tensor(1)}

    inferencer = Inferencer()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: inferencer)
    monkeypatch.setattr(module.torch.cuda, "device", lambda *_args: nullcontext())
    model = module.PickScoreNativeModel(device="cuda:1", retain_weights_on_cpu=True)
    request = asyncio.create_task(model.infer(["prompt"], [object()]))
    assert await asyncio.to_thread(entered.wait, 2)
    sleep = asyncio.create_task(model.sleep())
    await asyncio.sleep(0)
    assert not sleep.done() and not inferencer.model.moves
    release.set()
    assert len(await request) == 1
    await sleep
    assert inferencer.model.moves == [torch.device("cpu")]
    await model.close()


@pytest.mark.asyncio
async def test_cancelled_sleep_finishes_drain_and_serializes_wake(monkeypatch):
    module = _load_module()
    entered = threading.Event()
    release = threading.Event()

    class Weights:
        def __init__(self):
            self.moves = []

        def to(self, target):
            self.moves.append(torch.device(target))
            return self

    class Inferencer:
        def __init__(self):
            self.model = Weights()
            self.device = torch.device("cuda:1")
            self.processor = object()

        def infer(self, prompts, images):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release consumer")
            values = torch.ones((len(prompts), 2))
            return {"text_embeddings": values, "image_embeddings": values, "logit_scale": torch.tensor(1)}

    inferencer = Inferencer()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: inferencer)
    monkeypatch.setattr(module.torch.cuda, "device", lambda *_args: nullcontext())
    model = module.PickScoreNativeModel(device="cuda:1", retain_weights_on_cpu=True)
    request = asyncio.create_task(model.infer(["prompt"], [object()]))
    assert await asyncio.to_thread(entered.wait, 2)
    sleep = asyncio.create_task(model.sleep())
    await asyncio.sleep(0)
    sleep.cancel()
    sleep.cancel()
    wake = asyncio.create_task(model.wake_up())
    await asyncio.sleep(0)
    assert not sleep.done() and not wake.done()
    release.set()
    await request
    with pytest.raises(asyncio.CancelledError):
        await sleep
    await wake
    assert inferencer.model.moves == [torch.device("cpu"), torch.device("cuda:1")]
    assert not model._asleep
    await model.close()


@pytest.mark.asyncio
async def test_failed_transfer_closes_model_permanently(monkeypatch):
    module = _load_module()

    class Weights:
        def to(self, target):
            raise OSError("transfer failed")

    inferencer = type("Inferencer", (), {"model": Weights(), "processor": object(), "device": torch.device("cuda:0")})()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: inferencer)
    model = module.PickScoreNativeModel(device="cuda:0", retain_weights_on_cpu=True)
    with pytest.raises(OSError, match="transfer failed"):
        await model.sleep()
    assert model._closed and not hasattr(model, "_inferencer")
    with pytest.raises(RuntimeError, match="closed"):
        await model.wake_up()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
async def test_cancelled_device_transfer_settles_before_close(monkeypatch, operation):
    module = _load_module()
    entered = threading.Event()
    release = threading.Event()
    target_type = "cpu" if operation == "sleep" else "cuda"

    class Weights:
        def __init__(self):
            self.moves = []

        def to(self, target):
            target = torch.device(target)
            if target.type == target_type:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release transfer")
            self.moves.append(target)
            return self

    weights = Weights()
    inferencer = type("Inferencer", (), {"model": weights, "device": torch.device("cuda:2")})()
    monkeypatch.setattr(module, "_PickScoreInferencer", lambda **kwargs: inferencer)
    model = module.PickScoreNativeModel(device="cuda:2", retain_weights_on_cpu=True)
    if operation == "wake_up":
        await model.sleep()
    transition = asyncio.create_task(getattr(model, operation)())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        transition.cancel()
        await asyncio.sleep(0)
        transition.cancel()
        close = asyncio.create_task(model.close())
        await asyncio.sleep(0)
        assert not transition.done() and not close.done()
        assert model._asleep
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await transition
    await close
    assert model._closed and not hasattr(model, "_inferencer")
    assert weights.moves[-1].type == target_type
