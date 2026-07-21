"""Tests for vLLM-Omni full-model weight updates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

pytestmark = pytest.mark.cpu


def test_full_weight_update_streams_buckets_and_finalizes_once(monkeypatch):
    loaded_buckets = []
    finalized = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            assert kwargs["device"] == torch.device("cpu")

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("layer.0.weight", torch.tensor([1.0]))])
            on_bucket_received([("layer.1.weight", torch.tensor([2.0]))])

    class FakeModel:
        def load_weights(self, weights):
            loaded_buckets.append([name for name, _tensor in weights])

    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as transfer_mod
    import vllm.model_executor.model_loader.utils as loader_utils

    monkeypatch.setattr(transfer_mod, "BucketedWeightReceiver", FakeReceiver)
    monkeypatch.setattr(
        loader_utils,
        "process_weights_after_loading",
        lambda model, config, device: finalized.append((model, config, device)),
    )

    worker = object.__new__(rollout_utils.vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-full-weight-update.sock"
    model = FakeModel()
    model_config = SimpleNamespace()
    worker._get_standard_weight_model_and_config = lambda: (model, model_config)

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=False)

    assert loaded_buckets == [["layer.0.weight"], ["layer.1.weight"]]
    assert finalized == [(model, model_config, torch.device("cpu"))]
