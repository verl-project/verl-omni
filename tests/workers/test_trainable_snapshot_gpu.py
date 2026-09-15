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
"""Two-rank snapshot mechanism test, not pretrained precision or throughput evidence.

Run: torchrun --standalone --nproc-per-node=2 tests/workers/test_trainable_snapshot_gpu.py --output /tmp/snapshots
"""

import argparse
import gc
import json
import os
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig

from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig, DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig, DiffusionRolloutAlgoConfig
from verl_omni.workers.detach_actor_worker import DiffusionDetachActorWorker, _TrainableSnapshot
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss

SEED = 20260911


class _FullSnapshotWorker(DiffusionDetachActorWorker):
    def _supports_trainable_snapshot(self, module):
        return False


def _worker(engine, trainable_only):
    cls = DiffusionDetachActorWorker if trainable_only else _FullSnapshotWorker
    worker = object.__new__(cls)
    worker.actor = SimpleNamespace(engine=engine)
    worker.config = OmegaConf.create({"actor": {"strategy": "fsdp2"}})
    worker.cpu_saved_models = {}
    worker._strategy_handlers = None
    return worker


def _state(engine, gradients=False):
    result = {}
    for name, param in engine.module.named_parameters():
        tensor = param.grad if gradients else param
        if tensor is not None:
            if isinstance(tensor, DTensor):
                tensor = tensor.to_local()
            result[name] = tensor.detach().cpu().clone()
    return result


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def _bytes(state):
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(_bytes(value) for value in state.values())
    if isinstance(state, tuple | list):
        return sum(_bytes(value) for value in state)
    return 0


def _fixture(root):
    torch.manual_seed(SEED)
    model = QwenImageTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=2,
        in_channels=64,
        out_channels=16,
        patch_size=2,
        joint_attention_dim=32,
        axes_dims_rope=(8, 12, 12),
        guidance_embeds=False,
    )
    model.save_pretrained(root / "transformer")
    FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True).save_pretrained(root / "scheduler")
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}))


def _engine(model_path, offload):
    torch.manual_seed(SEED)
    model = DiffusionModelConfig(
        path=str(model_path),
        algorithm="flow_grpo",
        load_tokenizer=False,
        attn_backend="native",
        enable_gradient_checkpointing=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dtype="float32",
        pipeline=DiffusionPipelineConfig(height=64, width=64, num_inference_steps=10, true_cfg_scale=1.0),
        algo=DiffusionRolloutAlgoConfig(noise_level=0.8, sde_type="sde"),
    )
    config = FSDPEngineConfig(
        strategy="fsdp2",
        fsdp_size=dist.get_world_size(),
        ulysses_sequence_parallel_size=1,
        model_dtype="bfloat16",
        dtype="bfloat16",
        use_orig_params=True,
        mixed_precision={"param_dtype": "bfloat16", "reduce_dtype": "float32", "buffer_dtype": "float32"},
        param_offload=offload,
        optimizer_offload=False,
        forward_only=False,
    )
    engine = PPODiffusersFSDPEngine(
        model, config, FSDPOptimizerConfig(lr=1e-4, clip_grad=1.0, total_training_steps=10), CheckpointConfig()
    )
    engine.initialize()
    return engine


def _batch(engine):
    generator = torch.Generator().manual_seed(SEED + dist.get_rank())
    data = TensorDict(
        {
            "prompt_embeds": torch.randn(2, 8, 32, generator=generator, dtype=torch.bfloat16),
            "prompt_embeds_mask": torch.ones(2, 8, dtype=torch.int32),
            "all_latents": torch.randn(2, 3, 16, 64, generator=generator),
            "all_timesteps": engine.scheduler.timesteps[:2].cpu().expand(2, -1).clone(),
            "advantages": torch.tensor([[0.25, 0.25], [-0.25, -0.25]]),
            "old_log_probs": torch.zeros(2, 2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=1, height=64, width=64, vae_scale_factor=8)
    return data


def _run_case(model_path, offload):
    engine = _engine(model_path, offload)
    baseline = _worker(engine, False)
    baseline.save_model_to_cpu(900)
    initial = _state(engine)
    data = _batch(engine)
    actor = DiffusionActorConfig(
        strategy="fsdp2",
        ppo_micro_batch_size_per_gpu=1,
        rollout_n=1,
        diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2),
    )
    loss_fn = partial(diffusion_loss, config=actor)
    histories = {}
    report = {"offload": offload, "rank": dist.get_rank(), "arms": {}}

    def compute_old(_trainer, batch):
        with engine.eval_mode():
            return engine.infer_batch(batch)["model_output"]["log_probs"].detach().cpu()

    for trainable_only in (False, True):
        baseline.restore_model_from_cpu(900)
        engine.optimizer_zero_grad()
        engine.optimizer.state.clear()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        worker = _worker(engine, trainable_only)
        trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
        trainer.parameter_sync_step = 3
        trainer.actor_rollout_wg = worker
        history = []
        with patch.object(PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", compute_old):
            for local_step in range(3):
                trainer.local_trigger_step = local_step
                before = _state(engine)
                old = trainer._compute_old_log_prob(data.clone())
                _assert_equal(_state(engine), before)
                if local_step == 0:
                    saved = worker.cpu_saved_models[0]
                    assert isinstance(saved, _TrainableSnapshot) == trainable_only
                    report["arms"][str(trainable_only)] = {
                        "payload_bytes": _bytes(saved.state if trainable_only else saved)
                    }
                else:
                    torch.testing.assert_close(old, history[0]["old"], rtol=0, atol=0)
                batch = data.clone()
                batch["old_log_probs"] = old
                with engine.train_mode():
                    output = engine.train_batch(batch, loss_fn)
                    gradients = _state(engine, gradients=True)
                    assert gradients
                history.append({"old": old, "params": _state(engine), "grads": gradients, "loss": output["loss"]})
        assert worker.cpu_saved_models == {}
        histories[trainable_only] = history
        assert any(not torch.equal(history[-1]["params"][name], initial[name]) for name in initial)

        # Isolate snapshot cost; these tiny-fixture timings are diagnostic only.
        seconds = []
        for _ in range(3):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            worker.save_model_to_cpu(10)
            worker.restore_model_from_cpu(10)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            worker.clear_cpu_model(10)
        report["arms"][str(trainable_only)]["save_restore_seconds"] = seconds

    for expected, actual in zip(histories[False], histories[True], strict=True):
        torch.testing.assert_close(actual["old"], expected["old"], rtol=0, atol=0)
        _assert_equal(actual["params"], expected["params"])
        _assert_equal(actual["grads"], expected["grads"])
        torch.testing.assert_close(torch.as_tensor(actual["loss"]), torch.as_tensor(expected["loss"]), rtol=0, atol=0)
    assert report["arms"]["True"]["payload_bytes"] < report["arms"]["False"]["payload_bytes"]

    worker = _worker(engine, True)
    worker.save_model_to_cpu(0)
    before = _state(engine)
    changed = next(param for param in engine.module.parameters() if param.requires_grad)
    if dist.get_rank() == 1:
        changed.requires_grad_(False)
    try:
        worker.restore_model_from_cpu(0)
    except RuntimeError as error:
        assert "parameter mapping changed" in str(error)
    else:
        raise AssertionError("rank-local mapping drift was not rejected on every rank")
    finally:
        changed.requires_grad_(True)
    _assert_equal(_state(engine), before)
    worker.clear_cpu_model(0)

    # A locally ineligible rank must make every rank select the same full-snapshot representation.
    with patch.object(engine, "_uses_fsdp2_cpu_offload_policy", dist.get_rank() == 1):
        worker.save_model_to_cpu(1)
    assert not isinstance(worker.cpu_saved_models[1], _TrainableSnapshot)
    worker.restore_model_from_cpu(1)
    _assert_equal(_state(engine), before)
    worker.clear_cpu_model(1)
    baseline.clear_cpu_model(900)
    report["verdict"] = "passed"
    del worker, trainer, baseline, histories
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    """Exercise real FSDP2 shards with a tiny random Qwen-Image fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    if dist.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        _fixture(args.output / "model")
    dist.barrier()
    results = [_run_case(args.output / "model", offload) for offload in (False, True)]
    path = args.output / f"result-rank{dist.get_rank()}.json"
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps({"scope": "tiny random mechanism only", "results": results}, indent=2))
    pending.replace(path)
    print(f"SNAPSHOT_PASS rank={dist.get_rank()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
