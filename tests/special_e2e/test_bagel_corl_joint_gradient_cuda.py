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

"""Manual two-rank production engine/FSDP2 check on a tiny shared actor.

Run: torchrun --standalone --nproc_per_node=2 tests/special_e2e/test_bagel_corl_joint_gradient_cuda.py
Requires Linux/CUDA plus the production training dependencies. Exercises the real
BaseEngine.train_batch, composite dispatch, GEN timestep/microbatch loop, loss
routing, UND episode PPO/KL/entropy reduction, and clipped optimizer step. Only
architecture forwards are replaced with TinyJoint adapters. This is not a real
BAGEL model or vLLM serving/parity/convergence test.
"""

import copy
import hashlib
import json
import os
from datetime import timedelta
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.fsdp import fully_shard
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.engine.base import BaseEngine

from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss
from verl_omni.workers.engine.fsdp import bagel_corl_composite as composite
from verl_omni.workers.engine.fsdp import diffusers_impl as engine_impl
from verl_omni.workers.utils.losses import bagel_composite_loss, bagel_und_episode_loss


def policy_parameter_digest(module) -> dict:
    """Hash every local actor parameter without gathering FSDP shards or changing state.

    Compare each rank against itself across calibration phases. This includes frozen
    backbone and adapter parameters, but excludes gradients, buffers and optimizer
    state; it is not a standalone receipt of what a serving engine loaded.
    """
    import torch
    from torch.distributed.tensor import DTensor

    checksum = hashlib.sha256()
    count = local_numel = trainable_local_numel = 0
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        if local.device.type == "meta" or local.layout != torch.strided:
            raise ValueError("Policy hashing requires materialized dense local parameters")
        value = local.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite policy parameter: {name}")
        layout = {
            "name": name,
            "dtype": str(value.dtype),
            "global_shape": list(parameter.shape),
            "local_shape": list(value.shape),
            "requires_grad": parameter.requires_grad,
            "placements": [str(item) for item in parameter.placements] if isinstance(parameter, DTensor) else [],
        }
        header = json.dumps(layout, sort_keys=True).encode()
        checksum.update(len(header).to_bytes(8, "big"))
        checksum.update(header)
        # uint8 supports bf16 without a dtype conversion; memoryview avoids copying
        # large parameter byte strings in addition to any required CPU transfer.
        raw = value.reshape(-1).view(torch.uint8).numpy()
        checksum.update(memoryview(raw))
        count += 1
        local_numel += value.numel()
        if parameter.requires_grad:
            trainable_local_numel += value.numel()
    if not count:
        raise ValueError("Policy hashing requires actor parameters")
    return {
        "parameters_sha256": checksum.hexdigest(),
        "parameter_count": count,
        "local_numel": local_numel,
        "trainable_local_numel": trainable_local_numel,
        "scope": "rank_local_parameters_only",
    }


class TinyJoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.und = torch.nn.Linear(2, 1, bias=False)
        self.gen = torch.nn.Linear(2, 1, bias=False)

    def forward(self, value, lane):
        text = self.und(value).squeeze(-1)
        return text if lane == "und" else self.gen(value).squeeze(-1) + 0.1 * text


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.001)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def und_loss(model, indices, config, device, dp_size):
    lengths = [2, 8, 3, 2]
    mask = torch.zeros(len(indices), 8, device=device)
    logits = []
    for row, episode in enumerate(indices):
        length = lengths[episode]
        # Each rank executes the same number of architecture forwards, including rejected rows.
        values = torch.arange((length + 1) * 2, device=device).float().reshape(-1, 2) / 32
        logits.append(model(values, "und"))
        if episode < 2:
            mask[row, :length] = 1
    data = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor(
                [torch.ones(1, device=device, dtype=torch.long) for _ in indices], layout=torch.jagged
            ),
            "responses": torch.nested.as_nested_tensor(
                [torch.ones(lengths[i], device=device, dtype=torch.long) for i in indices], layout=torch.jagged
            ),
            "response_mask": mask,
            "old_log_probs": torch.zeros_like(mask),
            "advantages": torch.ones_like(mask),
            "ref_log_prob": torch.full_like(mask, 0.1),
        },
        batch_size=[len(indices)],
    )
    tu.assign_non_tensor(
        data,
        max_response_len=8,
        dp_size=1,
        batch_num_tokens=None,
        global_batch_size=None,
        und_episode_normalizer=2 / dp_size,
        corl_dp_size=dp_size,
    )
    output = {
        "log_probs": torch.nested.as_nested_tensor(logits, layout=torch.jagged),
        "entropy": torch.nested.as_nested_tensor([x + 0.8 for x in logits], layout=torch.jagged),
    }
    loss, metrics = bagel_und_episode_loss(config, output, data)
    assert torch.isfinite(loss)
    return loss, metrics


def make_gen(case):
    parents = torch.tensor([10, 10, 10, 20] if case == "uneven" else [10, 10, 10, 10])
    valid = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    if case == "globally_empty":
        valid.zero_()
    return TensorDict(
        {
            "parent_episode_id": parents,
            "gen_row_id": torch.arange(4),
            "all_latents": torch.arange(32).float().reshape(4, 4, 2) / 32,
            "all_timesteps": torch.full((4, 3), 0.8),
            "gen_valid": torch.ones(4),
            "gen_transition_valid": valid,
            "old_log_probs": torch.zeros(4, 3),
            "advantages": torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]),
        },
        batch_size=[4],
    )


def make_engine(model, optimizer, reference, ref_optimizer, config, rank, device, micro_size, gen):
    engine = SimpleNamespace(
        model_config={"composite_mode": "bagel_corl"},
        module=model,
        scheduler=None,
        optimizer=optimizer,
        optimizer_config=SimpleNamespace(clip_grad=1e6),
        engine_config=SimpleNamespace(micro_batch_size_per_gpu=micro_size),
        ulysses_sequence_parallel_size=1,
        get_data_parallel_group=lambda: dist.group.WORLD,
        prepare_model_inputs=lambda **kwargs: ({}, {}),
        is_mp_src_rank_with_outputs=lambda: True,
    )
    for name in ("optimizer_zero_grad", "_run_forward_backward_batch", "postprocess_batch_func"):
        setattr(engine, name, MethodType(getattr(engine_impl.DiffusersFSDPEngine, name), engine))
    engine._reslice_per_row_non_tensors = engine_impl.DiffusersFSDPEngine._reslice_per_row_non_tensors
    for name in ("forward_backward_batch", "forward_step", "prepare_model_outputs"):
        setattr(engine, name, MethodType(getattr(engine_impl.PPODiffusersFSDPEngine, name), engine))

    def check_and_step():
        # Called by real BaseEngine.train_batch only after both lane backwards.
        ref_optimizer.zero_grad()
        serial_und, _ = und_loss(reference, [0, 2, 1, 3], config, device, 1)
        serial_und.backward()
        valid = gen["gen_transition_valid"].to(device)
        if valid.sum() > 0:
            logp = reference(gen["all_latents"][:, :3].to(device).reshape(-1, 2), "gen").reshape(4, 3)
            serial_gen, _ = FlowGRPOLoss.compute_loss(
                old_log_prob=gen["old_log_probs"].to(device),
                log_prob=logp,
                advantages=gen["advantages"].to(device),
                config=config,
                unit_weights=valid,
                loss_normalizer=float(valid.sum()),
            )
            serial_gen.backward()
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected = expected_params[name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                actual = parameter.grad.full_tensor() if hasattr(parameter.grad, "full_tensor") else parameter.grad
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        norm = engine_impl.DiffusersFSDPEngine.optimizer_step(engine)
        ref_optimizer.step()
        for name, parameter in model.named_parameters():
            actual = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
            torch.testing.assert_close(actual, expected_params[name], atol=1e-6, rtol=1e-5)
        return norm

    engine.optimizer_step = check_and_step

    def und_adapter(_engine, _data, _loss_function, forward_only=False):
        assert not forward_only
        loss, _ = und_loss(model, [rank, rank + 2], config, device, 2)
        loss.backward()
        assert optimizer.steps == ref_optimizer.steps  # No early step between the lanes.
        return {"model_output": {}, "loss": [float(loss.detach())], "metrics": {}}

    def gen_adapter(**kwargs):
        batch, step = kwargs["scheduler_inputs"], kwargs["step"]
        logp = model(batch["all_latents"][:, step], "gen")
        return logp, torch.zeros_like(logp), torch.ones_like(logp), torch.ones_like(logp)

    return engine, und_adapter, gen_adapter


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This manual FSDP2 check requires CUDA; no CPU success fallback")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    if dist.get_world_size() != 2:
        raise ValueError("Run with exactly two ranks")
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        entropy_coeff=0.05,
        use_kl_loss=True,
        kl_loss_coef=0.01,
        kl_loss_type="kl",
    )
    config.diffusion_loss = OmegaConf.create({"loss_mode": "flow_grpo", "clip_ratio": 0.02, "adv_clip_max": 5.0})
    try:
        for case in ("uneven", "empty_rank", "globally_empty"):
            for micro_size in (1, 2):
                torch.manual_seed(42)
                model = TinyJoint().to(device)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.mul_(0.01)
                reference = copy.deepcopy(model)
                fully_shard(model)
                weight_digest = policy_parameter_digest(model)
                assert weight_digest == policy_parameter_digest(model)
                optimizer = CountingSGD(model.parameters())
                ref_optimizer = CountingSGD(reference.parameters())
                gen = make_gen(case)
                engine, und_adapter, gen_adapter = make_engine(
                    model, optimizer, reference, ref_optimizer, config, rank, device, micro_size, gen
                )
                data = TensorDict(
                    {
                        "input_ids": torch.ones(1, 2, device=device),
                        "bagel_episode_id": torch.tensor([10 if rank == 0 else 20], device=device),
                        "old_log_probs": torch.zeros(1, 1, device=device),
                        "advantages": torch.ones(1, 1, device=device),
                    },
                    batch_size=[1],
                )
                tu.assign_non_tensor(data, bagel_corl_gen=gen, has_complete_gen_groups=True, skip_gen=False)
                with (
                    patch.object(composite, "run_und_token_forward_backward", und_adapter),
                    patch.object(engine_impl, "forward_and_sample_previous_step", gen_adapter),
                ):
                    for minibatch in range(2):
                        BaseEngine.train_batch(engine, data, partial(bagel_composite_loss, config))
                        assert optimizer.steps == minibatch + 1
                        assert ref_optimizer.steps == minibatch + 1
                        updated_digest = policy_parameter_digest(model)
                        # Tiny output-size-one layers leave an empty FSDP shard on
                        # one rank. The rank-tagged aggregate policy must change;
                        # an empty local shard correctly retains its digest.
                        changed = torch.tensor(
                            int(weight_digest["parameters_sha256"] != updated_digest["parameters_sha256"]),
                            device=device,
                        )
                        dist.all_reduce(changed, op=dist.ReduceOp.SUM)
                        assert changed.item() > 0
                        weight_digest = updated_digest
                        dist.barrier()
                if rank == 0:
                    print(f"PASS {case}, micro={micro_size}: real engine gradients/parameters, two shared steps")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
