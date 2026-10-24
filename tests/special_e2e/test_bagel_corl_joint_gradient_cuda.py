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

"""Manual two-rank FSDP2 math check; requires the Linux training dependencies.

Run: torchrun --standalone --nproc_per_node=2 tests/special_e2e/test_bagel_corl_joint_gradient_cuda.py
Checks production GEN partition/loss math, sequential backwards and one optimizer step
on a tiny shared actor. It does not substitute for a real BAGEL/vLLM end-to-end run.
"""

import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.fsdp import fully_shard
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss
from verl_omni.trainer.omni.bagel_corl_batch import prepare_gen_shard


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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This manual FSDP2 check requires CUDA; no CPU success fallback")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        raise ValueError("Run with exactly two ranks")
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    config = SimpleNamespace(diffusion_loss=SimpleNamespace(clip_ratio=0.02, adv_clip_max=5.0))
    try:
        for case in ("uneven", "empty_rank", "globally_empty"):
            torch.manual_seed(42)
            model = TinyJoint().to(device)
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(0.01)
            reference = copy.deepcopy(model)
            fully_shard(model)
            optimizer = CountingSGD(model.parameters())
            inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            # One episode on each rank: DP averaging turns this into the global mean.
            (-model(inputs[rank : rank + 1], "und").mean()).backward()
            parents = torch.tensor([10, 10, 10, 20] if case == "uneven" else [10, 10, 10, 10])
            valid = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
            if case == "globally_empty":
                valid.zero_()
            gen = TensorDict(
                {
                    "parent_episode_id": parents,
                    "gen_row_id": torch.arange(4),
                    "all_latents": torch.arange(32).float().reshape(4, 4, 2) / 32,
                    "all_timesteps": torch.full((4, 3), 0.8),
                    "gen_valid": torch.ones(4),
                    "gen_transition_valid": valid,
                    "old_log_probs": torch.zeros(4, 3),
                    "advantages": torch.tensor(
                        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]]
                    ),
                },
                batch_size=[4],
            )
            shard = prepare_gen_shard(
                gen,
                torch.tensor([10 if rank == 0 else 20]),
                dp_group=dist.group.WORLD,
                micro_size=1,
                collective_device=device,
            )
            if shard is not None:
                shard = shard.to(device)
                normalizer = tu.get_non_tensor_data(shard, "loss_normalizer", None)
                for row in range(len(shard)):
                    for step in range(3):
                        logp = model(shard["all_latents"][row : row + 1, step], "gen")
                        loss, _ = FlowGRPOLoss.compute_loss(
                            old_log_prob=shard["old_log_probs"][row : row + 1, step],
                            log_prob=logp,
                            advantages=shard["advantages"][row : row + 1, step],
                            config=config,
                            unit_weights=shard["gen_valid"][row : row + 1]
                            * shard["gen_transition_valid"][row : row + 1, step],
                            loss_normalizer=normalizer,
                        )
                        loss.backward()
            (-reference(inputs, "und").mean()).backward()
            if valid.sum() > 0:
                logp = reference(gen["all_latents"][:, :3].to(device).reshape(-1, 2), "gen").reshape(4, 3)
                loss, _ = FlowGRPOLoss.compute_loss(
                    old_log_prob=gen["old_log_probs"].to(device),
                    log_prob=logp,
                    advantages=gen["advantages"].to(device),
                    config=config,
                    unit_weights=valid.to(device),
                    loss_normalizer=float(valid.sum()),
                )
                loss.backward()
            reference_params = dict(reference.named_parameters())
            for name, p in model.named_parameters():
                expected = reference_params[name].grad
                if expected is None:
                    assert p.grad is None
                else:
                    actual = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            optimizer.step()
            assert optimizer.steps == 1
            dist.barrier()
            if rank == 0:
                print(f"PASS {case}: serial/distributed gradients agree, one optimizer step")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
