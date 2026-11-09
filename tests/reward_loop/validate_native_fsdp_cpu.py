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

"""Manual CPU/Gloo acceptance of real sharding, named dispatch and actor cleanup.

Use --tiny for deterministic contract checks, or supply a pretrained checkpoint.
Outputs belong in a local validation directory, not in the repository.
"""

import argparse
import asyncio
import gc
import json
import os
import signal
import tempfile
from pathlib import Path

import numpy as np
import ray
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel
from verl import DataProto

from tests.reward_loop.test_cpu_native_reward_ray_on_cpu import _full_config
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.utils.reward_score.clipscore_reward import CLIPScoreInferencer, pairwise_clipscore
from verl_omni.utils.reward_score.pickscore_reward import _pairwise_pickscore, _PickScoreInferencer

_ADAPTERS = {
    "clipscore": (
        "verl_omni.utils.reward_score.clipscore_reward",
        "CLIPScoreNativeModel",
        CLIPScoreInferencer,
        pairwise_clipscore,
    ),
    "pickscore": (
        "verl_omni.utils.reward_score.pickscore_reward",
        "PickScoreNativeModel",
        _PickScoreInferencer,
        _pairwise_pickscore,
    ),
}


def make_tiny_checkpoint(path):
    """Save actual CLIP weights with a small architecture and the standard tokenizer vocabulary."""
    torch.manual_seed(432)
    config = CLIPConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4},
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 16,
        },
        projection_dim=16,
    )
    CLIPModel(config).save_pretrained(path)
    return str(path)


def make_inputs():
    """Use an odd batch with repeated and truncated prompts to exercise replica dispatch."""
    prompts = ["a cat", "a dog", "a cat", "a tree", "a landscape" + " with details" * 100]
    images = [
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        for index in range(len(prompts))
    ]
    data = DataProto.from_dict(
        tensors={
            "responses": torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) for image in images])
        },
        non_tensors={
            "data_source": ["cpu_fsdp_acceptance"] * len(prompts),
            "reward_model": [{"ground_truth": prompt} for prompt in prompts],
            "media_kind": ["image"] * len(prompts),
        },
    )
    return prompts, images, data


def assert_dead(actors):
    """Confirm actual actor death before shutting down the test's Ray cluster."""
    for actor in actors:
        try:
            ray.get(actor.__ray_ready__.remote(), timeout=10)
        except ray.exceptions.RayActorError:
            continue
        raise AssertionError("An owned actor survived deployment close")


def main():
    """Run the production CPU FSDP path without reserving accelerator resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", choices=sorted(_ADAPTERS), default="clipscore")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--fault", choices=["none", "rank_exit", "facade_exit", "facade_hang"], default="none")
    args = parser.parse_args()
    if args.replicas < 1 or args.cycles < 1:
        parser.error("replicas and cycles must be positive")
    if torch.cuda.is_available():
        parser.error("Hide accelerators with CUDA_VISIBLE_DEVICES='' for CPU acceptance")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model_path = make_tiny_checkpoint(args.output / "tiny-clip") if args.tiny else args.model_path
    module, model_class, inferencer_class, score_fn = _ADAPTERS[args.adapter]
    prompts, images, data = make_inputs()
    reference = inferencer_class(model_path=model_path, processor_path=args.processor_path, device="cpu")
    expected = score_fn(**reference.infer(prompts, images))
    global_numel = sum(parameter.numel() for parameter in reference.model.parameters())
    del reference
    gc.collect()

    cfg = _full_config(args.output / "tokenizer", not args.resident)
    cfg.trainer.use_v1 = True
    cfg.trainer.v1.trainer_mode = "sync"
    cfg.reward.reward_model.enable_resource_pool = False
    cfg.reward.reward_model.nnodes = 1
    cfg.reward.models = {
        args.adapter: {
            "backend": "native",
            "model_path": model_path,
            "offload": not args.resident,
            "placement": {
                "resource": "cpu",
                "replica_groups": [[2 * index, 2 * index + 1] for index in range(args.replicas)],
                "cpus_per_worker": 1,
            },
            "executor": {"model": f"{module}:{model_class}", "kwargs": {"processor_path": args.processor_path}},
            "fsdp": {"timeout": 120, "collective_timeout": 30},
        }
    }
    cfg.reward.reward_functions = {
        args.adapter: {"path": f"pkg://{module}", "name": f"compute_score_{args.adapter}_native", "required": True}
    }
    report = {
        "passed": False,
        "backend": "cpu:gloo",
        "adapter": args.adapter,
        "replica_groups": args.replicas,
        "cycles": args.cycles,
        "resident": args.resident,
        "model_path": model_path,
        "global_numel": global_numel,
        "fault": args.fault,
    }
    ray.init(
        address="local",
        num_cpus=2 * args.replicas + 4,
        num_gpus=0,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="fsdp-cpu-"),
        object_store_memory=256 * 1024**2,
    )
    manager = None
    primary = None
    try:
        assert ray.cluster_resources().get("GPU", 0) == 0
        manager = OmniRewardLoopManager(cfg)
        owner = manager.multi_reward_model_manager.models[args.adapter]
        asyncio.run(owner.wake_up())
        shards = ray.get([worker.status.remote() for worker in owner.rank_workers], timeout=15)
        assert len(shards) == args.replicas * 2
        for index in range(args.replicas):
            group = shards[index * 2 : index * 2 + 2]
            assert all(state["device_type"] == "cpu" and state["all_parameters_sharded"] for state in group)
            assert sum(state["local_numel"] for state in group) == global_numel
        report["shards"] = shards
        errors = []
        for _ in range(args.cycles):
            for validate in [False, True]:
                data.meta_info["validate"] = validate
                result = manager.compute_rm_score(data)
                actual = result.batch["rm_scores"].flatten()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                errors.append(float((actual - expected).abs().max()))
            states = ray.get([worker.status.remote() for worker in owner.rank_workers], timeout=15)
            assert all(state["awake"] == args.resident for state in states)
        report["max_score_abs_error"] = max(errors)
        if args.fault != "none":
            owner.timeout = 3
            if args.fault == "rank_exit":
                ray.kill(owner.rank_workers[1], no_restart=True)
            elif args.fault == "facade_exit":
                ray.kill(owner._workers[0], no_restart=True)
            else:
                actor = ray._private.state.actors()[owner._workers[0]._actor_id.hex()]
                os.kill(actor["Pid"], signal.SIGSTOP)
            try:
                manager.compute_rm_score(data)
            except Exception as error:
                report["expected_fault_error"] = type(error).__name__
            else:
                raise AssertionError("Injected deployment failure returned successful scores")
            assert owner._closed
        report["passed"] = True
    except BaseException as error:
        primary = error
    finally:
        try:
            if manager is not None:
                owner = manager.multi_reward_model_manager.models[args.adapter]
                actors = [*owner.rank_workers, *(owner._workers or [])]
                asyncio.run(manager.multi_reward_model_manager.close_native_models())
                assert_dead(actors)
                report["all_owned_actors_dead_before_ray_shutdown"] = True
        except BaseException as cleanup:
            primary = (
                cleanup if primary is None else BaseExceptionGroup("Acceptance and cleanup failed", [primary, cleanup])
            )
        finally:
            report["passed"] = report["passed"] and primary is None
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            ray.shutdown()
    if primary is not None:
        raise primary
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
