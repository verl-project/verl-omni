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
"""Real two-GPU PickScore FSDP acceptance, including group failure and cleanup.

Run as python -m tests.reward_loop.validate_native_fsdp with local model and
processor paths. This checks a reward pipeline, not a full RL training cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from ray.util.placement_group import remove_placement_group
from transformers import CLIPConfig, CLIPModel
from verl import DataProto
from verl.single_controller.ray import RayResourcePool

from verl_omni.reward_loop.native_fsdp import NativeFSDPExecutor as NativeHandle
from verl_omni.reward_loop.native_fsdp import build_replica
from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.utils.reward_score.jpeg_compressibility import compute_score as jpeg_score
from verl_omni.utils.reward_score.pickscore_reward import (
    PickScoreNativeModel,
    _pairwise_pickscore,
    compute_score_pickscore_native,
)


def _images():
    arrays = []
    for index in range(3):
        rng = np.random.default_rng(432 + index)
        arrays.append(rng.integers(0, 256, (224 + index * 8, 224, 3), dtype=np.uint8))
    return [Image.fromarray(array) for array in arrays]


def _scores(outputs):
    return torch.stack([_pairwise_pickscore(**output) for output in outputs]).cpu()


def finish_validation(replica, pool, output, report, primary_error=None):
    """Attempt every owned cleanup and report write, preserving every failure."""
    errors = [] if primary_error is None else [primary_error]
    if replica is not None:
        try:
            replica.close()
        except BaseException as exc:
            errors.append(exc)
    # The getter allocates placement groups when none exist; teardown must not.
    for pg in pool.pgs or []:
        try:
            remove_placement_group(pg)
        except BaseException as exc:
            errors.append(exc)
    try:
        ray.shutdown()
    except BaseException as exc:
        errors.append(exc)
    report["passed"] = report.get("passed", False) and not errors
    report["errors"] = [{"type": type(exc).__name__, "message": str(exc)} for exc in errors]
    try:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise BaseExceptionGroup("FSDP validation failed", errors)


async def _reference(model_path, processor_path, prompts, images):
    model = PickScoreNativeModel(model_path=model_path, processor_path=processor_path, device="cuda:0")
    try:
        output = await model.infer(prompts=prompts, images=images)
        result = [{key: value.detach().cpu() for key, value in item.items()} for item in output]
        memory = torch.cuda.memory_allocated()
        count = sum(parameter.numel() for parameter in model._inferencer.model.parameters())
        return result, memory, count
    finally:
        await model.close()


async def _manager_check(handle, prompts, images, reference):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"pipeline": {"output_type": "image"}, "val_kwargs": {"pipeline": {"output_type": "image"}}}
            },
            "reward": {
                "models": {"pickscore": {"backend": "native"}},
                "reward_functions": {
                    "pickscore": {
                        "path": "pkg://verl_omni.utils.reward_score.pickscore_reward",
                        "name": "compute_score_pickscore_native",
                        "weight": 0.7,
                        "required": True,
                    },
                    "jpeg": {
                        "path": "pkg://verl_omni.utils.reward_score.jpeg_compressibility",
                        "name": "compute_score",
                        "weight": 0.3,
                    },
                },
            },
        }
    )
    manager = MultiVisualRewardManager(config, tokenizer=None, compute_score=None)
    manager.set_reward_executors({}, {"pickscore": handle})
    phases = {}
    for validate in (False, True):
        samples = [
            DataProto.from_dict(
                tensors={"responses": torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)},
                non_tensors={
                    "data_source": np.array(["fsdp_acceptance"]),
                    "reward_model": np.array([{"ground_truth": prompt}]),
                    "media_kind": np.array(["image"]),
                },
                meta_info={"validate": validate},
            )
            for prompt, image in zip(prompts, images, strict=True)
        ]
        results = await asyncio.gather(*(manager.run_single(sample) for sample in samples))
        for index, (result, sample) in enumerate(zip(results, samples, strict=True)):
            expected = 0.7 * _pairwise_pickscore(**reference[index]).item() + 0.3 * jpeg_score(
                sample.batch["responses"][0], {"media_kind": "image"}
            )
            torch.testing.assert_close(
                torch.tensor(result["reward_score"]), torch.tensor(expected), rtol=1e-4, atol=2e-5
            )
            assert "reward/pickscore/pickscore_raw" in result["reward_extra_info"]
            assert "reward/jpeg" in result["reward_extra_info"]
            assert manager.assemble_rm_scores(sample, [result["reward_score"]]).shape == (1, 1)
        phases["validation" if validate else "training"] = results
    return phases


def main():
    """Compare full weights with real FSDP shards, then kill an actual rank."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-jsonl", type=Path)
    parser.add_argument("--residency-cycles", type=int, default=1)
    parser.add_argument("--tiny", action="store_true", help="Create a small random CLIP checkpoint under the output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(432)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select exactly two GPUs using CUDA_VISIBLE_DEVICES")
    model_path = args.model_path
    if args.tiny:
        config = CLIPConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 16,
            },
            projection_dim=16,
        )
        model_path = str(args.output / "tiny-clip")
        CLIPModel(config).save_pretrained(model_path)
    prompts = ["a photograph of a landscape", "a colorful pattern", "a photograph of a landscape"]
    images = _images()
    if args.media_jsonl:
        rows = [json.loads(line) for line in args.media_jsonl.read_text().splitlines()]
        images = [Image.open(rows[i % len(rows)]["output"]).convert("RGB") for i in range(17)]
        prompts = [rows[i % len(rows)]["input"] for i in range(17)]
        prompts[-1] += " with details" * 100
    reference, memory, count = asyncio.run(_reference(model_path, args.processor_path, prompts, images))
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path.resolve().relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__),
                Path("verl_omni/reward_loop/native_fsdp.py"),
                Path("verl_omni/reward_loop/reward_loop.py"),
                Path("verl_omni/workers/config/reward.py"),
                Path("verl_omni/trainer/main_diffusion_v1.py"),
                Path("verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"),
            )
        },
        "torch": torch.__version__,
        "model_path": str(Path(model_path).resolve()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(2)],
        "media_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in images],
        "unsharded_allocated_bytes": memory,
        "model_numel": count,
        "reference_scores": _scores(reference).tolist(),
        "scope": "Synchronous reward pipeline; real checkpoint; no RL throughput claim",
        "media_source": str(args.media_jsonl.resolve()) if args.media_jsonl else "deterministic synthetic RGB",
    }
    ray.init(
        num_cpus=8,
        num_gpus=2,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="nfsdp-"),
        object_store_memory=256 * 1024**2,
        runtime_env={"env_vars": {"OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}},
    )
    pool = RayResourcePool([2], max_colocate_count=1, name_prefix="fsdp_reward_acceptance")
    replica = None
    primary_error = None
    try:
        replica = build_replica(pool, model_path, args.processor_path)
        report["shards"] = replica.wake_up()
        assert len({rank["gpu_uuid"] for rank in report["shards"]}) == 2
        assert [rank["rank"] for rank in report["shards"]] == [0, 1]
        for rank in report["shards"]:
            assert rank["all_parameters_sharded"]
            assert rank["global_numel"] == count
            assert 0 < rank["local_numel"] < count * 0.6
        handle = NativeHandle(replica)
        results = []
        cases = [[0], [0, 1, 2], [2, 0, 1]]
        if len(images) >= 17:
            cases.extend([list(range(16)), list(reversed(range(17)))])
        for indices in cases:
            actual = asyncio.run(handle.infer([prompts[i] for i in indices], [images[i] for i in indices]))
            for index, output in zip(indices, actual, strict=True):
                for key in reference[index]:
                    torch.testing.assert_close(output[key], reference[index][key], rtol=1e-4, atol=2e-5)
            expected = _scores([reference[i] for i in indices])
            scores = _scores(actual)
            torch.testing.assert_close(scores, expected, rtol=1e-4, atol=2e-5)
            results.append(
                {"indices": indices, "scores": scores.tolist(), "max_abs_error": (scores - expected).abs().max().item()}
            )
        report["parity"] = results
        image_tensor = torch.from_numpy(np.asarray(images[0]).copy()).permute(2, 0, 1)
        scorer_result = asyncio.run(
            compute_score_pickscore_native("validation", image_tensor, prompts[0], {"media_kind": "image"}, handle)
        )
        torch.testing.assert_close(torch.tensor(scorer_result["score"]), _scores(reference)[0], rtol=1e-4, atol=2e-5)
        report["existing_native_scorer"] = scorer_result
        report["existing_multi_reward_manager"] = asyncio.run(_manager_check(handle, prompts, images, reference))
        report["asleep"] = replica.sleep()
        assert all(not status["awake"] and status["local_numel"] == 0 for status in report["asleep"])
        try:
            replica.infer([prompts[0]], [images[0]])
        except RuntimeError:
            report["asleep_inference_rejected"] = True
        else:
            raise AssertionError("asleep inference was accepted")
        report["reloaded"] = replica.wake_up()
        reloaded = asyncio.run(handle.infer(prompts, images))
        torch.testing.assert_close(_scores(reloaded), _scores(reference), rtol=1e-4, atol=2e-5)
        cycles = []
        for _ in range(args.residency_cycles):
            status = replica.sleep()
            assert all(row["local_numel"] == 0 for row in status)
            cycles.append([row["allocated_bytes"] for row in status])
            replica.wake_up()
            actual = asyncio.run(handle.infer(prompts, images))
            torch.testing.assert_close(_scores(actual), _scores(reloaded), rtol=0, atol=0)
        if len(cycles) >= 3:
            for rank in (0, 1):
                assert max(row[rank] for row in cycles[2:]) - min(row[rank] for row in cycles[2:]) < 2 * 1024**2
        report["residency_cycles"] = cycles
        # Inject a real worker exit, not a mocked exception or a failed model forward.
        ray.kill(replica.worker_group.workers[1], no_restart=True)
        started = time.monotonic()
        try:
            replica.infer([prompts[0]], [images[0]])
        except Exception as exc:
            report["rank_failure"] = {"type": type(exc).__name__, "seconds": time.monotonic() - started}
        else:
            raise AssertionError("rank loss returned success")
        try:
            replica.wake_up()
        except RuntimeError:
            report["failed_group_reuse_rejected"] = True
        else:
            raise AssertionError("failed replica was reused")
        for worker in replica.worker_group.workers:
            try:
                ray.get(worker.status.remote(), timeout=10)
            except ray.exceptions.RayActorError:
                continue
            raise AssertionError("rank actor survived group failure")
        report["all_rank_actors_dead"] = True
        # A fresh group must also close cleanly without relying on a failure.
        replica.close()
        replica = build_replica(pool, model_path, args.processor_path)
        replica.wake_up()
        replica.close()
        report["clean_close"] = True
        report["passed"] = True
    except BaseException as exc:
        primary_error = exc
    finally:
        finish_validation(replica, pool, args.output, report, primary_error)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
