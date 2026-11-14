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
"""Reproduce the 64-step SD3.5 study and compare checkpoint-32 continuation.

Run prepare, reference, resume, verify and plot in order. Reference/resume need three
GPUs exposed by CUDA_VISIBLE_DEVICES and real local SD3.5/PickScore checkpoints.
The source data and selection rule are pinned; preparation needs network access.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_REVISION = "4b13ec7efddcb93bbd65a027f89f791364d8416e"
SOURCE_SHA256 = {
    "train": "2f425adcbb9a6d0feee7c6c448e1cb83a1f71deca9434b803907e2578eca837f",
    "test": "d75c025fa451866e70f7e36243f7cdd93095523cb51f5f9651f2ce3236d20262",
}


def prepare(args):
    """Select 128/32 disjoint real prompts and write portable Hydra overrides."""
    from examples.flowgrpo_trainer.data_process.sd3_pickscore_sfw import build_split, read_prompt_file

    source, data = args.output / "source-data", args.output / "data"
    source.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    raw = {}
    for split, expected in SOURCE_SHA256.items():
        path = source / f"{split}.txt"
        if not path.exists():
            url = (
                f"https://raw.githubusercontent.com/yifan123/flow_grpo/{SOURCE_REVISION}/dataset/pickscore/{split}.txt"
            )
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
        raw[split] = read_prompt_file(path)
    selected = {}
    train_all = set(raw["train"])
    for split, count in (("train", 128), ("test", 32)):
        seen, items = set(), []
        for index, text in enumerate(raw[split]):
            if text in seen or (split == "test" and text in train_all):
                continue
            if not (5 <= len(text.split()) <= 20 and len(text) <= 140):
                continue
            seen.add(text)
            items.append({"source_index": index, "prompt": text})
            if len(items) == count:
                break
        assert len(items) == count
        selected[split] = items
        build_split([item["prompt"] for item in items], split, data_source="flow_grpo/pickscore").to_parquet(
            str(data / f"{split}.parquet")
        )
    assert not ({x["prompt"] for x in selected["train"]} & {x["prompt"] for x in selected["test"]})
    manifest = {"revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA256, "selected": selected}
    manifest["parquet_sha256"] = {
        split: hashlib.sha256((data / f"{split}.parquet").read_bytes()).hexdigest() for split in selected
    }
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    template = json.loads(Path(__file__).with_name("native_fsdp_sd35_overrides.json").read_text())
    for phase in ("reference", "resumed"):
        tokens = {
            "@DATA@": str(data),
            "@RUN@": str(args.output / "runs" / phase),
            "@SD35@": str(args.sd35_model),
            "@PICKSCORE@": str(args.pickscore_model),
            "@PROCESSOR@": str(args.processor),
            "@STEPS@": str(args.steps),
            "@INTERVAL@": str(args.interval),
            "@RAY_TMP@": f"/tmp/nfsdp-{hashlib.sha256(str(args.output).encode()).hexdigest()[:8]}-{phase}",
        }
        values = []
        for value in template:
            for token, replacement in tokens.items():
                value = value.replace(token, replacement)
            if phase == "resumed" and value == "trainer.resume_mode=disable":
                value = "trainer.resume_mode=resume_path"
            values.append(value)
        if phase == "resumed":
            values.append(
                f"trainer.resume_from_path={args.output}/runs/reference/checkpoints/global_step_{args.resume_step}"
            )
        (args.output / f"{phase}-overrides.json").write_text(json.dumps(values, indent=2) + "\n")


def run_training(args, phase):
    """Use the existing trainer-exit validator, retaining its real process status."""
    command = [
        sys.executable,
        "-m",
        "tests.special_e2e.validate_native_fsdp_training_failure",
        "--overrides",
        str(args.output / f"{phase}-overrides.json"),
        "--checkpoint-root",
        str(args.output / "runs" / phase / "checkpoints"),
        "--expected-step",
        str(args.steps),
        "--expect-success",
        "--output",
        str(args.output / f"{phase}-exit.json"),
    ]
    with (args.output / f"{phase}.log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execv(sys.executable, command)


def media_rows(path, expected):
    """Validate saved images and key responses independently of random Ray UIDs."""
    from PIL import Image

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == expected, path
    result = {}
    for row in rows:
        assert math.isfinite(row["score"])
        with Image.open(row["output"]) as image:
            assert image.size == (384, 384) and any(low < high for low, high in image.getextrema())
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
        response = row["uid"].rsplit("_", 2)[-2] if row.get("uid") else "0"
        key = (row["gts"] or row["input"], response)
        assert key not in result
        result[key] = {"score": row["score"], "pixels_sha256": pixels}
    return result


def verify(args):
    """Check updates, complete curves, checkpoint state and matched continuation."""
    import torch

    from tests.special_e2e.verify_native_fsdp_training import local, optimizer_steps, validate_training_log

    torch.set_num_threads(4)
    curves, observations, gradients = [], {}, {}
    skipped_training_media = {}
    manifest = json.loads((args.output / "data/manifest.json").read_text())
    prompts = {split: {row["prompt"] for row in rows} for split, rows in manifest["selected"].items()}
    for phase, first_step in (("reference", 1), ("resumed", args.resume_step + 1)):
        exit_report = json.loads((args.output / f"{phase}-exit.json").read_text())
        actors = exit_report["actors_before_ray_shutdown"]
        assert exit_report["passed"] and len(actors) == 5 and all(row["State"] == "DEAD" for row in actors)
        assert exit_report["latest_checkpoint"] == args.steps
        metrics = {}
        log = (args.output / f"{phase}.log").read_text()
        validate_training_log(log, args.steps, first_step - 1)
        for line in log.splitlines():
            if "actor/grad_norm:" not in line:
                continue
            fields = {}
            for name in ("training/global_step", "actor/grad_norm", "critic/rewards/mean"):
                fields[name] = float(re.search(re.escape(name) + r":[ \t]*([^\s]+)", line).group(1))
                assert math.isfinite(fields[name])
            step = int(fields["training/global_step"])
            assert step not in metrics and fields["actor/grad_norm"] > 0
            metrics[step] = fields
        assert sorted(metrics) == list(range(first_step, args.steps + 1))
        gradients[phase] = {step: fields["actor/grad_norm"] for step, fields in metrics.items()}
        observations[phase] = {}
        skipped_training_media[phase] = []
        run = args.output / "runs" / phase
        for kind, steps, count in (
            ("rollouts", range(first_step, args.steps + 1), 8),
            ("validation", range(0 if phase == "reference" else args.resume_step, args.steps + 1, args.interval), 32),
        ):
            observations[phase][kind] = {}
            for step in steps:
                media_path = run / kind / f"{step}.jsonl"
                if kind == "rollouts" and not media_path.exists():
                    assert f"Skipping media dump at step {step}: previous dump is still running" in log, media_path
                    skipped_training_media[phase].append(step)
                    curves.append(
                        {
                            "phase": phase,
                            "kind": kind,
                            "step": step,
                            "count": count,
                            "prompt_count": 4,
                            "mean": metrics[step]["critic/rewards/mean"],
                            "standard_error": "",
                            "media_saved": False,
                        }
                    )
                    continue
                rows = media_rows(media_path, count)
                observed_prompts = {prompt for prompt, _ in rows}
                if kind == "rollouts":
                    assert len(observed_prompts) == 4 and observed_prompts <= prompts["train"]
                    assert all((prompt, response) in rows for prompt in observed_prompts for response in ("0", "1"))
                else:
                    assert observed_prompts == prompts["test"]
                observations[phase][kind][step] = rows
                scores = [row["score"] for row in rows.values()]
                mean = sum(scores) / count
                if kind == "rollouts":
                    assert math.isclose(mean, metrics[step]["critic/rewards/mean"], rel_tol=1e-6, abs_tol=1e-7)
                prompt_means = [
                    sum(row["score"] for (text, _), row in rows.items() if text == prompt)
                    / (count / len(observed_prompts))
                    for prompt in observed_prompts
                ]
                variance = sum((score - mean) ** 2 for score in prompt_means) / (len(prompt_means) - 1)
                curves.append(
                    {
                        "phase": phase,
                        "kind": kind,
                        "step": step,
                        "count": count,
                        "prompt_count": len(prompt_means),
                        "mean": mean,
                        "standard_error": math.sqrt(variance / len(prompt_means)),
                        "media_saved": True,
                    }
                )
        optimizer_steps(run / f"checkpoints/global_step_{args.steps}/actor/optim_world_size_1_rank_0.pt", args.steps)
    with (args.output / "curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)

    matched = {}
    for kind in ("rollouts", "validation"):
        pairs = []
        for step, rows in observations["resumed"][kind].items():
            if step not in observations["reference"][kind]:
                assert kind == "rollouts" and step in skipped_training_media["reference"]
                continue
            reference = observations["reference"][kind][step]
            assert rows.keys() == reference.keys(), (kind, step, "prompt/response mismatch")
            pairs.extend((rows[key], reference[key]) for key in rows)
        matched[kind] = {
            "samples": len(pairs),
            "max_score_abs_error": max(abs(a["score"] - b["score"]) for a, b in pairs),
            "pixels_equal": all(a["pixels_sha256"] == b["pixels_sha256"] for a, b in pairs),
        }
    checkpoint_paths = [
        args.output / "runs" / phase / f"checkpoints/global_step_{args.steps}/actor"
        for phase in ("reference", "resumed")
    ]
    checkpoints = [
        torch.load(path / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
        for path in checkpoint_paths
    ]
    assert checkpoints[0].keys() == checkpoints[1].keys()
    model_equal = True
    model_max_error = 0.0
    for name, raw in checkpoints[0].items():
        left, right = local(raw), local(checkpoints[1][name])
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        model_equal = model_equal and torch.equal(left, right)
        if left.numel():
            model_max_error = max(model_max_error, (left.float() - right.float()).abs().max().item())
    previous_path = args.output / f"runs/reference/checkpoints/global_step_{args.resume_step}/actor"
    previous = torch.load(
        previous_path / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True
    )
    assert previous.keys() == checkpoints[0].keys()
    changed_lora, frozen = 0, 0
    for name, raw in previous.items():
        left, right = local(raw), local(checkpoints[0][name])
        assert left.shape == right.shape and left.dtype == right.dtype and torch.isfinite(left).all()
        if "lora_" in name:
            changed_lora += int(left.numel() > 0 and not torch.equal(left, right))
        else:
            assert torch.equal(left, right), (name, "frozen tensor changed")
            frozen += 1
    assert changed_lora > 0 and frozen > 0
    optimizer_steps(previous_path / "optim_world_size_1_rank_0.pt", args.resume_step)
    del previous
    del checkpoints
    optimizers = [
        torch.load(path / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
        for path in checkpoint_paths
    ]
    assert optimizers[0]["state"].keys() == optimizers[1]["state"].keys()
    optimizer_equal = optimizers[0]["param_groups"] == optimizers[1]["param_groups"]
    for index, state in optimizers[0]["state"].items():
        peer = optimizers[1]["state"][index]
        assert state.keys() == peer.keys()
        for key, value in state.items():
            other = peer[key]
            optimizer_equal = optimizer_equal and (
                torch.equal(local(value), local(other)) if isinstance(value, torch.Tensor) else value == other
            )
    report = {
        "skipped_training_media": skipped_training_media,
        "inspected_media_samples": {
            phase: {kind: sum(len(rows) for rows in steps.values()) for kind, steps in kinds.items()}
            for phase, kinds in observations.items()
        },
        "model_equal": model_equal,
        "model_max_abs_error": model_max_error,
        "optimizer_equal": optimizer_equal,
        "reference_checkpoint_transition": {"changed_lora_tensors": changed_lora, "unchanged_frozen_tensors": frozen},
        "matched": matched,
        "gradient_norms_equal": all(
            gradients["reference"][step] == value for step, value in gradients["resumed"].items()
        ),
    }
    report["passed"] = (
        model_equal
        and optimizer_equal
        and report["gradient_norms_equal"]
        and all(row["pixels_equal"] and row["max_score_abs_error"] == 0 for row in matched.values())
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["passed"], "Continuation differs; inspect the report before claiming equivalence"


def plot(args):
    """Plot every observed mean; validation shading is one SEM across 32 prompts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (args.output / "curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for axis, kind, title in zip(
        axes, ("rollouts", "validation"), ("Training rollouts", "Held-out validation"), strict=True
    ):
        for phase, label, color, style in (
            ("reference", f"Uninterrupted (1–{args.steps})", "#1764ab", "-"),
            ("resumed", f"Restored checkpoint {args.resume_step}", "#d55e00", "--"),
        ):
            selected = [row for row in rows if row["kind"] == kind and row["phase"] == phase]
            steps = [int(row["step"]) for row in selected]
            means = [float(row["mean"]) for row in selected]
            axis.plot(steps, means, style, color=color, label=label, marker="o", markersize=3, linewidth=1.4)
            if kind == "validation" and phase == "reference":
                errors = [float(row["standard_error"]) for row in selected]
                axis.fill_between(
                    steps,
                    [a - b for a, b in zip(means, errors, strict=True)],
                    [a + b for a, b in zip(means, errors, strict=True)],
                    color=color,
                    alpha=0.12,
                )
        axis.axvline(args.resume_step, color="0.4", linestyle=":", linewidth=1)
        axis.set(
            title=title, xlabel="Trainer step", ylabel="Mean PickScore (existing divisor 26)", xlim=(0, args.steps + 1)
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("SD3.5 Medium FlowGRPO · frozen two-rank FSDP2 PickScore", fontsize=13)
    fig.supxlabel(
        "384 × 384 · 12 denoising steps · seed 42 · "
        "training: 4 prompts × 2 responses/update; validation: 32 fixed prompts",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"sd35_reward_curve.{extension}", dpi=180)
    plt.close(fig)


def main():
    """Run one explicitly selected phase of the fixed acceptance study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "reference", "resume", "verify", "plot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sd35-model", type=Path)
    parser.add_argument("--pickscore-model", type=Path)
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--resume-step", type=int, default=32)
    parser.add_argument("--interval", type=int, default=8)
    args = parser.parse_args()
    if not (
        0 < args.resume_step < args.steps
        and args.interval > 0
        and args.steps % args.interval == 0
        and args.resume_step % args.interval == 0
    ):
        parser.error("steps and resume-step must be positive interval multiples, with resume-step < steps")
    args.output = args.output.resolve()
    if args.phase == "prepare":
        for value in (args.sd35_model, args.pickscore_model, args.processor):
            if value is None or not value.is_dir():
                parser.error("prepare requires existing --sd35-model, --pickscore-model and --processor directories")
        args.sd35_model, args.pickscore_model, args.processor = (
            value.resolve() for value in (args.sd35_model, args.pickscore_model, args.processor)
        )
        prepare(args)
    elif args.phase == "verify":
        verify(args)
    elif args.phase == "plot":
        plot(args)
    else:
        run_training(args, "resumed" if args.phase == "resume" else "reference")


if __name__ == "__main__":
    main()
