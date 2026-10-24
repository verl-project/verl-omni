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

"""Compose or run a named, explicitly isolated Linux/CUDA experiment profile."""

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROFILES = ("reproduction", "B0", "B1", "B2-1e-6", "B2-1e-5", "B3", "B4")


def build_profile(profile, seed, inherited):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    for key in ("START_CHECKPOINT", "UNICOT_TRAIN", "UNICOT_VAL", "EXPERIMENT_PREFIX"):
        if not inherited.get(key):
            raise ValueError(f"Explicit {key} is required")
    prefix = inherited["EXPERIMENT_PREFIX"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("EXPERIMENT_PREFIX must contain only letters, numbers, underscores and hyphens")
    name = f"{prefix}_{profile}_seed{seed}"
    output_root = Path(inherited.get("EXPERIMENT_ROOT", str(ROOT / "outputs/corl_convergence")))
    run_dir = output_root / name
    env = dict(inherited)
    env.update(
        BAGEL_MODEL_PATH=inherited["START_CHECKPOINT"],
        REBUILD_UNICOT="0",
        RESUME_MODE="disable",
        EXPERIMENT_NAME=name,
        RUN_NAME=name,
        RUN_DIR=str(run_dir / "train"),
        DEFAULT_LOCAL_DIR=str(output_root / "checkpoints" / name),
        PROJECT_NAME="bagel_corl_convergence",
        TOTAL_STEPS="50",
        SAVE_FREQ="25",
        MAX_CKPT_KEEP="3",
        TEST_FREQ="25",
        VAL_MAX_SAMPLES="-1",
        JUDGE_SERVER="0",
        TRAIN_BSZ="4",
        PPO_MINI="4",
        PPO_MICRO="1",
        GEN_HW="512",
        GEN_STEPS="10",
        VAL_GEN_STEPS="50",
        VAL_NOISE_LEVEL="0.0",
        NOISE_LEVEL="0.8",
        SDE_WINDOW_SIZE="2",
        SDE_WINDOW_RANGE="[0,2]",
        GEN_CLIP_RATIO="0.02",
        CFG_TEXT_SCALE="4.0",
        CFG_IMG_SCALE="1.5",
        GEN_DECODE="sde",
        NORM_ADV_BY_STD="True",
        ACTOR_LR="1e-5",
        GEN_LR="3e-5",
        N="4",
        S="2",
        MAX_GENERATE_PASSES="1",
    )
    freeze_und = freeze_gen = no_step = False
    und_weight = gen_weight = 1.0
    if profile == "reproduction":
        env.update(ACTOR_LR="3e-5", GEN_LR="3e-5", N="8", S="1", MAX_GENERATE_PASSES="3", TOTAL_STEPS="200")
    elif profile == "B0":
        no_step = True
    elif profile == "B1":
        freeze_und, und_weight = True, 0.0
    elif profile.startswith("B2-"):
        freeze_gen, gen_weight = True, 0.0
        env.update(ACTOR_LR=profile.split("-", 1)[1], N="8", S="1")
    elif profile == "B4":
        env["MAX_GENERATE_PASSES"] = "3"
    overrides = [
        f"data.seed={seed}",
        f"actor_rollout_ref.actor.data_loader_seed={seed}",
        f"actor_rollout_ref.actor.fsdp_config.seed={seed}",
        f"actor_rollout_ref.rollout.seed={seed}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        f"+actor_rollout_ref.model.corl_freeze_und={freeze_und}",
        f"+actor_rollout_ref.model.corl_freeze_gen={freeze_gen}",
        f"+actor_rollout_ref.model.corl_no_optimizer_step={no_step}",
        f"++actor_rollout_ref.actor.diffusion_loss.loss_weight_und={und_weight}",
        f"++actor_rollout_ref.actor.diffusion_loss.loss_weight_gen={gen_weight}",
        f"++agentic_image_gen.corl_reward_version={'v1' if profile == 'reproduction' else 'v2'}",
    ]
    if profile == "B1":
        overrides.append("+actor_rollout_ref.rollout.agent.corl_fixed_gen_prompt=__user_request__")
    return env, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print profile env/argv without invoking Bash or training"
    )
    parser.add_argument(
        "--resolve-only", action="store_true", help="Hydra composition plus allowlisted manifest, no training"
    )
    parser.add_argument(
        "--gates", type=Path, help="Host-verified JSON with gate_A_passed and frozen_policy_parity_passed"
    )
    parser.add_argument("overrides", nargs="*", help="Additional reviewed Hydra overrides")
    args = parser.parse_args()
    env, overrides = build_profile(args.profile, args.seed, os.environ)
    command = [
        "bash",
        str(ROOT / "examples/agenticllmgrpo_trainer/bagel/run_agentic_bagel_rpco_lora.sh"),
        *overrides,
        *args.overrides,
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "environment_overrides": {k: v for k, v in env.items() if os.environ.get(k) != v},
                    "command": shlex.join(command),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if os.name == "nt":
        parser.error("Composition/training requires the Linux training host; --dry-run works on Windows")
    if not args.resolve_only:
        if not args.gates:
            parser.error("Training requires --gates from completed manual CPU/CUDA/publication/parity checks")
        gates = json.loads(args.gates.read_text(encoding="utf-8"))
        if gates.get("gate_A_passed") is not True or gates.get("frozen_policy_parity_passed") is not True:
            parser.error("Gate A and frozen-policy parity must both pass before training")
    # Resolve first using the recipe's real CLI/environment. Do not persist raw
    # stdout/config: only explicit allowlisted fields enter artifacts.
    resolved = subprocess.run(
        [*command, "--cfg", "job", "--resolve"], env=env, cwd=ROOT, text=True, capture_output=True, check=True
    )
    import yaml

    match = re.search(r"^[A-Za-z_][A-Za-z0-9_]*:", resolved.stdout, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not locate Hydra YAML in composition output")
    config = yaml.safe_load(resolved.stdout[match.start() :])
    if not isinstance(config, dict) or "actor_rollout_ref" not in config:
        raise ValueError("Hydra composition did not yield the trainer config")
    config["source_root"] = str(ROOT)
    spec = importlib.util.spec_from_file_location(
        "write_run_manifest", ROOT / "scripts/bagel_corl/write_run_manifest.py"
    )
    manifest_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest_tool)
    artifact_dir = Path(env["RUN_DIR"]).parent / "provenance"
    manifest = manifest_tool.write_run_manifest(artifact_dir, config, env["START_CHECKPOINT"])
    # This intentionally contains only reviewed semantic fields, never credential
    # subtrees. A runtime snapshot must later record effective optimizer group LRs.
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))["resolved_config_allowlist"]
    (artifact_dir / "resolved-config.allowlisted.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest)
    if not args.resolve_only:
        subprocess.run(command, env=env, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
