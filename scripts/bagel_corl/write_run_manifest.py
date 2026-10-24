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

"""Write secret-free, content-addressed provenance from an already resolved JSON config.

No environment variables or arbitrary config subtrees are serialized. This is an
offline snapshot; the caller must invoke it again after resolving runtime LRs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

CONFIG_ALLOWLIST = (
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.model.lr_gen",
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio",
    "actor_rollout_ref.actor.optim.lr_scheduler_type",
    "actor_rollout_ref.actor.optim.min_lr_ratio",
    "actor_rollout_ref.actor.optim.weight_decay",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.model.tokenizer_path",
    "actor_rollout_ref.model.lora_alpha",
    "actor_rollout_ref.model.lora_dtype",
    "actor_rollout_ref.model.lora.merge",
    "actor_rollout_ref.model.target_modules",
    "actor_rollout_ref.model.algorithm",
    "actor_rollout_ref.model.composite_mode",
    "actor_rollout_ref.model.architecture",
    "actor_rollout_ref.actor.strategy",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_epochs",
    "actor_rollout_ref.actor.data_loader_seed",
    "actor_rollout_ref.actor.fsdp_config.seed",
    "actor_rollout_ref.rollout.seed",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.kl_loss_coef",
    "actor_rollout_ref.actor.entropy_coeff",
    "actor_rollout_ref.actor.diffusion_loss.clip_ratio",
    "actor_rollout_ref.actor.diffusion_loss.loss_weight_und",
    "actor_rollout_ref.actor.diffusion_loss.loss_weight_gen",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.agent.gen_samples_per_call",
    "actor_rollout_ref.rollout.agent.max_generate_passes",
    "algorithm.norm_adv_by_std_in_grpo",
    "actor_rollout_ref.rollout.pipeline.height",
    "actor_rollout_ref.rollout.pipeline.width",
    "actor_rollout_ref.rollout.pipeline.num_inference_steps",
    "actor_rollout_ref.rollout.algo.noise_level",
    "actor_rollout_ref.rollout.pipeline.cfg_text_scale",
    "actor_rollout_ref.rollout.pipeline.cfg_img_scale",
    "actor_rollout_ref.rollout.pipeline.max_sequence_length",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps",
    "actor_rollout_ref.rollout.val_kwargs.algo.noise_level",
    "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config",
    "actor_rollout_ref.rollout.agent.und_deploy_config",
    "actor_rollout_ref.rollout.agent.max_und_turns",
    "actor_rollout_ref.rollout.algo.sde_window_size",
    "actor_rollout_ref.rollout.algo.sde_window_range",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.repetition_penalty",
    "actor_rollout_ref.actor.fsdp_config.model_dtype",
    "data.train_files",
    "data.val_files",
    "data.holdout_files",
    "data.seed",
    "trainer.seed",
    "data.train_batch_size",
    "data.val_batch_size",
    "data.val_max_samples",
    "data.max_prompt_length",
    "data.max_response_length",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.total_training_steps",
    "trainer.project_name",
    "trainer.experiment_name",
    "scorer_version",
    "evaluation_scorer_version",
    "logprob_convention",
    "checkpoint_step",
    "effective_optimizer_group_lrs.und",
    "effective_optimizer_group_lrs.gen",
    "actor_rollout_ref.model.freeze_und",
    "actor_rollout_ref.model.corl_freeze_und",
    "actor_rollout_ref.model.corl_freeze_gen",
    "actor_rollout_ref.model.corl_no_optimizer_step",
    "actor_rollout_ref.rollout.agent.corl_fixed_gen_prompt",
    "agentic_image_gen.corl_reward_version",
    "agentic_image_gen.unified_good_enough_threshold",
    "agentic_image_gen.score_backend",
    "agentic_image_gen.good_enough_threshold",
    "agentic_image_gen.pickscore_good_enough_threshold",
    "reward.reward_model.model_path",
    "reward.reward_model.enable",
    "reward.custom_reward_function.path",
    "reward.custom_reward_function.name",
)
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".cpp", ".cu", ".h"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get(config, dotted, default=None):
    value = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def canonical(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def tree_hashes(path):
    if path.is_file():
        return {path.name: file_hash(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}


def write_run_manifest(output_dir: Path, resolved_config: dict, checkpoint_path: str | None) -> Path:
    root = Path(resolved_config.get("source_root", Path(__file__).resolve().parents[2])).resolve()
    output_dir = Path(output_dir).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    tracked = git("ls-files", "-z").decode().split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")

    def hashes(names, *, exclude_output=False):
        return {
            name: file_hash(root / name)
            for name in sorted(names)
            if name
            and Path(name).suffix.lower() in SOURCE_SUFFIXES
            and (root / name).is_file()
            and not (exclude_output and (root / name).resolve().is_relative_to(output_dir))
        }

    safe = {key: get(resolved_config, key) for key in CONFIG_ALLOWLIST if get(resolved_config, key) is not None}
    # A path may be allowed, but an authenticated URL must never become an artifact.
    for value in safe.values():
        values = value if isinstance(value, list | tuple) else [value]
        for leaf in values:
            if leaf is not None and not isinstance(leaf, str | int | float | bool):
                raise ValueError("Allowlisted config leaves must be scalar values or lists of scalars")
            if isinstance(leaf, str) and re.search(r"\w+://[^/]*[@?]|[?&](?:token|key|password)=", leaf, re.I):
                raise ValueError("Authenticated URLs are not permitted in provenance fields")
    datasets = {}
    for role in ("train", "val", "holdout"):
        values = get(resolved_config, f"data.{role}_files", [])
        if isinstance(values, str):
            values = [values]
        datasets[role] = []
        for value in values:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            content = tree_hashes(path)
            datasets[role].append(
                {"path": str(path.resolve()), "files_sha256": content, "sha256": sha256(canonical(content))}
            )
    checkpoint = {"path": None, "step": None, "status": "unavailable"}
    if checkpoint_path:
        path = Path(checkpoint_path).resolve()
        files = tree_hashes(path)
        if not files:
            raise ValueError("Checkpoint is empty")
        step = re.search(r"(?:global_step|step)[_-]?(\d+)", str(path))
        checkpoint = {
            "path": str(path),
            "step": get(resolved_config, "checkpoint_step", int(step[1]) if step else None),
            "files_sha256": files,
            "sha256": sha256(canonical(files)),
            "status": "content_hashed",
        }
    versions = {
        dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "git_sha": git("rev-parse", "HEAD").decode().strip(),
            "dirty_patch_sha256": sha256(git("diff", "HEAD", "--binary")),
            "tracked_source_sha256": hashes(tracked),
            "untracked_source_sha256": hashes(untracked, exclude_output=True),
        },
        "resolved_config_allowlist": safe,
        "config_sha256": sha256(canonical(safe)),
        "dependencies": dict(sorted(versions.items())),
        "datasets": datasets,
        "checkpoint": checkpoint,
        "optimizer_group_lrs": {
            "und": get(
                resolved_config,
                "effective_optimizer_group_lrs.und",
                get(resolved_config, "actor_rollout_ref.actor.optim.lr"),
            ),
            "gen": get(
                resolved_config,
                "effective_optimizer_group_lrs.gen",
                get(resolved_config, "actor_rollout_ref.model.lr_gen"),
            ),
        },
        "optimizer_lrs_source": "runtime" if "effective_optimizer_group_lrs" in resolved_config else "configured_peak",
        "scorer_versions": {
            "training": get(resolved_config, "scorer_version"),
            "evaluation": get(resolved_config, "evaluation_scorer_version"),
        },
        "logprob_convention": get(resolved_config, "logprob_convention", "unverified"),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "run_manifest.json"
    output.write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved JSON; never pass credential fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(write_run_manifest(args.output, json.loads(args.config.read_text(encoding="utf-8")), args.checkpoint))
