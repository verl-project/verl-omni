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

"""Opt-in frozen actor replay capture; no parity or replica-coverage claim is made here.

Set BAGEL_CORL_PARITY_CAPTURE_DIR and BAGEL_CORL_PARITY_MANIFEST (JSON file
with policy_version, calibration_id, verified logprob_convention and nonempty
kernel_config.{und,gen}). The manifest is a host operator's attestation of the
installed serving probability convention and resolved kernel configuration.
BAGEL_CORL_PARITY_REPLICA_LABEL optionally identifies a manually isolated physical
rollout replica during calibration. It must never label a mixed-replica batch.
Without that label every row must carry its real rollout_replica_id. Actor DP
rank and trainer process ID are deliberately never used as rollout identities.

JSONL rows can be consumed by scripts/bagel_corl/check_policy_parity.py. Combine
captures of identical contexts across physical replicas and weight publications
on the host; this module cannot establish such coverage or verify weight loads.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import uuid
from pathlib import Path

import torch


def exact(value):
    """JSON representation retaining tensor dtype, shape and all context elements."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.is_nested:
            return [exact(item) for item in value.unbind()]
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {str(key): exact(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [exact(item) for item in value]
    if hasattr(value, "tolist"):
        return exact(value.tolist())
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"Unsupported exact parity context value: {type(value).__name__}")


def digest(value):
    return hashlib.sha256(json.dumps(exact(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def metadata(row):
    return {**(row.get("extra_fields") or {}), **row}


def required(row, key):
    value = row.get(key)
    if value is None:
        raise ValueError(f"Parity capture requires {key}")
    return value


def scalar(value):
    return value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value


class FrozenPolicyCapture:
    def __init__(self, directory, manifest, manual_replica_label=None):
        self.manifest = manifest
        for key in ("policy_version", "calibration_id", "logprob_convention", "kernel_config"):
            required(manifest, key)
        if not manifest["calibration_id"] or manifest["logprob_convention"] not in {
            "raw_policy",
            "exact_behavior_transforms",
        }:
            raise ValueError("Parity capture requires calibrated, verified logprob_convention")
        for lane in ("und", "gen"):
            if not manifest["kernel_config"].get(lane):
                raise ValueError(f"Parity capture requires resolved kernel_config.{lane}")
        self.manual_replica_label = manual_replica_label
        self.path = Path(directory) / f"policy_parity_{uuid.uuid4().hex}.jsonl"
        self.versions = {}
        for package in ("torch", "verl", "verl-omni", "vllm"):
            try:
                self.versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.versions[package] = None

    @classmethod
    def from_environment(cls, trainer):
        directory = os.environ.get("BAGEL_CORL_PARITY_CAPTURE_DIR")
        if not directory:
            return None
        manifest_path = os.environ.get("BAGEL_CORL_PARITY_MANIFEST")
        if not manifest_path:
            raise ValueError("BAGEL_CORL_PARITY_CAPTURE_DIR requires BAGEL_CORL_PARITY_MANIFEST")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest.get("policy_version") != getattr(trainer, "global_steps", None):
            raise ValueError("Parity manifest policy_version differs from actor trainer global_steps")
        correction = trainer.config.algorithm.get("rollout_correction") or {}
        if correction.get("bypass_mode", False):
            raise ValueError("Parity capture requires actor recomputation; bypass_mode is enabled")
        if getattr(trainer, "actor_rollout_wg", None) is None:
            raise ValueError("Parity capture requires a live actor worker group")
        return cls(directory, manifest, os.environ.get("BAGEL_CORL_PARITY_REPLICA_LABEL"))

    def provenance(self, source):
        source = metadata(source)
        version = source.get("policy_version")
        if version is None:
            low, high = source.get("min_global_steps"), source.get("max_global_steps")
            if low is not None and scalar(low) == scalar(high):
                version = scalar(low)
        if version is None:
            raise ValueError("Parity capture requires rollout policy_version (or equal min/max_global_steps)")
        if scalar(version) != self.manifest["policy_version"]:
            raise ValueError("Stale rollout policy version in parity capture")
        replica = source.get("rollout_replica_id")
        identity_source = "rollout_metadata"
        if replica is None:
            replica = self.manual_replica_label
            identity_source = "manual_physical_replica_calibration_label"
        if replica is None or replica == "":
            raise ValueError(
                "Parity capture requires true rollout_replica_id or manual physical replica calibration label"
            )
        return dict(
            policy_version=scalar(version),
            rollout_policy_version=scalar(version),
            actor_policy_version=self.manifest["policy_version"],
            replica_id=scalar(replica),
            replica_identity_source=identity_source,
            complete_replica_coverage_verified=False,
        )

    def rows(self, lane, source, row_id, context, mask, rollout, old, repeat, times=None):
        provenance = self.provenance(source)
        episode = scalar(required(metadata(source), "episode_uid"))
        vectors = [torch.as_tensor(item).reshape(-1) for item in (mask, rollout, old, repeat)]
        if len({item.numel() for item in vectors}) != 1:
            raise ValueError("Parity mask/logprob lengths differ")
        mask, rollout, old, repeat = vectors
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Invalid parity mask")
        context = exact(context)
        common = dict(
            lane=lane,
            **provenance,
            episode_id=episode,
            gen_row_id=scalar(row_id),
            context=context,
            context_hash=digest(context),
            mask=exact(mask),
            kernel_config_hash=digest(self.manifest["kernel_config"][lane]),
            kernel_config=self.manifest["kernel_config"][lane],
            package_versions=self.versions,
            calibration_id=self.manifest["calibration_id"],
            logprob_convention=self.manifest["logprob_convention"],
            scope="Captured rows only; physical replica assignment and publication coverage require host verification",
        )
        result = []
        for index in range(mask.numel()):
            if not mask[index]:
                continue
            values = [float(item[index]) for item in (rollout, old, repeat)]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite parity log-probability")
            result.append(
                dict(
                    common,
                    timestep=index,
                    transition_timestep=None if times is None else float(times[index]),
                    rollout_logp=values[0],
                    actor_old_logp=values[1],
                    actor_repeat_logp=values[2],
                )
            )
        return result

    def append(self, rows):
        # Validate and serialize the complete lane before creating any partial artifact.
        payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)

    @staticmethod
    def und_context(row):
        required(row, "input_ids")
        required(row, "responses")
        required(row, "position_ids")
        return {
            key: row[key]
            for key in (
                "input_ids",
                "prompts",
                "responses",
                "position_ids",
                "attention_mask",
                "multi_modal_inputs",
                "response_mask",
                "loss_mask",
            )
            if key in row
        }

    def write_und(self, rollout_rows, old_rows, repeat_rows):
        if len({len(items) for items in (rollout_rows, old_rows, repeat_rows)}) != 1:
            raise ValueError("Parity UND row counts differ")
        output = []
        for raw, old, repeat in zip(rollout_rows, old_rows, repeat_rows, strict=True):
            context = self.und_context(raw)
            if any(digest(self.und_context(row)) != digest(context) for row in (old, repeat)):
                raise ValueError("UND context changed between frozen replays")
            output.extend(
                self.rows(
                    "und",
                    raw,
                    "und",
                    context,
                    required(raw, "response_mask"),
                    required(raw, "rollout_log_probs"),
                    required(old, "old_log_probs"),
                    required(repeat, "old_log_probs"),
                )
            )
        self.append(output)

    def write_gen(self, proto, old, repeat, gen_rows, und_rows):
        parents = {metadata(row)["episode_uid"]: row for row in und_rows}
        usable = [
            row
            for row in gen_rows
            if all(
                row.get(key) is not None for key in ("all_latents", "rollout_log_probs", "rm_score", "gen_group_uid")
            )
            and torch.as_tensor(row["rollout_log_probs"]).numel() > 0
            and (row.get("all_timesteps") is not None or row.get("timesteps") is not None)
        ]
        if len(usable) != len(proto):
            raise ValueError("Cannot align exact GEN capture rows with actor proto")
        output = []
        for index, raw in enumerate(usable):
            parent = parents.get(required(raw, "episode_uid"))
            if parent is None:
                raise ValueError("GEN capture lacks parent rollout provenance")
            source = {**metadata(parent), **raw}
            context = {key: proto.batch[key][index] for key in ("all_latents", "all_timesteps", "gen_transition_valid")}
            context["prompt_token_ids"] = proto.non_tensor_batch["prompt_token_ids"][index]
            required(context, "prompt_token_ids")
            output.extend(
                self.rows(
                    "gen",
                    source,
                    raw.get("gen_sample_uid", scalar(proto.batch["gen_row_id"][index])),
                    context,
                    proto.batch["gen_transition_valid"][index],
                    proto.batch["old_log_probs"][index],
                    old.batch["old_log_probs"][index],
                    repeat.batch["old_log_probs"][index],
                    proto.batch["all_timesteps"][index],
                )
            )
        self.append(output)

    @staticmethod
    def fetch_und(batch, normalize):
        import transfer_queue as tq

        keys = list(batch.keys)
        data = tq.kv_batch_get(keys=keys, partition_id=batch.partition_id)
        normalized = normalize(data, keys)
        # Clone tensor contexts now: subsequent actor work can mutate TQ-backed storage.
        import copy

        rows = [copy.deepcopy(normalized[str(key)]["fields"]) for key in keys]
        return rows, data

    def recompute_und(self, batch, metrics, recompute, normalize, rollout_rows):
        import transfer_queue as tq

        result = recompute(batch, metrics)
        first, first_data = self.fetch_und(result, normalize)
        preserved = first_data.select("old_log_probs", "entropy").clone()
        try:
            repeated = recompute(result, {})
            second, _ = self.fetch_und(repeated, normalize)
            self.write_und(rollout_rows, first, second)
        finally:
            tq.kv_batch_put(keys=result.keys, partition_id=result.partition_id, fields=preserved)
        return result
