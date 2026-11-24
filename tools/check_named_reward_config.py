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
"""Check named reward model schemas and scorer references in a composed config."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path


class _ConfigCheckError(ValueError):
    """A controlled, value-free diagnostic for the scoped configuration check."""


def _model_schema_mapping(entry):
    """Resolve only fields read by the named-model schemas, preserving attached context."""
    from omegaconf import OmegaConf

    if not OmegaConf.is_config(entry):
        return entry

    model = OmegaConf.to_container(entry, resolve=False)

    def resolve_fields(source, target, fields):
        for field in fields:
            if field in source:
                value = source[field]
                target[field] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    def materialize_mapping(source, target, section):
        if section not in source:
            return None
        value = source[section]
        if not isinstance(value, Mapping):
            target[section] = value
            return None
        target[section] = OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else dict(value)
        return value

    resolve_fields(entry, model, ("backend",))
    backend = model.get("backend")
    if backend == "engine":
        resolve_fields(entry, model, ("offload", "model_path", "replicas", "n_gpus_per_node", "nnodes"))
        materialize_mapping(entry, model, "rollout")
    elif backend == "native":
        resolve_fields(entry, model, ("offload", "model_path", "dispatch_batch_size"))
        nested_fields = {
            "placement": ("devices", "replica_groups", "resource", "cpus_per_worker", "num_replicas"),
            "fsdp": ("timeout", "collective_timeout"),
        }
        for section, fields in nested_fields.items():
            value = materialize_mapping(entry, model, section)
            if value is not None:
                resolve_fields(value, model[section], fields)
        executor = materialize_mapping(entry, model, "executor")
        if executor is not None:
            resolve_fields(executor, model["executor"], ("model",))
            kwargs = materialize_mapping(executor, model["executor"], "kwargs")
            if kwargs is not None:
                resolve_fields(kwargs, model["executor"]["kwargs"], ("device",))
    return model


def _inspect_named_rewards(config) -> dict:
    """Return a narrow, ordered report after existing reward validation."""
    from verl_omni.workers.config.reward import (
        parse_reward_model_config,
        resolve_reward_model_name,
        reward_role_required,
        validate_reward_model_terms,
    )

    reward = config.reward
    if not isinstance(reward, Mapping):
        raise _ConfigCheckError("reward must be a mapping")
    models = reward.get("models")
    terms = reward.get("reward_functions")
    if models is None:
        models = {}
    if terms is None:
        terms = {}
    if not isinstance(models, Mapping):
        raise _ConfigCheckError("reward.models must be a mapping")
    if not isinstance(terms, Mapping):
        raise _ConfigCheckError("reward.reward_functions must be a mapping")
    for entry in models.values():
        if not isinstance(entry, Mapping):
            raise _ConfigCheckError("reward.models entries must be a mapping")
        if "placement" in entry and not isinstance(entry["placement"], Mapping):
            raise _ConfigCheckError("reward.models placement must be a mapping")
    for entry in terms.values():
        if not isinstance(entry, Mapping):
            raise _ConfigCheckError("reward.reward_functions entries must be a mapping")

    reward_model = reward.get("reward_model")
    if reward_model is None:
        raise _ConfigCheckError("reward.reward_model is required; compose a trainer root with its defaults")
    if not isinstance(reward_model, Mapping):
        raise _ConfigCheckError("reward.reward_model must be a mapping")
    if models and reward_model.get("enable", False):
        raise _ConfigCheckError("reward.reward_model.enable cannot be combined with reward.models")

    try:
        validate_reward_model_terms(config)
        named_terms = [
            {"name": name, "model": model_name}
            for name, term in terms.items()
            if (model_name := resolve_reward_model_name(name, term, models)) is not None
        ]
    except (ValueError, TypeError) as exc:
        raise _ConfigCheckError(
            "reward.reward_functions: check model references and path/name; "
            "check native FSDP term and pool requirements"
        ) from exc

    parsed_models = []
    for name, entry in models.items():
        try:
            model = _model_schema_mapping(entry)
            parsed_models.append(parse_reward_model_config(name, model))
        except (ValueError, TypeError) as exc:
            raise _ConfigCheckError("reward.models: check backend, placement, executor, and supported fields") from exc
    try:
        needs_role = reward_role_required(config)
    except (ValueError, TypeError) as exc:
        raise _ConfigCheckError("reward role requirements: check reward.reward_model and reward.models") from exc
    return {
        "scope": "named_reward_schema_and_references",
        "models": [{"name": model.name, "backend": model.backend} for model in parsed_models],
        "named_terms": named_terms,
        "requires_reward_role": needs_role,
    }


def main(argv: list[str] | None = None) -> int:
    """Compose trainer configuration and check only named model schemas/references."""
    parser = argparse.ArgumentParser(
        description="Check named reward schemas and references; not import availability or training readiness.",
        epilog="Validation imports normal repository dependencies. Configured scorer/model strings are not resolved.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "verl_omni" / "trainer" / "config",
        help="Hydra configuration root (default: this checkout's verl_omni/trainer/config)",
    )
    parser.add_argument("--config-name", default="diffusion_trainer", help="Hydra config name")
    args, overrides = parser.parse_known_args(argv)

    # Composition and selected validator imports may print; reserve stdout for JSON.
    with redirect_stdout(sys.stderr):
        checkout_root = str(Path(__file__).resolve().parents[1])
        if sys.path[0] != checkout_root:
            sys.path.insert(0, checkout_root)
        from hydra import compose, initialize_config_dir
        from hydra.errors import HydraException
        from omegaconf.errors import OmegaConfBaseException

        try:
            config_dir = args.config_dir.expanduser().resolve()
            if not config_dir.is_dir():
                raise _ConfigCheckError("Configuration directory does not exist; check --config-dir")
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                config = compose(config_name=args.config_name, overrides=overrides)
            report = _inspect_named_rewards(config)
        except (HydraException, OmegaConfBaseException) as exc:
            print(
                f"Named reward configuration error ({type(exc).__name__}): "
                "check the config root/name and Hydra override keys or syntax; details omitted to protect values",
                file=sys.stderr,
            )
            return 2
        except _ConfigCheckError as exc:
            print(f"Named reward configuration error: {exc}", file=sys.stderr)
            return 2

    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
