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
"""Config retargeting for ``bagel_corl_sync``: omni entry config -> diffusion model/rollout schemas."""

from __future__ import annotations

from dataclasses import fields

from omegaconf import DictConfig, OmegaConf, open_dict

from verl_omni.workers.config.diffusion import DiffusionRolloutConfig, DiffusionSamplingConfig

#: ``actor_rollout_ref.rollout.agent.*`` knobs with no code default (RFC §5 knob SoT);
#: ``verl_omni/utils/config.py`` validates them at launch, so the trainer must not invent values.
BAGEL_AGENT_REQUIRED_KNOBS = ("gen_samples_per_call", "max_generate_passes", "max_und_turns")

#: Fields accepted by ``DiffusionModelConfig`` (plus ``_target_``); other omni YAML keys are stripped.
DIFFUSION_MODEL_KEYS = frozenset(
    {
        "path",
        "architecture",
        "transformer_config",
        "algorithm",
        "local_path",
        "tokenizer_path",
        "local_tokenizer_path",
        "hf_config",
        "model_type",
        "composite_mode",
        "load_tokenizer",
        "tokenizer",
        "processor",
        "extra_tokenizers",
        "extra_tokenizer_map",
        "use_shm",
        "trust_remote_code",
        "custom_chat_template",
        "external_lib",
        "enable_gradient_checkpointing",
        "attn_backend",
        "lora_rank",
        "lora_alpha",
        "lora_init_weights",
        "lr_gen",
        "corl_freeze_und",
        "corl_freeze_gen",
        "corl_no_optimizer_step",
        "target_modules",
        "target_parameters",
        "exclude_modules",
        "lora",
        "lora_adapter_path",
        "policy_state_adapters",
        "lora_dtype",
        "mtp",
        "pipeline",
        "algo",
        "fsdp_layer_prefixes",
        "config_path",
        "transformer_subfolder",
    }
)

_DIFFUSION_CFG = "verl_omni.workers.config.diffusion"
_AGENT_LOOP_MANAGER = "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.transfer_queue.BagelCorlAgentLoopManagerTQ"
_AGENT_CONFIG = "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.config.BagelCorlAgentLoopConfig"


def actor_merges_lora(model) -> bool:
    """True when the recipe asks the *actor* to merge LoRA into the base weights."""
    lora_cfg = model.get("lora") if hasattr(model, "get") else getattr(model, "lora", None)
    if lora_cfg is None:
        return False
    if hasattr(lora_cfg, "get"):
        return bool(lora_cfg.get("merge", False))
    return bool(getattr(lora_cfg, "merge", False))


def expected_s(config: DictConfig) -> int:
    """Seeds per ``generate_image`` call. No silent default: the GEN seed gate cannot infer S."""
    agent = config.actor_rollout_ref.rollout.get("agent") or {}
    raw = agent.get("gen_samples_per_call")
    if raw is None:
        raise ValueError(
            "bagel_corl_sync requires actor_rollout_ref.rollout.agent.gen_samples_per_call; "
            "the dual-lane GEN seed gate cannot infer S"
        )
    return int(raw)


def gen_adv_estimator(config: DictConfig) -> str:
    """GEN FlowGRPO estimator; never ``algorithm.adv_estimator`` (that is the UND token estimator)."""
    for candidate in (config.algorithm.get("gen_adv_estimator"), config.actor_rollout_ref.model.get("algorithm")):
        if candidate:
            return str(candidate)
    return "flow_grpo"


def _diffusion_model_dict(model_cfg: DictConfig) -> dict:
    """Filter the omni model node onto ``DiffusionModelConfig`` and pin the Bagel Co-RL defaults."""
    raw = OmegaConf.to_container(model_cfg, resolve=True) or {}
    filtered = {k: v for k, v in raw.items() if k in DIFFUSION_MODEL_KEYS}
    filtered["_target_"] = f"{_DIFFUSION_CFG}.DiffusionModelConfig"
    filtered.setdefault("algorithm", "flow_grpo")
    filtered["model_type"] = "diffusion_model"
    filtered.setdefault("architecture", "OmniBagelForConditionalGeneration")
    filtered.setdefault("composite_mode", "bagel_corl")
    filtered.setdefault("lr_gen", 3e-5)
    filtered.setdefault("trust_remote_code", True)
    lora_targets = filtered.get("target_modules")
    if not isinstance(lora_targets, list | tuple):
        raise ValueError(
            "bagel_corl_sync requires actor_rollout_ref.model.target_modules as an explicit "
            f"list of UND/GEN LoRA module names, got {lora_targets!r} (RFC §4.0.2 dual-lane LoRA)"
        )
    return filtered


def _agent_dict(agent_raw) -> dict:
    """Bind the agent node to ``BagelCorlAgentLoopConfig`` and require the RFC §5 knobs."""
    agent = dict(agent_raw or {})
    agent["_target_"] = _AGENT_CONFIG
    missing = [k for k in BAGEL_AGENT_REQUIRED_KNOBS if agent.get(k) is None]
    if missing:
        raise ValueError(
            f"bagel_corl_sync requires actor_rollout_ref.rollout.agent.{', '.join(missing)} "
            "(RFC §5 knob SoT, no code default)"
        )
    agent.setdefault("agent_loop_manager_class", _AGENT_LOOP_MANAGER)
    agent.setdefault("default_agent_loop", "bagel_multiturn_agent")
    agent.setdefault("und_ar_serving_ready", False)
    return agent


def _diffusion_rollout_dict(rollout_cfg: DictConfig) -> dict:
    """Filter the omni rollout node onto ``DiffusionRolloutConfig``.

    ``free_cache_engine`` is forced on (a falsy value disables ``AsyncOmni.sleep`` and the first
    ``update_weights`` OOMs). ``pipeline`` and ``val_kwargs`` carry explicit ``_target_`` because
    Hydra otherwise hands plain mappings to the typed fields.
    """
    allowed = {f.name for f in fields(DiffusionRolloutConfig)}
    raw = OmegaConf.to_container(rollout_cfg, resolve=True) or {}
    rollout = {k: v for k, v in raw.items() if k in allowed}
    rollout["_target_"] = f"{_DIFFUSION_CFG}.DiffusionRolloutConfig"
    rollout.setdefault("rollout_adapter", "default")
    rollout["free_cache_engine"] = True
    rollout.setdefault("enable_sleep_mode", True)
    rollout["agent"] = _agent_dict(rollout.get("agent"))
    rollout["calculate_log_probs"] = True
    algo = dict(rollout.get("algo") or {})
    algo.setdefault("noise_level", 0.7)
    algo.setdefault("sde_window_size", 2)
    algo.setdefault("sde_window_range", [0, 7])
    algo.setdefault("sde_type", "sde")
    rollout["algo"] = algo
    pipeline = dict(rollout.get("pipeline") or {})
    pipeline.setdefault("num_inference_steps", 10)
    pipeline.setdefault("cfg_text_scale", 1.0)
    pipeline.setdefault("cfg_img_scale", 1.0)
    pipeline["_target_"] = f"{_DIFFUSION_CFG}.DiffusionPipelineConfig"
    rollout["pipeline"] = pipeline
    val_allowed = {f.name for f in fields(DiffusionSamplingConfig)}
    val_kwargs = {k: v for k, v in (rollout.get("val_kwargs") or {}).items() if k in val_allowed and k != "_target_"}
    val_kwargs["_target_"] = f"{_DIFFUSION_CFG}.DiffusionSamplingConfig"
    rollout["val_kwargs"] = val_kwargs
    return rollout


def _ensure_actor_diffusion_loss(actor: DictConfig, loss_mode: str) -> None:
    """Bind ``actor.diffusion_loss`` to ``DiffusionLossConfig`` so every knob the loss reads exists."""
    target = f"{_DIFFUSION_CFG}.DiffusionLossConfig"
    if actor.get("diffusion_loss") is None:
        actor.diffusion_loss = OmegaConf.create({"_target_": target, "loss_mode": loss_mode})
        return
    if actor.diffusion_loss.get("loss_mode") is None:
        actor.diffusion_loss.loss_mode = loss_mode
    if actor.diffusion_loss.get("_target_") is None:
        actor.diffusion_loss["_target_"] = target


def _ensure_continuous_token(config: DictConfig) -> None:
    """verl's ``AgentLoopBase`` reads ``data.continuous_token``; a missing key aborts every episode."""
    data = config.get("data")
    if data is None:
        config.data = data = OmegaConf.create({})
    if data.get("continuous_token") is None:
        data.continuous_token = OmegaConf.create({"enable": False, "model_family": "auto"})
    elif data.continuous_token.get("enable") is None:
        data.continuous_token.enable = False


def _check_cfg_parity(config: DictConfig) -> None:
    """Training and rollout must share one transition kernel (RFC §5): both CFG scales must agree."""
    model = config.actor_rollout_ref.model.pipeline
    rollout = config.actor_rollout_ref.rollout.pipeline
    for key, hint in (
        ("cfg_text_scale", "set both to the same value (1.0 for CFG-free, 4.0 for canonical BAGEL)"),
        ("cfg_img_scale", "pin both via '+actor_rollout_ref.rollout.pipeline.cfg_img_scale=1.5'"),
    ):
        if float(model[key]) != float(rollout[key]):
            raise ValueError(
                f"bagel_corl_sync CFG parity violated: model.pipeline.{key}={float(model[key])} != "
                f"rollout.pipeline.{key}={float(rollout[key])}. Training and rollout must share one "
                f"transition kernel (RFC §5); {hint}."
            )


def rewrite_bagel_corl_configs(config: DictConfig) -> None:
    """Retarget ``actor_rollout_ref`` model/rollout to the diffusion schemas, in place.

    The model node also receives the rollout's ``pipeline``/``algo`` so the training adapter scores
    every GEN transition under the same kernel the rollout sampled from.
    """
    model = _diffusion_model_dict(config.actor_rollout_ref.model)
    rollout = _diffusion_rollout_dict(config.actor_rollout_ref.rollout)
    model["pipeline"] = dict(rollout["pipeline"])
    model["algo"] = {**rollout["algo"], "_target_": f"{_DIFFUSION_CFG}.DiffusionRolloutAlgoConfig"}
    with open_dict(config):
        config.actor_rollout_ref.model = OmegaConf.create(model)
        config.actor_rollout_ref.rollout = OmegaConf.create(rollout)
        _ensure_actor_diffusion_loss(config.actor_rollout_ref.actor, str(model.get("algorithm", "flow_grpo")))
        _ensure_continuous_token(config)
    _check_cfg_parity(config)
