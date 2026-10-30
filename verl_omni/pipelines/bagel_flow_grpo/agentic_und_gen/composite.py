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
"""Bagel Co-RL (Joint-Training) composite actor: UND token path + GEN diffusion path on one FSDP module.

RFC: one ``update_actor`` / one ``optimizer.step``. Outer owner is
``verl.trainer.main_ppo.TaskRunnerV1`` via ``OmniBagelCoRLTrainerSync``. GEN reuses
diffusion V1 engine math (``PPODiffusersFSDPEngine`` timestep loop + ``diffusion_loss``)
and bound ``PolicyGradientDiffusionTrainerV1`` hooks, not ``fit()``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.workers.engine.utils import prepare_micro_batches

logger = logging.getLogger(__name__)

_UND_SELECT = (
    "bagel_episode_id",
    "input_ids",
    "attention_mask",
    "response_mask",
    "old_log_probs",
    "advantages",
    "responses",
    "prompts",
    "position_ids",
    "ref_log_prob",
    "rollout_is_weights",
)


def is_bagel_corl_composite(model_config) -> bool:
    if model_config is None:
        return False
    return model_config.get("composite_mode") == "bagel_corl"


def unwrap_bagel_module(module: torch.nn.Module) -> torch.nn.Module:
    """Peel FSDP / PEFT wrappers until ``compute_und_log_prob`` is reachable."""
    current = module
    for _ in range(8):
        if hasattr(current, "compute_und_log_prob"):
            return current
        nxt = getattr(current, "_fsdp_wrapped_module", None)
        if nxt is None:
            nxt = getattr(current, "module", None)
        if nxt is None:
            nxt = getattr(current, "base_model", None)
            if nxt is not None and hasattr(nxt, "model"):
                nxt = nxt.model
        if nxt is None or nxt is current:
            break
        current = nxt
    raise AttributeError(
        "Bagel Co-RL (Joint-Training) composite UND path requires BagelForCoRL.compute_und_log_prob; "
        f"got {type(module).__name__}"
    )


def register_und_forward_method(engine: Any, module: torch.nn.Module) -> None:
    """Make FSDP2 treat ``compute_und_log_prob`` like ``forward``."""
    try:
        from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
    except ImportError:  # pragma: no cover - torch without the FSDP2 registration API
        return

    # ``unwrap_bagel_module`` may have peeled a PEFT wrapper; registration has to land on
    # the object FSDP2 actually manages, so prefer the engine's own module.
    for candidate in (getattr(engine, "module", None), module):
        if candidate is None or not isinstance(candidate, FSDPModule):
            continue
        if not hasattr(candidate, "compute_und_log_prob"):
            continue
        if getattr(candidate, "_bagel_corl_und_forward_registered", False):
            return
        register_fsdp_forward_method(candidate, "compute_und_log_prob")
        candidate._bagel_corl_und_forward_registered = True
        return


def composite_forward_mode(data: TensorDict, *, forward_only: bool) -> str:
    """Select UND infer / GEN-only diffusion / composite train / empty.

    GEN-only (``all_latents``, no ``input_ids``) is the diffusion V1
    ``infer_actor_batch`` old-logprob path and must use the vanilla timestep loop.
    """
    has_und = "input_ids" in data.keys()
    has_latents = "all_latents" in data.keys() and ("all_timesteps" in data.keys() or "timesteps" in data.keys())
    gen_view = tu.get_non_tensor_data(data, "bagel_corl_gen", default=None)
    has_complete = bool(tu.get_non_tensor_data(data, "has_complete_gen_groups", default=False))
    skip_gen = bool(tu.get_non_tensor_data(data, "skip_gen", default=not has_complete))
    run_gen_view = (not skip_gen) and has_complete and gen_view_has_traj(gen_view)

    if has_latents and not has_und:
        return "gen_only_diffusion"
    if forward_only and has_und and not run_gen_view:
        return "und_infer"
    if has_und or run_gen_view:
        return "composite_train"
    return "empty"


def gen_view_has_traj(gen_view) -> bool:
    if gen_view is None:
        return False
    keys = getattr(gen_view, "keys", None)
    if keys is None:
        return False
    keyset = set(keys()) if callable(keys) else set(keys)
    return "all_latents" in keyset and ("all_timesteps" in keyset or "timesteps" in keyset)


def materialize_gen_train_batch(gen_view: TensorDict, flags: dict[str, Any]) -> TensorDict:
    """Build a diffusion train TensorDict from ``bagel_corl_gen`` (FlowGRPO + traj)."""
    if isinstance(gen_view, TensorDict):
        gen_data = gen_view.clone()
    else:
        gen_data = TensorDict(dict(gen_view), batch_size=getattr(gen_view, "batch_size", []))

    if "all_timesteps" not in gen_data.keys() and "timesteps" in gen_data.keys():
        gen_data["all_timesteps"] = gen_data["timesteps"]

    for key, val in flags.items():
        if val is not None:
            tu.assign_non_tensor(gen_data, **{key: val})
    # After ingress all GEN fields are row-aligned tensors, sliced by normal microbatching.
    # Reattaching the full view here silently broadcasts one row's forward over all rewards.
    gen_data.pop("bagel_corl_gen", None)
    tu.assign_non_tensor(gen_data, bagel_corl_lane="gen")
    tu.assign_non_tensor(gen_data, skip_gen=False)
    tu.assign_non_tensor(gen_data, has_complete_gen_groups=True)
    return gen_data


def _pad_left_response_mask(response_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pad response-only mask on the left to full sequence length (prompt | response)."""
    if response_mask.shape[-1] == seq_len:
        return response_mask
    if response_mask.shape[-1] > seq_len:
        raise ValueError(f"response_mask length {response_mask.shape[-1]} > seq_len {seq_len}")
    pad = seq_len - response_mask.shape[-1]
    return F.pad(response_mask, (pad, 0))


def und_jagged_rows(values: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    """Pad a ``(B, L-1)`` UND grid to ``prompt + response`` rows for the trainer's nested contract."""
    lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()
    rows: list[torch.Tensor] = []
    for i, length in enumerate(lengths):
        if length < 1:
            raise ValueError("Bagel Co-RL UND pass: attention_mask reports an empty sequence")
        # ``values`` was built on the padded width; a valid row always exceeds its response, since a
        # scored row has at least one prompt token (``response_len == L`` would make
        # ``response_from_nested`` reach back into the previous row).
        rows.append(torch.cat([values[i, : length - 1], values.new_zeros(1)]))
    return rows


def postprocess_und_batch(
    output_lst: list[dict],
    indices: Any,
    data: TensorDict,
    *,
    forward_only: bool,
) -> dict | TensorDict:
    """Merge UND micro-batch blobs *without* the diffusion postprocessor's step dimension."""
    if tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=False):
        raise NotImplementedError(
            "Bagel Co-RL UND postprocess cannot reorder micro-batches for dynamic bsz; the UND "
            "pass pins use_dynamic_bsz=False. Drop the override or reorder before publishing."
        )

    losses: list = []
    aggregated_metrics: dict = {}
    nested_log_probs: list[torch.Tensor] = []
    nested_entropy: list[torch.Tensor] = []
    for output in output_lst:
        losses.extend(output.get("loss") or [])
        for metrics in output.get("metrics") or []:
            append_to_dict(aggregated_metrics, metrics)
        nested = output.get("und_nested") or {}
        nested_log_probs.extend(nested.get("log_probs") or [])
        nested_entropy.extend(nested.get("entropy") or [])

    if not forward_only:
        return {"model_output": {}, "loss": losses, "metrics": aggregated_metrics}

    if not nested_log_probs:
        raise RuntimeError(
            "Bagel Co-RL UND infer pass produced no per-row log-probs; nothing to publish to the "
            "TransferQueue for the trainer's old_log_probs."
        )
    # Same outer shape every other engine's postprocess returns -- ``_postprocess_output`` pops ``metrics`` and
    # ``loss`` and then folds the leftover ``model_output`` into the published TensorDict
    # (``engine_workers.py:243,249,298``, reaching ``tu.get_tensordict`` at :300), and ``infer_actor_batch`` calls
    # ``.cpu()`` on that result.
    return {
        "model_output": {
            "log_probs": torch.nested.as_nested_tensor(nested_log_probs, layout=torch.jagged),
            "entropy": torch.nested.as_nested_tensor(nested_entropy, layout=torch.jagged),
        },
        "loss": losses,
        "metrics": aggregated_metrics,
    }


def run_und_token_forward_backward(
    engine,
    data: TensorDict,
    loss_function: Callable,
    forward_only: bool,
) -> dict:
    """AR-style UND pass on Bagel MoT text path; one backward per micro-batch when training."""
    from contextlib import nullcontext

    und = data.select(*[k for k in _UND_SELECT if k in data.keys()], strict=False)
    if "input_ids" not in und.keys():
        raise KeyError("Bagel Co-RL (Joint-Training) UND phase requires input_ids on the actor batch")
    if not forward_only and ("old_log_probs" not in und.keys() or "advantages" not in und.keys()):
        raise KeyError(
            "Bagel Co-RL (Joint-Training) UND train phase requires old_log_probs and advantages (token GRPO)"
        )

    tu.assign_non_tensor(und, sp_size=engine.ulysses_sequence_parallel_size)
    tu.assign_non_tensor(und, use_dynamic_bsz=False)

    # ``data.select`` above drops non-tensor metadata, but on the ``use_dynamic_bsz=False`` branch
    # ``prepare_micro_batches`` reads ``micro_batch_size_per_gpu`` straight off the batch
    # (verl/verl/workers/engine/utils.py:89).
    micro_batch_size_per_gpu = tu.get_non_tensor_data(data=data, key="micro_batch_size_per_gpu", default=None)
    if micro_batch_size_per_gpu is None:
        config_key = "infer_micro_batch_size_per_gpu" if forward_only else "micro_batch_size_per_gpu"
        micro_batch_size_per_gpu = getattr(getattr(engine, "engine_config", None), config_key, None)
    if micro_batch_size_per_gpu is None:
        raise KeyError(
            "Bagel Co-RL (Joint-Training) UND pass requires micro_batch_size_per_gpu: this composite "
            "forces use_dynamic_bsz=False, so prepare_micro_batches reads it from the batch. Set "
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu for the train pass, and "
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu (or "
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu) for the infer pass."
        )
    tu.assign_non_tensor(und, micro_batch_size_per_gpu=int(micro_batch_size_per_gpu))

    und_episode_normalizer = None
    if not forward_only:
        masks = list(und["response_mask"].unbind())
        ids = und.get("bagel_episode_id", None)
        if ids is not None:
            masks = [mask * (int(ids[i]) >= 0) for i, mask in enumerate(masks)]
            und["response_mask"] = (
                torch.nested.as_nested_tensor(masks, layout=torch.jagged)
                if und["response_mask"].is_nested
                else torch.stack(masks)
            )
        count = torch.tensor(float(sum(bool(mask.any()) for mask in masks)), device=get_device_id())
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        dp_size = torch.distributed.get_world_size(engine.get_data_parallel_group()) if distributed else 1
        if distributed:
            torch.distributed.all_reduce(count, group=engine.get_data_parallel_group())
        und_episode_normalizer = max(float(count.item()), 1.0) / dp_size
    micro_batches, indices = prepare_micro_batches(
        data=und, dp_group=engine.get_data_parallel_group(), same_micro_num_in_dp=True
    )
    gradient_accumulation_steps = max(len(micro_batches), 1)
    # Train mode drops ``model_output`` after the per-micro-batch token loss has run (same policy as
    # the transformer engine); only the infer path keeps it, since that is what its postprocessor
    # turns into the published ``log_probs``/``entropy``.
    return_model_output = forward_only or bool(
        tu.get_non_tensor_data(data=data, key="return_model_output", default=False)
    )
    output_lst = []
    ctx = torch.no_grad() if forward_only else nullcontext()
    module = unwrap_bagel_module(engine.module)
    # FSDP2 has to know that ``compute_und_log_prob`` is a forward-shaped entry point,
    # otherwise it runs against sharded parameters with plain activations and raises
    # ``aten.mul.Tensor got mixed torch.Tensor and DTensor`` inside ``RMSNorm.forward``.
    register_und_forward_method(engine, module)

    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
        tu.assign_non_tensor(micro_batch, skip_gen=True)
        tu.assign_non_tensor(micro_batch, has_complete_gen_groups=False)
        tu.assign_non_tensor(micro_batch, num_gen_rows=0)

        padded = micro_batch
        pad_keys = [
            k
            for k in (
                "input_ids",
                "attention_mask",
                "response_mask",
                "old_log_probs",
                "advantages",
                "ref_log_prob",
                "rollout_is_weights",
            )
            if k in micro_batch.keys()
        ]
        if hasattr(micro_batch, "to_padded_tensor"):
            padded = micro_batch.select(*pad_keys).to_padded_tensor()

        input_ids = padded["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = padded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        response_mask = padded["response_mask"]
        if response_mask.ndim == 1:
            response_mask = response_mask.unsqueeze(0)
        full_response_mask = _pad_left_response_mask(response_mask, input_ids.shape[1])
        if "old_log_probs" in padded.keys():
            resp_len = int(padded["old_log_probs"].shape[-1])
        else:
            resp_len = int(response_mask.shape[-1])

        with ctx:
            # Entropy is only needed on the infer pass: ``_compute_old_log_prob`` reads it off the
            # TransferQueue alongside ``log_probs`` (``trainer_base.py:1506,1513``). The train pass
            # deliberately leaves it out so ``ppo_loss`` keeps skipping the entropy term, and so
            # the log-prob grid is not duplicated on the hot path.
            want_entropy = forward_only or bool(tu.get_non_tensor_data(data, "calculate_entropy", default=False))
            if want_entropy:
                token_logp, token_entropy = module.compute_und_log_prob(
                    input_ids, attention_mask, full_response_mask, with_entropy=True
                )
                und_rows = und_jagged_rows(token_logp, attention_mask)
                und_nested = {
                    "log_probs": und_rows,
                    "entropy": und_jagged_rows(token_entropy, attention_mask),
                }
            else:
                token_logp = module.compute_und_log_prob(input_ids, attention_mask, full_response_mask)
                und_rows = und_jagged_rows(token_logp, attention_mask)
                und_nested = None
            # ``model_output["log_probs"]`` must be the *full-sequence* grid, not the response-only slice:
            # ``ppo_loss`` re-derives the response slice itself by calling ``no_padding_2_padding`` first
            # (``verl/workers/utils/losses.py:59`` -> ``workers/utils/padding.py:99``), which walks ``values[offset -
            # resp_len - 1 : offset - 1]`` per row and therefore needs each row to span ``prompt + response``.
            und_grid = torch.nested.as_nested_tensor(und_rows, layout=torch.jagged)
            model_output = {"und": {"log_probs": und_grid}, "modality": "und", "log_probs": und_grid}
            if want_entropy and not forward_only:
                model_output["und"]["entropy"] = torch.nested.as_nested_tensor(
                    und_jagged_rows(token_entropy, attention_mask), layout=torch.jagged
                )
                und_nested = None

            # Infer (old log-prob): no token-GRPO fields yet — return log_probs only.
            infer_only = forward_only and ("old_log_probs" not in padded.keys() or "advantages" not in padded.keys())
            if infer_only or loss_function is None:
                loss = torch.tensor(1.0, device=get_device_id())
                metrics = {}
            else:
                resp_mask_loss = response_mask if response_mask.shape[-1] == resp_len else response_mask[:, -resp_len:]
                loss_data = TensorDict(
                    {
                        "response_mask": resp_mask_loss,
                        "old_log_probs": padded["old_log_probs"],
                        "advantages": padded["advantages"],
                    },
                    batch_size=padded["old_log_probs"].shape[:1],
                )
                # ``no_padding_2_padding`` also reads the per-row ``prompts``/``responses`` to locate each response
                # inside the grid.
                for key in ("prompts", "responses"):
                    if key not in micro_batch.keys():
                        raise KeyError(
                            f"Bagel Co-RL UND train loss needs '{key}' on the micro-batch: ppo_loss "
                            "locates each response inside the full-sequence log-prob grid with it "
                            "(workers/utils/padding.py:127). Check the UND row's TQ fields."
                        )
                    value = micro_batch[key]
                    # A padded 2-D pair is only read correctly when every row shares one prompt length -- the padded
                    # branch takes ``attention_mask[:, :prompt_ids.shape[1]]`` (``padding.py:130``) as the prompt, so
                    # on a batch that mixes prompt lengths it counts response tokens as prompt and scores the wrong
                    # entries.
                    if value.is_nested is False and micro_batch.batch_size[0] > 1:
                        raise ValueError(
                            f"Bagel Co-RL UND train loss: '{key}' arrived padded "
                            f"(shape {tuple(value.shape)}) for a {micro_batch.batch_size[0]}-row "
                            "micro-batch. ppo_loss can only locate responses from jagged per-row "
                            "lengths; a padded pair would score the wrong tokens whenever the rows "
                            "mix prompt lengths. Keep the UND row's 'prompts'/'responses' nested."
                        )
                    loss_data[key] = value
                # Pin the grid's right-pad target to the width the rest of ``ppo_loss`` pads
                # ``response_mask``/``old_log_probs`` to (``padding.py:116,119``). Left at its
                # default it is recomputed as ``max(response_lens)``, which must equal ``resp_len``
                # anyway -- pinning turns a width mismatch into an obvious error instead of a
                # broadcast failure further down.
                tu.assign_non_tensor(loss_data, max_response_len=resp_len)
                for opt in ("ref_log_prob", "rollout_is_weights"):
                    if opt in padded.keys():
                        loss_data[opt] = padded[opt]
                # Loss units (RFC §4.4 + §4.10): "both branches normalize by their own micro-batch counts ... each
                # branch is a mean over its own loss units; relative scale is governed by the lane weights".
                tu.assign_non_tensor(loss_data, dp_size=1)
                tu.assign_non_tensor(loss_data, batch_num_tokens=None)
                tu.assign_non_tensor(loss_data, global_batch_size=None)

                tu.assign_non_tensor(loss_data, und_episode_normalizer=und_episode_normalizer, corl_dp_size=dp_size)
                tu.assign_non_tensor(loss_data, bagel_corl_und=loss_data)
                tu.assign_non_tensor(loss_data, skip_gen=True)
                tu.assign_non_tensor(loss_data, has_complete_gen_groups=False)
                tu.assign_non_tensor(loss_data, num_gen_rows=0)
                tu.assign_non_tensor(
                    loss_data,
                    gradient_accumulation_steps=tu.get_non_tensor_data(
                        micro_batch, "gradient_accumulation_steps", default=gradient_accumulation_steps
                    ),
                    sp_size=tu.get_non_tensor_data(micro_batch, "sp_size", default=1),
                )
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=loss_data,
                    dp_group=engine.get_data_parallel_group(),
                )

            if not forward_only and not engine.model_config.get("corl_freeze_und", False):
                loss.backward()

        # ``model_output`` only survives on the lanes that need it: the transformer engine drops it
        # for plain training (``transformer_impl.py:710``), and the diffusion postprocessor has no
        # equivalent pop, so carrying the nested UND blob through train-mode postprocessing is what
        # fed a dict to ``torch.stack``. On infer the payload is not ``model_output`` at all but the
        # jagged rows the trainer's ``response_from_nested`` consumes.
        meta_info_lst = {
            "model_output": [model_output] if return_model_output else [],
            "loss": [loss.detach().item()],
            "metrics": [metrics],
        }
        if und_nested is not None:
            meta_info_lst["und_nested"] = und_nested
        output_lst.append(meta_info_lst)

    return postprocess_und_batch(output_lst=output_lst, indices=indices, data=und, forward_only=forward_only)


def _flatten_losses(value) -> list[float]:
    """Flatten one lane's ``loss`` blob into a flat list of scalars."""
    flat: list[float] = []
    for item in value if isinstance(value, list | tuple) else [value]:
        if isinstance(item, list | tuple):
            flat.extend(_flatten_losses(item))
        elif item is not None:
            flat.append(float(item))
    return flat


def merge_composite_outputs(parts: list[dict]) -> dict:
    """Merge UND + GEN ``postprocess_batch_func`` dicts for one train_batch metrics blob.

    A single part is returned untouched, which is what carries the UND infer pass's published
    ``log_probs``/``entropy`` up to ``infer_actor_batch``. Merging is only valid for the two
    train-mode dicts, because the merge keeps just ``parts[-1]``'s ``model_output``.
    """
    if not parts:
        return {"model_output": {}, "loss": [], "metrics": {}}
    if len(parts) == 1:
        return parts[0]
    shadowed = [part.get("model_output") for part in parts[:-1] if part.get("model_output")]
    if shadowed:
        raise RuntimeError(
            "Bagel Co-RL: a non-final lane produced model_output "
            f"({[sorted(blob) for blob in shadowed]}) that merge_composite_outputs would drop by "
            "keeping only the last part's. Refusing rather than silently losing log-probs."
        )

    merged_loss: list = []
    merged_metrics: dict = {}
    for part in parts:
        # Per-lane ``loss`` shapes differ (see ``_flatten_losses``); normalize before merging or the
        # worker's ``torch.tensor(...)`` reduction dies on "must be real number, not list".
        merged_loss.extend(_flatten_losses(part.get("loss") or []))
        # Metrics must go through ``append_to_dict``, which is ``Metric``-aware: the UND part publishes ``Metric``
        # objects and the GEN part (via ``diffusion_loss`` → ``Metric.from_dict``) does too, but a hand-rolled merge
        # wraps each lane's single ``Metric`` in a **list**, so the key ends up as ``[Metric, Metric]`` and the
        # driver's legacy reducer does ``np.mean`` over it (``TypeError: unsupported operand type(s) for +: 'Metric'
        # and 'Metric'`` in ``reduce_metrics``).
        append_to_dict(merged_metrics, part.get("metrics") or {})
    return {
        "model_output": parts[-1].get("model_output") or {},
        "loss": merged_loss,
        "metrics": merged_metrics,
    }
