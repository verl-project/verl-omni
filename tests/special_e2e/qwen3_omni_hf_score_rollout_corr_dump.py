#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _emit(record: dict, output_file: str | None) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _finite_pairs(left: list[float], right: list[float], mask: list[int]) -> list[tuple[float, float]]:
    pairs = []
    for a, b, keep in zip(left, right, mask):
        if not keep:
            continue
        a = float(a)
        b = float(b)
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    return pairs


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = torch.tensor([x for x, _ in pairs], dtype=torch.float32)
    right = torch.tensor([y for _, y in pairs], dtype=torch.float32)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1].item())


def _stats(values: list[float], mask: list[int]) -> dict:
    kept = [float(value) for value, keep in zip(values, mask) if keep and math.isfinite(float(value))]
    if not kept:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "zero_fraction": None,
            "near_zero_fraction": None,
        }
    return {
        "count": len(kept),
        "mean": sum(kept) / len(kept),
        "min": min(kept),
        "max": max(kept),
        "zero_fraction": sum(1 for x in kept if x == 0.0) / len(kept),
        "near_zero_fraction": sum(1 for x in kept if abs(x) < 1e-6) / len(kept),
    }


def _compare(name: str, hf: list[float], other: list[float], mask: list[int]) -> dict:
    pairs = _finite_pairs(hf, other, mask)
    if not pairs:
        return {
            f"hf_vs_{name}/paired_count": 0,
            f"hf_vs_{name}/abs_diff_mean": None,
            f"hf_vs_{name}/abs_diff_max": None,
            f"hf_vs_{name}/signed_diff_mean": None,
            f"hf_vs_{name}/mult_prob_error_mean": None,
            f"hf_vs_{name}/mult_prob_error_max": None,
            f"hf_vs_{name}/corr": None,
        }
    diffs = [a - b for a, b in pairs]
    abs_diffs = [abs(x) for x in diffs]
    mult_errors = [math.exp(min(x, 80.0)) for x in abs_diffs]
    return {
        f"hf_vs_{name}/paired_count": len(pairs),
        f"hf_vs_{name}/abs_diff_mean": sum(abs_diffs) / len(abs_diffs),
        f"hf_vs_{name}/abs_diff_max": max(abs_diffs),
        f"hf_vs_{name}/signed_diff_mean": sum(diffs) / len(diffs),
        f"hf_vs_{name}/mult_prob_error_mean": sum(mult_errors) / len(mult_errors),
        f"hf_vs_{name}/mult_prob_error_max": max(mult_errors),
        f"hf_vs_{name}/corr": _corr(pairs),
    }


def _load_records(path: str, limit: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "rollout_corr_sample":
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No rollout_corr_sample records found in {path}")
    return records


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    """Return a cross-process identifier for one unpadded token sequence."""
    values = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _vector_row_stats(rows: torch.Tensor | None) -> list[dict] | None:
    if rows is None or rows.ndim != 2:
        return None
    rows = rows.detach().float()
    head_width = min(8, rows.shape[-1])
    return [
        {
            "sum": float(row.sum().cpu().item()),
            "square_sum": float((row * row).sum().cpu().item()),
            "head": [float(value) for value in row[:head_width].cpu().tolist()],
        }
        for row in rows
    ]


def _tensor_fingerprint(value: torch.Tensor | None) -> dict | None:
    if value is None or value.ndim != 2:
        return None
    tensor = value.detach().float()
    height, width = tensor.shape
    anchors = sorted(
        {
            (0, 0),
            (0, min(1, width - 1)),
            (min(1, height - 1), 0),
            (height // 2, width // 2),
            (height - 1, width - 1),
        }
    )
    return {
        "shape": [int(height), int(width)],
        "sum": float(tensor.sum().cpu().item()),
        "square_sum": float((tensor * tensor).sum().cpu().item()),
        "anchors": [
            {"index": [row, column], "value": float(tensor[row, column].cpu().item())}
            for row, column in anchors
        ],
    }


def _parse_layer_set(raw: str) -> set[int]:
    return {int(value) for value in raw.split(",") if value.strip().isdigit()}


def _moe_router_rows(
    logits: torch.Tensor,
    router_input: torch.Tensor | None,
    layer: int,
    positions: list[int],
    input_ids: torch.Tensor,
    top_k: int,
) -> list[dict]:
    # Qwen3-Omni's sparse block flattens [B, S, H] before invoking `gate`,
    # while the standard HF route is [B, S, E]. Both layouts are equivalent
    # for this single-sequence scorer.
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or logits.shape[0] != 1:
        return []
    weights = torch.softmax(logits.float(), dim=-1)
    weights, expert_ids = torch.topk(weights, min(top_k, weights.shape[-1]), dim=-1)
    if router_input is not None and router_input.ndim == 2:
        router_input = router_input.unsqueeze(0)
    if router_input is not None and (router_input.ndim != 3 or router_input.shape[0] != 1):
        router_input = None
    rows = []
    for response_index, model_position in enumerate(positions):
        if not 0 <= model_position < weights.shape[1]:
            continue
        raw_logits = logits[0, model_position].detach().float()
        raw_top_values, raw_top_ids = torch.topk(raw_logits, min(16, raw_logits.numel()))
        raw_topk_size = min(top_k, raw_logits.numel())
        raw_topk_margin = None
        if 0 < raw_topk_size < raw_logits.numel():
            raw_boundary = torch.topk(raw_logits, raw_topk_size + 1).values
            raw_topk_margin = float(raw_boundary[raw_topk_size - 1] - raw_boundary[raw_topk_size])
        input_stats = None
        if router_input is not None and model_position < router_input.shape[1]:
            input_stats = _vector_row_stats(router_input[0, model_position].unsqueeze(0))[0]
        rows.append(
            {
                "layer": layer,
                "response_index": response_index,
                "model_position": model_position,
                "input_token_id": int(input_ids[0, model_position].item()),
                "expert_ids": [int(value) for value in expert_ids[0, model_position].detach().cpu().tolist()],
                "expert_probs": [float(value) for value in weights[0, model_position].detach().cpu().tolist()],
                "router_input_stats": input_stats,
                "router_logits_stats": _vector_row_stats(raw_logits.unsqueeze(0))[0],
                "router_logit_top_expert_ids": [int(value) for value in raw_top_ids.cpu().tolist()],
                "router_logit_top_values": [float(value) for value in raw_top_values.cpu().tolist()],
                "router_raw_topk_size": raw_topk_size,
                "router_raw_topk_margin": raw_topk_margin,
            }
        )
    return rows


def _first_tensor_output(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor_output(item)
            if tensor is not None:
                return tensor
    return None


def _decoder_layers(model) -> list:
    for path in ("thinker.model.layers", "model.layers", "layers"):
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
        except AttributeError:
            continue
        return list(module)
    return []


def _decoder_component_rows(
    value, layer: int, component: str, positions: list[int], input_ids: torch.Tensor
) -> list[dict]:
    tensor = _first_tensor_output(value)
    if tensor is None or tensor.ndim != 3 or tensor.shape[0] != 1:
        return []
    if not positions or max(positions) >= tensor.shape[1]:
        return []
    stats = _vector_row_stats(tensor[0, positions, :])
    if stats is None:
        return []
    return [
        {
            "layer": layer,
            "component": component,
            "response_index": response_index,
            "model_position": position,
            "input_token_id": int(input_ids[0, position].item()),
            "stats": stat,
        }
        for response_index, (position, stat) in enumerate(zip(positions, stats, strict=True))
    ]


def _attention_stage_rows(
    tensor: torch.Tensor,
    *,
    layer: int,
    stage: str,
    positions: list[int],
    input_ids: torch.Tensor,
    tp_rank: int,
) -> list[dict]:
    if tensor.ndim < 3 or tensor.shape[0] != 1 or not positions or max(positions) >= tensor.shape[1]:
        return []
    vectors = tensor[0, positions].reshape(len(positions), -1)
    stats = _vector_row_stats(vectors)
    if stats is None:
        return []
    return [
        {
            "layer": layer,
            "stage": stage,
            "tp_rank": tp_rank,
            "response_index": response_index,
            "model_position": position,
            "input_token_id": int(input_ids[0, position].item()),
            "shape": list(tensor.shape),
            "stats": stat,
        }
        for response_index, (position, stat) in enumerate(zip(positions, stats, strict=True))
    ]


def _local_causal_attention_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Reference GQA causal attention in the same BSHD layout used by HF hooks."""
    if query.shape[2] != key.shape[2]:
        repeat = query.shape[2] // key.shape[2]
        key = key.repeat_interleave(repeat, dim=2)
        value = value.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bshd,bthd->bhst", query.float(), key.float()) * (query.shape[-1] ** -0.5)
    causal_mask = torch.ones((query.shape[1], key.shape[1]), dtype=torch.bool, device=query.device).triu(1)
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("bhst,bthd->bshd", probabilities, value).reshape(query.shape[0], query.shape[1], -1)


def _tp_head_slice(value: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Select the contiguous HF head shard corresponding to a Megatron TP rank."""
    if value.ndim != 4 or value.shape[2] % tp_size:
        raise ValueError(f"Cannot split attention heads {tuple(value.shape)} across TP={tp_size}")
    width = value.shape[2] // tp_size
    return value[:, :, tp_rank * width : (tp_rank + 1) * width, :]


def _tp_hidden_slice(value: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[-1] % tp_size:
        raise ValueError(f"Cannot split hidden states {tuple(value.shape)} across TP={tp_size}")
    width = value.shape[-1] // tp_size
    return value[:, :, tp_rank * width : (tp_rank + 1) * width]


def _output_embedding_weight(model) -> torch.Tensor | None:
    output_embedding = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    weight = getattr(output_embedding, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    for path in ("lm_head", "thinker.lm_head", "model.lm_head"):
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
        except AttributeError:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
    return None


def _output_embedding_bias(model) -> torch.Tensor | None:
    output_embedding = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    bias = getattr(output_embedding, "bias", None)
    return bias if isinstance(bias, torch.Tensor) else None


def _input_embedding_weight(model) -> torch.Tensor | None:
    input_embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    weight = getattr(input_embedding, "weight", None)
    return weight if isinstance(weight, torch.Tensor) else None


def _final_norm(model):
    for path in ("thinker.model.norm", "model.norm", "thinker.norm"):
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
        except AttributeError:
            continue
        if isinstance(getattr(module, "weight", None), torch.Tensor):
            return module
    return None


def _final_norm_weight(model) -> torch.Tensor | None:
    module = _final_norm(model)
    return getattr(module, "weight", None) if module is not None else None


def _last_hidden_state(output) -> torch.Tensor | None:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF-score fixed rollout-corr samples dumped from verl.")
    parser.add_argument("--dump-jsonl", required=True)
    parser.add_argument(
        "--model-path",
        default="/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-chattemplate",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--position-ids-mode", choices=["none", "dump"], default="none")
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=None,
        help="Divide logits by this temperature before scoring; defaults to the dumped rollout temperature.",
    )
    parser.add_argument("--record-limit", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--attention-audit-tp-size", type=int, default=2)
    parser.add_argument("--moe-router-audit-layers", default="1,4,6,16,32,48")
    parser.add_argument("--output-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text("", encoding="utf-8")

    import verl_omni.models.transformers.qwen3_omni_thinker  # noqa: F401

    dtype = args.dtype
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float32":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    output_embedding_weight = _output_embedding_weight(model)
    output_embedding_bias = _output_embedding_bias(model)
    input_embedding_weight = _input_embedding_weight(model)
    final_norm_weight = _final_norm_weight(model)
    final_norm = _final_norm(model)
    hf_capture: dict[str, torch.Tensor] = {}
    final_norm_hook = None
    if final_norm is not None:
        def _capture_pre_final_norm(_module, args, kwargs=None):
            hidden = args[0] if args and isinstance(args[0], torch.Tensor) else None
            if hidden is None:
                for name in ("hidden_states", "input_"):
                    candidate = (kwargs or {}).get(name)
                    if isinstance(candidate, torch.Tensor):
                        hidden = candidate
                        break
            if hidden is not None:
                hf_capture["pre_final_norm_hidden"] = hidden.detach()

        final_norm_hook = final_norm.register_forward_pre_hook(_capture_pre_final_norm, with_kwargs=True)

    decoder_capture: list[dict] = []
    attention_stage_capture: list[dict] = []
    moe_router_capture: list[dict] = []
    moe_router_weight_audit: list[dict] = []
    attention_state: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
    decoder_positions: list[int] = []
    decoder_hooks = []
    moe_router_layers = _parse_layer_set(args.moe_router_audit_layers)

    def _capture_decoder_component(layer: int, component: str, *, pre_hook: bool = False):
        def _capture(value):
            rows = _decoder_component_rows(
                value,
                layer,
                component,
                decoder_positions,
                input_ids,
            )
            if rows:
                decoder_capture.extend(rows)

        if pre_hook:
            def _pre_hook(_module, args):
                _capture(args)

            return _pre_hook

        def _forward_hook(_module, _args, output):
            _capture(output)

        return _forward_hook

    for layer_index, decoder_layer in enumerate(_decoder_layers(model), start=1):
        decoder_hooks.append(
            decoder_layer.register_forward_pre_hook(
                _capture_decoder_component(layer_index, "layer_input", pre_hook=True)
            )
        )
        decoder_hooks.append(decoder_layer.register_forward_hook(_capture_decoder_component(layer_index, "layer")))
        self_attention = getattr(decoder_layer, "self_attn", None)
        if self_attention is not None:
            decoder_hooks.append(
                self_attention.register_forward_hook(_capture_decoder_component(layer_index, "self_attention"))
            )
        post_attention_layernorm = getattr(decoder_layer, "post_attention_layernorm", None)
        if post_attention_layernorm is not None:
            decoder_hooks.append(
                post_attention_layernorm.register_forward_pre_hook(
                    _capture_decoder_component(layer_index, "post_attention_residual", pre_hook=True)
                )
            )
            decoder_hooks.append(
                post_attention_layernorm.register_forward_hook(
                    _capture_decoder_component(layer_index, "post_attention_norm")
                )
            )
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is not None:
            decoder_hooks.append(mlp.register_forward_hook(_capture_decoder_component(layer_index, "mlp")))
            gate = getattr(mlp, "gate", None)
            if gate is not None and layer_index in moe_router_layers:
                gate_input = {"value": None}

                def _capture_moe_router_input(_module, args, *, state=gate_input):
                    state["value"] = _first_tensor_output(args)

                def _capture_moe_router(
                    _module, _args, output, *, layer=layer_index, moe_module=mlp, state=gate_input
                ):
                    logits = _first_tensor_output(output)
                    if logits is not None:
                        moe_router_capture.extend(
                            _moe_router_rows(
                                logits,
                                state["value"],
                                layer,
                                decoder_positions,
                                input_ids,
                                int(getattr(moe_module, "top_k", 1)),
                            )
                        )

                fingerprint = _tensor_fingerprint(getattr(gate, "weight", None))
                if fingerprint is not None:
                    moe_router_weight_audit.append({"layer": layer_index, **fingerprint})
                decoder_hooks.append(gate.register_forward_pre_hook(_capture_moe_router_input))
                decoder_hooks.append(gate.register_forward_hook(_capture_moe_router))
        if layer_index != 1 or self_attention is None:
            continue

        def _capture_attention_tensor(name: str):
            def _hook(_module, _args, output):
                tensor = _first_tensor_output(output)
                if tensor is not None:
                    attention_state[name] = tensor.detach()

            return _hook

        def _capture_attention_inputs(_module, _args, kwargs):
            positions = kwargs.get("position_embeddings")
            if isinstance(positions, tuple) and len(positions) == 2 and all(
                isinstance(value, torch.Tensor) for value in positions
            ):
                attention_state["position_embeddings"] = (positions[0].detach(), positions[1].detach())

        decoder_hooks.append(self_attention.register_forward_pre_hook(_capture_attention_inputs, with_kwargs=True))
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
            module = getattr(self_attention, name, None)
            if module is None:
                continue
            if name == "o_proj":
                def _capture_o_proj_input(_module, args):
                    tensor = _first_tensor_output(args)
                    if tensor is not None:
                        attention_state["attention_context"] = tensor.detach()

                decoder_hooks.append(module.register_forward_pre_hook(_capture_o_proj_input))
                decoder_hooks.append(module.register_forward_hook(_capture_attention_tensor("attention_projection_output")))
            else:
                decoder_hooks.append(module.register_forward_hook(_capture_attention_tensor(name)))

    records = _load_records(args.dump_jsonl, args.record_limit)
    _emit(
        {
            "event": "hf_rollout_corr_score_start",
            "dump_jsonl": args.dump_jsonl,
            "model_path": args.model_path,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "position_ids_mode": args.position_ids_mode,
            "record_count": len(records),
        },
        args.output_file,
    )

    for record in records:
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([record["attention_mask"]], dtype=torch.long)
        position_ids = None
        if args.position_ids_mode == "dump":
            if "position_ids" not in record:
                raise ValueError(f"Record row={record.get('row')} has no position_ids in {args.dump_jsonl}")
            position_ids = torch.tensor([record["position_ids"]], dtype=torch.long)
        response_mask = [int(x) for x in record["response_mask"]]
        responses = [int(x) for x in record["responses"]]
        response_len = len(responses)
        response_start = input_ids.size(1) - response_len
        decoder_positions = list(range(response_start - 1, response_start - 1 + response_len))
        score_temperature = args.score_temperature
        if score_temperature is None:
            score_temperature = float(record.get("temperature", 1.0))
        if score_temperature <= 0:
            raise ValueError(f"row={record.get('row')} has non-positive score temperature: {score_temperature}")

        first_device = next(model.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        if position_ids is not None:
            position_ids = position_ids.to(first_device)

        with torch.inference_mode():
            hf_capture.clear()
            decoder_capture.clear()
            attention_stage_capture.clear()
            moe_router_capture.clear()
            attention_state.clear()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            logits = output.logits.float()
            target_ids = input_ids[0, response_start : response_start + response_len]
            pred_logits = logits[0, response_start - 1 : response_start - 1 + response_len, :]
            scaled_pred_logits = pred_logits / score_temperature
            hf_target_logits_tensor = scaled_pred_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            hf_logsumexp_tensor = torch.logsumexp(scaled_pred_logits, dim=-1)
            hf_logprobs_tensor = hf_target_logits_tensor - hf_logsumexp_tensor
            hf_hidden = _last_hidden_state(output)
            tp_size = args.attention_audit_tp_size
            q = attention_state.get("q_norm")
            k = attention_state.get("k_norm")
            v = attention_state.get("v_proj")
            context = attention_state.get("attention_context")
            projected = attention_state.get("attention_projection_output")
            for name, value, shard_heads in (
                ("q_post_qk_norm", q, True),
                ("k_post_qk_norm", k, True),
                ("v_pre_attention", v, False),
                ("attention_context", context, False),
            ):
                if not isinstance(value, torch.Tensor):
                    continue
                if name == "v_pre_attention":
                    if not isinstance(k, torch.Tensor) or value.shape[-1] % k.shape[-1]:
                        continue
                    value = value.view(*value.shape[:2], k.shape[2], k.shape[-1])
                    shard_heads = True
                for tp_rank in range(tp_size):
                    shard = _tp_head_slice(value, tp_rank, tp_size) if shard_heads else _tp_hidden_slice(value, tp_rank, tp_size)
                    attention_stage_capture.extend(
                        _attention_stage_rows(
                            shard,
                            layer=1,
                            stage=name,
                            positions=decoder_positions,
                            input_ids=input_ids,
                            tp_rank=tp_rank,
                        )
                    )
            if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                positions = attention_state.get("position_embeddings")
                if isinstance(positions, tuple):
                    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import apply_rotary_pos_emb

                    q_rope, k_rope = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *positions)
                    q_rope = q_rope.transpose(1, 2)
                    k_rope = k_rope.transpose(1, 2)
                    for name, value in (("q_post_rope", q_rope), ("k_post_rope", k_rope)):
                        for tp_rank in range(tp_size):
                            attention_stage_capture.extend(
                                _attention_stage_rows(
                                    _tp_head_slice(value, tp_rank, tp_size),
                                    layer=1,
                                    stage=name,
                                    positions=decoder_positions,
                                    input_ids=input_ids,
                                    tp_rank=tp_rank,
                                )
                            )
                    if isinstance(v, torch.Tensor) and v.shape[-1] % k_rope.shape[-1] == 0:
                        v_heads = v.view(*v.shape[:2], k_rope.shape[2], k_rope.shape[-1])
                        reference_context = _local_causal_attention_reference(q_rope, k_rope, v_heads)
                        for tp_rank in range(tp_size):
                            attention_stage_capture.extend(
                                _attention_stage_rows(
                                    _tp_hidden_slice(reference_context, tp_rank, tp_size),
                                    layer=1,
                                    stage="local_causal_reference_context",
                                    positions=decoder_positions,
                                    input_ids=input_ids,
                                    tp_rank=tp_rank,
                                )
                            )
            if isinstance(projected, torch.Tensor):
                for tp_rank in range(tp_size):
                    attention_stage_capture.extend(
                        _attention_stage_rows(
                            projected,
                            layer=1,
                            stage="attention_projection_output",
                            positions=decoder_positions,
                            input_ids=input_ids,
                            tp_rank=tp_rank,
                        )
                    )
            hf_pre_final_norm_hidden = hf_capture.get("pre_final_norm_hidden")
            hf_pre_lm_hidden_stats = _vector_row_stats(
                hf_hidden[0, response_start - 1 : response_start - 1 + response_len, :]
                if hf_hidden is not None
                else None
            )
            hf_pre_final_norm_hidden_stats = _vector_row_stats(
                hf_pre_final_norm_hidden[0, response_start - 1 : response_start - 1 + response_len, :]
                if hf_pre_final_norm_hidden is not None
                else None
            )
            hf_lm_head_weight_stats = _vector_row_stats(
                output_embedding_weight.index_select(0, target_ids.to(output_embedding_weight.device))
                if output_embedding_weight is not None
                else None
            )
            input_token_ids = input_ids[0, response_start - 1 : response_start - 1 + response_len]
            hf_input_embedding_stats = _vector_row_stats(
                input_embedding_weight.index_select(0, input_token_ids.to(input_embedding_weight.device))
                if input_embedding_weight is not None
                else None
            )
            hf_final_norm_weight_stats = _vector_row_stats(
                final_norm_weight.detach().float().unsqueeze(0) if final_norm_weight is not None else None
            )
            hf_manual_target_logits_tensor = None
            if hf_hidden is not None and output_embedding_weight is not None:
                weight_rows = output_embedding_weight.index_select(
                    0, target_ids.to(output_embedding_weight.device)
                ).float()
                # With device_map=auto, hidden states are returned on the input
                # device while lm_head may be placed on the final device.
                hidden_rows = hf_hidden[0, response_start - 1 : response_start - 1 + response_len, :].to(
                    weight_rows.device, dtype=torch.float32
                )
                hf_manual_target_logits_tensor = (hidden_rows * weight_rows).sum(dim=-1)
                if output_embedding_bias is not None:
                    hf_manual_target_logits_tensor = hf_manual_target_logits_tensor + output_embedding_bias.index_select(
                        0, target_ids.to(output_embedding_bias.device)
                    ).float()
                hf_manual_target_logits_tensor = hf_manual_target_logits_tensor / score_temperature

        hf_logprobs = [float(x) for x in hf_logprobs_tensor.detach().cpu().tolist()]
        hf_target_logits = [float(x) for x in hf_target_logits_tensor.detach().cpu().tolist()]
        hf_logsumexp = [float(x) for x in hf_logsumexp_tensor.detach().cpu().tolist()]
        unpadded_input_ids = input_ids[0, attention_mask[0].bool()]
        rollout = [float(x) for x in record["rollout_log_probs"]]
        actor_old = [float(x) for x in record["actor_old_log_probs"]]
        ref = [float(x) for x in record.get("ref_log_probs", [])]

        decoded = tokenizer.decode(
            [tok for tok, keep in zip(responses, response_mask) if keep][: args.sample_limit],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result = {
            "event": "hf_rollout_corr_score_row",
            "row": record["row"],
            "input_len": int(input_ids.size(1)),
            "response_len": response_len,
            "response_start": response_start,
            "score_temperature": score_temperature,
            "input_ids_sha256": _token_ids_sha256(unpadded_input_ids),
            "unpadded_input_len": int(unpadded_input_ids.numel()),
            "position_ids_mode": args.position_ids_mode,
            "valid_tokens": sum(response_mask),
            "decoded_valid_head": decoded,
            "hf_stats": _stats(hf_logprobs, response_mask),
            "rollout_stats": _stats(rollout, response_mask),
            "actor_old_stats": _stats(actor_old, response_mask),
            "hf_vs_rollout": _compare("rollout", hf_logprobs, rollout, response_mask),
            "hf_vs_actor_old": _compare("actor_old", hf_logprobs, actor_old, response_mask),
            "sample": [
                {
                    "i": i,
                    "model_position": response_start - 1 + i,
                    "token_id": responses[i],
                    "input_token_id": int(input_token_ids[i].detach().cpu().item()),
                    "mask": response_mask[i],
                    "hf_target_logit": round(hf_target_logits[i], 6),
                    "hf_logsumexp": round(hf_logsumexp[i], 6),
                    "hf": round(hf_logprobs[i], 6),
                    "rollout": round(rollout[i], 6),
                    "actor_old": round(actor_old[i], 6),
                    **({"ref": round(ref[i], 6)} if ref else {}),
                }
                for i in range(min(args.sample_limit, response_len))
            ],
            "components": [
                {
                    "response_index": i,
                    "model_position": response_start - 1 + i,
                    "token_id": responses[i],
                    "input_token_id": int(input_token_ids[i].detach().cpu().item()),
                    "hf_target_logit": hf_target_logits[i],
                    "hf_logsumexp": hf_logsumexp[i],
                    "hf_logprob": hf_logprobs[i],
                    "hf_manual_lm_head_target_logit": (
                        float(hf_manual_target_logits_tensor[i].detach().cpu().item())
                        if hf_manual_target_logits_tensor is not None
                        else None
                    ),
                    "hf_lm_head_weight": (
                        hf_lm_head_weight_stats[i] if hf_lm_head_weight_stats is not None else None
                    ),
                    "hf_input_embedding": (
                        hf_input_embedding_stats[i] if hf_input_embedding_stats is not None else None
                    ),
                    "hf_pre_lm_hidden": (
                        hf_pre_lm_hidden_stats[i] if hf_pre_lm_hidden_stats is not None else None
                    ),
                    "hf_pre_final_norm_hidden": (
                        hf_pre_final_norm_hidden_stats[i]
                        if hf_pre_final_norm_hidden_stats is not None
                        else None
                    ),
                    "hf_final_norm_weight": (
                        hf_final_norm_weight_stats[0] if hf_final_norm_weight_stats is not None else None
                    ),
                }
                for i in range(response_len)
                if response_mask[i]
            ],
            "decoder_component_audit": decoder_capture,
            "attention_stage_audit": attention_stage_capture,
            "moe_router_audit": moe_router_capture,
            "moe_router_weight_audit": moe_router_weight_audit,
        }
        if ref:
            result["ref_stats"] = _stats(ref, response_mask)
            result["hf_vs_ref"] = _compare("ref", hf_logprobs, ref, response_mask)
        _emit(result, args.output_file)

    if final_norm_hook is not None:
        final_norm_hook.remove()
    for decoder_hook in decoder_hooks:
        decoder_hook.remove()
    _emit({"event": "hf_rollout_corr_score_done"}, args.output_file)


if __name__ == "__main__":
    main()
