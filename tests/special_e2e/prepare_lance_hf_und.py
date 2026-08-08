#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Prepare a HF-loadable understanding-only checkpoint from Lance_3B MoT weights.

The HF Lance repo ships ``llm_config.json`` + MoT ``model.safetensors`` (with
``*_moe_gen`` dual paths) but no ``config.json``. Agentic FSDP training needs a
standard HF CausalLM layout for the understanding path.

This script:
  1. Remaps ``language_model.*`` und weights (drops ``*_moe_gen`` / connectors)
  2. Writes a Qwen2 ``config.json`` + remapped ``model.safetensors``
  3. Copies tokenizer files

Usage:
  python3 tests/special_e2e/prepare_lance_hf_und.py \\
    --src .../Lance_3B --dst .../Lance_3B_hf_und
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="Lance_3B directory")
    parser.add_argument("--dst", required=True, type=Path, help="Output HF und directory")
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    llm_cfg_path = src / "llm_config.json"
    if not llm_cfg_path.exists():
        raise FileNotFoundError(f"missing {llm_cfg_path}")

    llm_cfg = json.loads(llm_cfg_path.read_text())
    # Understanding path is Qwen2-compatible text LM (ViT lives separately).
    hf_cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "attention_dropout": llm_cfg.get("attention_dropout", 0.0),
        "bos_token_id": llm_cfg.get("bos_token_id", 151643),
        "eos_token_id": llm_cfg.get("eos_token_id", 151645),
        "hidden_act": llm_cfg.get("hidden_act", "silu"),
        "hidden_size": llm_cfg["hidden_size"],
        "initializer_range": llm_cfg.get("initializer_range", 0.02),
        "intermediate_size": llm_cfg["intermediate_size"],
        "max_position_embeddings": llm_cfg.get("max_position_embeddings", 32768),
        "max_window_layers": llm_cfg.get("max_window_layers", 28),
        "num_attention_heads": llm_cfg["num_attention_heads"],
        "num_hidden_layers": llm_cfg["num_hidden_layers"],
        "num_key_value_heads": llm_cfg["num_key_value_heads"],
        "rms_norm_eps": llm_cfg.get("rms_norm_eps", 1e-6),
        "rope_theta": llm_cfg.get("rope_theta", 1000000.0),
        "sliding_window": llm_cfg.get("sliding_window"),
        # Force untie: Lance und export already stores distinct embed + lm_head
        # tensors, and FSDP wrapping both modules while tied corrupts lm_head
        # (RuntimeError: size mismatch ... vec (~vocab*hidden/world_size)).
        "tie_word_embeddings": False,
        "torch_dtype": llm_cfg.get("torch_dtype", "bfloat16"),
        "use_cache": True,
        "vocab_size": llm_cfg["vocab_size"],
        # Keep q_norm / k_norm present in Lance und weights (Qwen2.5-style).
        "qk_norm": True,
    }
    (dst / "config.json").write_text(json.dumps(hf_cfg, indent=2) + "\n")

    weight_path = src / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"missing {weight_path}")

    state = load_file(str(weight_path))
    und = {}
    skipped = 0
    for k, v in state.items():
        if "moe_gen" in k:
            skipped += 1
            continue
        if k.startswith(("time_embedder", "llm2vae", "vae2llm", "latent_pos")):
            skipped += 1
            continue
        if not k.startswith("language_model."):
            skipped += 1
            continue
        und[k[len("language_model.") :]] = v

    save_file(und, str(dst / "model.safetensors"))
    print(f"wrote {len(und)} und tensors (skipped {skipped}) -> {dst / 'model.safetensors'}")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.json",
    ):
        src_f = src / name
        if src_f.exists():
            shutil.copy2(src_f, dst / name)

    # Tool-aware Qwen2.5 chat template (required so ToolAgentLoop's tools=schemas
    # are rendered and Hermes <tool_call> format is instructed). Same packaging as
    # upstream verl tests: a raw ``.jinja2`` file (see
    # tests/experimental/agent_loop/qwen_vl_tool_chat_template.jinja2).
    repo_root = Path(__file__).resolve().parents[2]
    tool_tmpl_path = repo_root / "tests/special_e2e/qwen2_tool_chat_template.jinja2"
    if not tool_tmpl_path.exists():
        raise FileNotFoundError(
            f"missing tool chat template at {tool_tmpl_path}; "
            "expected tests/special_e2e/qwen2_tool_chat_template.jinja2"
        )
    qwen2_chat_template = tool_tmpl_path.read_text()

    tok_cfg_path = dst / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
    else:
        tok_cfg = {
            "tokenizer_class": "Qwen2Tokenizer",
            "bos_token": None,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "unk_token": None,
            "model_max_length": hf_cfg["max_position_embeddings"],
        }
    tok_cfg.setdefault("eos_token", "<|im_end|>")
    tok_cfg.setdefault("pad_token", "<|endoftext|>")
    tok_cfg.setdefault("tokenizer_class", "Qwen2Tokenizer")
    tok_cfg["chat_template"] = qwen2_chat_template
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")
    (dst / "chat_template.jinja").write_text(qwen2_chat_template)

    print(f"Prepared HF und checkpoint at {dst}")
    print("architecture=Qwen2ForCausalLM")


if __name__ == "__main__":
    main()
