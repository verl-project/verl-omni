#!/usr/bin/env python3
"""Convert LTX-2.3 diffusers checkpoint to VeOmni-compatible format.

Two things need to happen:
1. config.json: add VeOmni field names (cross_attention_adaln, apply_gated_attention, etc.)
2. Weight keys: rename diffusers parameter names to VeOmni parameter names

Usage:
    python convert_to_veomni.py [--in-place]
"""

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

CKPT_DIR = os.path.dirname(os.path.abspath(__file__))

# VeOmni registers the LTX-2.3 transformer under this class name
# (see VeOmni/veomni/models/diffusers/ltx2_3/ltx_transformer/__init__.py).
# The diffusers checkpoint ships "_class_name": "LTX2VideoTransformer3DModel",
# which VeOmni's MODEL_CONFIG_REGISTRY does not recognise.
VEOMNI_CLASS_NAME = "LTXVideoTransformerModel"

# ── config field mapping ──────────────────────────────────────────────
CONFIG_ADDITIONS = {
    "cross_attention_adaln": ("cross_attn_mod", False),
    "apply_gated_attention": ("gated_attn", False),
    "with_audio": (None, True),          # always True for t2av model
    "av_ca_timestep_scale_multiplier": ("cross_attn_timestep_scale_multiplier", 1),
    "caption_proj_before_connector": (None, True),
}

# ── weight key rename mapping (prefix replacements) ──────────────────
# Applied in order; each replaces a prefix of the checkpoint key.
KEY_RENAMES = [
    # top-level adaln modules
    ("time_embed.", "adaln_single."),
    ("prompt_adaln.", "prompt_adaln_single."),
    ("audio_time_embed.", "audio_adaln_single."),
    ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
    ("av_cross_attn_video_scale_shift.", "av_ca_video_scale_shift_adaln_single."),
    ("av_cross_attn_audio_scale_shift.", "av_ca_audio_scale_shift_adaln_single."),
    ("av_cross_attn_video_a2v_gate.", "av_ca_a2v_gate_adaln_single."),
    ("av_cross_attn_audio_v2a_gate.", "av_ca_v2a_gate_adaln_single."),
    # transformer_blocks scale_shift_table renames
    ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
    ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
    # Q/K normalization parameter renames
    ("norm_q", "q_norm"),
    ("norm_k", "k_norm"),
    # patchify projection renames
    ("audio_proj_in", "audio_patchify_proj"),
    ("proj_in", "patchify_proj"),
]


def fix_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    original_class = config.get("_class_name")
    if original_class != VEOMNI_CLASS_NAME:
        config["_class_name"] = VEOMNI_CLASS_NAME
        print(f"  [_class_name] {original_class} -> {VEOMNI_CLASS_NAME}")
    for veomni_field, (src_field, default) in CONFIG_ADDITIONS.items():
        if veomni_field not in config:
            val = config.get(src_field, default) if src_field else default
            config[veomni_field] = val
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Updated {config_path}")
    for veomni_field, (src_field, _) in CONFIG_ADDITIONS.items():
        print(f"  {veomni_field} = {config[veomni_field]}")
    return config


def rename_key(key: str) -> str:
    for old, new in KEY_RENAMES:
        if old in key:
            key = key.replace(old, new)
    return key


def convert_weights(ckpt_dir: str, in_place: bool = False):
    index_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    # Build reverse map: file -> [keys]
    file_to_keys: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        file_to_keys.setdefault(fname, []).append(k)

    new_weight_map = {}
    for old_key, fname in weight_map.items():
        new_key = rename_key(old_key)
        new_weight_map[new_key] = fname
        if new_key != old_key:
            print(f"  [rename] {old_key} -> {new_key}")

    # Process each shard file
    for fname, keys in sorted(file_to_keys.items()):
        fpath = os.path.join(ckpt_dir, fname)
        print(f"[weights] Loading {fname} ...")
        state = load_file(fpath)
        new_state = {}
        for old_key in keys:
            new_key = rename_key(old_key)
            new_state[new_key] = state[old_key]
        out_path = fpath if in_place else fpath  # always in-place for now
        save_file(new_state, out_path, metadata={"format": "pt"})
        print(f"  Saved {len(new_state)} tensors to {fname}")

    # Update index
    index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[index] Updated {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true", default=True)
    args = parser.parse_args()

    print(f"=== Converting checkpoint in {CKPT_DIR} ===\n")

    # 1. Fix config.json
    config_path = os.path.join(CKPT_DIR, "config.json")
    fix_config(config_path)

    print()

    # 2. Rename weight keys
    convert_weights(CKPT_DIR, in_place=args.in_place)

    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()
