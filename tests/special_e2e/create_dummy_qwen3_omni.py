"""Build a tiny random-weight Qwen3-Omni model for smoke tests.

Shrinks the thinker text config to 2 hidden layers so the model fits in
seconds without the ~60 GB download.  The full model config and tokenizer
are pulled from HF; only the weight tensors are randomly initialised.

Usage:
    python tests/special_e2e/create_dummy_qwen3_omni.py --output_dir ~/models/tiny-random/Qwen3-Omni

The tokenizer is copied from the Instruct variant so that chat_template is
available for verl's dataset loader.
"""

import argparse
import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Importing verl_omni applies the patch that registers Qwen3OmniMoeConfig with
# AutoModelForCausalLM (the Thinker is decoder-only despite its config class),
# so AutoModelForCausalLM.from_config below can build it.
import verl_omni  # noqa: F401, E402

SRC = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def build(output_dir: str) -> None:
    config = AutoConfig.from_pretrained(SRC)
    text_cfg = config.thinker_config.text_config
    text_cfg.num_hidden_layers = 2
    text_cfg.hidden_size = 256
    text_cfg.intermediate_size = 512
    text_cfg.num_experts = 4
    text_cfg.num_experts_per_tok = 2

    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(output_dir)

    AutoTokenizer.from_pretrained(SRC).save_pretrained(output_dir)
    print(f"Dummy Qwen3-Omni saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.expanduser("~"), "models", "tiny-random", "Qwen3-Omni"),
    )
    args = parser.parse_args()
    build(args.output_dir)
