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
"""Create toy prompt parquet files for Mode (2a) agentic GRPO smoke / e2e.

Seeds a reflection-oriented Hermes multi-turn chat for VisionCreator-R1-style
GRPO on Lance_3B_hf_und (frozen Lance MoT diffusion tool).
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DATA_SOURCE = "jpeg_compressibility"
ABILITY = "agentic_prompt_rewrite"
EXPECTED_NUM_IMAGES = 2

SYSTEM_PROMPT = """You are a visual creation agent that improves images by reflection.

Workflow (required, every task):
1) Emit Hermes tool call generate_image with a detailed first prompt.
2) After the tool observation, write one short line starting with \
"Reflection:" that names what to improve (detail, lighting, attributes).
3) Emit a SECOND Hermes generate_image call with a REWRITTEN prompt \
(must differ from the first; add missing attributes / quality cues).
4) After the second tool result, give a short final confirmation (no tool call).

Exact tool format:
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "<complete image prompt>"}}
</tool_call>

Never copy or paraphrase tool observations (no "Lance frozen MoT", no \
"agentic_tool ok=", no path= lines). Never emit bare JSON without <tool_call> \
XML. First assistant turn must be a Hermes tool call only.
"""

FEWSHOT_USER = "Generate an image of a red apple"
FEWSHOT_ASSISTANT_1 = (
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting"}}\n'
    "</tool_call>"
)
# Keep tool obs short/machine-readable so the policy does not memorize prose echoes.
FEWSHOT_TOOL = "agentic_tool ok=1 images=1 path=/tmp/example/image_00.png"
FEWSHOT_ASSISTANT_2 = (
    "Reflection: edges soft and reds muted; rewrite for sharper detail.\n"
    "<tool_call>\n"
    '{"name": "generate_image", "arguments": {"prompt": '
    '"a bright red apple on a white table, soft studio lighting, '
    'highly detailed, sharp focus, richer reds, coherent composition"}}\n'
    "</tool_call>"
)

USER_PROMPTS = [
    "Generate an image of a cat wearing a blue hat",
    "Create a sunset over snowy mountains with a red cabin",
    "Draw a silver robot painting a colorful landscape on an easel",
    "A glass of orange juice next to three green apples on a wooden table",
    "A yellow bicycle leaning against a blue brick wall in soft morning light",
    "A small brown dog wearing red sunglasses sitting on a white sofa",
    "An astronaut holding a purple umbrella on the surface of Mars",
    "A vintage typewriter with the word HELLO typed in bold letters",
]


def build_prompt_messages(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_USER},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_1},
        {"role": "tool", "content": FEWSHOT_TOOL},
        {"role": "assistant", "content": FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": user_text},
    ]


def build_ground_truth(user_text: str) -> dict:
    """Reflection-weighted GT for 100-step overfit AC."""
    return {
        "user_request": user_text,
        "expected_num_images": EXPECTED_NUM_IMAGES,
        "w_format": 0.25,
        "w_reflect": 0.35,
        "w_tool": 0.2,
        "w_result": 0.2,
        "forced_consolation": 0.05,
    }


def build_rows(split: str, n: int, prompts: list[str] | None = None) -> list[dict]:
    prompt_pool = prompts or USER_PROMPTS
    rows = []
    for i in range(n):
        prompt_text = prompt_pool[i % len(prompt_pool)]
        gt = build_ground_truth(prompt_text)
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": build_prompt_messages(prompt_text),
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "raw_prompt": prompt_text,
                    "toy_agentic": True,
                    "expected_num_images": EXPECTED_NUM_IMAGES,
                    "require_multiturn_tools": True,
                    **{k: gt[k] for k in ("w_format", "w_reflect", "w_tool", "w_result")},
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy agentic GRPO parquet seeds for the one-step smoke")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/agentic"))
    parser.add_argument("--train_size", type=int, default=64)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Repeat 2 prompts only (accelerates reflection learning in ~100 steps)",
    )
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)
    prompts = USER_PROMPTS[:2] if args.overfit else None
    train_df = pd.DataFrame(build_rows("train", args.train_size, prompts))
    val_df = pd.DataFrame(build_rows("val", args.val_size, prompts))
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")
    print(f"reflection few-shot; overfit={args.overfit}; w_reflect={build_ground_truth('x')['w_reflect']}")


if __name__ == "__main__":
    main()
