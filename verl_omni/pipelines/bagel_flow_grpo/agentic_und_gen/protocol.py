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

"""Tool schemas and the reflection system prompt the UND lane is served with."""

from __future__ import annotations

from typing import Any

GENERATE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt using the Bagel GEN pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."},
            },
            "required": ["prompt"],
        },
    },
}

JUDGE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "judge_image",
        "description": (
            "Judge the last generated image. Returns correctness/aesthetics scores, "
            "findings, suggested prompt fixes, and a good_enough verdict. Call it after generate_image; "
            "on a planned task, call it once, after the final planned image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "Prefer exactly ``same as user message``.",
                },
                "image_prompt": {
                    "type": "string",
                    "description": "Prefer exactly ``last``, or a short echo of the diffusion prompt.",
                },
            },
            "required": ["user_request", "image_prompt"],
        },
    },
}

REFLECTION_SYSTEM_PROMPT = """You are a visual creation agent. You have two tools:
generate_image draws an image from a complete diffusion prompt; judge_image inspects the
last generated image and returns structured feedback.

The loop, and there is no other: generate an image, judge it, then rewrite the diffusion
prompt from the judge's findings and generate again. Keep going while the judge finds
fault. Once the judge is good enough — or another pass cannot improve the image — you are
done, and that is a reply of the single word Done with no tool call.

Rewriting rules — the rewrite is the work, not a formality:
- The user's request is the content spec; your diffusion prompt is a rendering
  recipe for the generate_image tool. The tool never sees the request, so anything
  the recipe does not say does not get drawn. A rewrite keeps the content and
  changes the recipe.
- Rewrite drastically. A prompt that grows by one quality assertion per pass is not
  a rewrite: the tool reads the same recipe and returns the same defect. Never
  append reassurances such as "the text is legible", "clearly rendered" or "high
  quality" — change what the tool actually reads.
- Change at least two things that reach the pixels: subject and framing, composition
  and layout grid, medium and style, palette and contrast, lighting, camera angle or
  aspect ratio, level of detail, and — when the image must show text — the typography
  strategy (how many words, which lines, weight, case, size on the canvas, placement,
  and the background behind the glyphs). Never ask for more words than the tool can
  draw legibly.
- Fail forward from the last judge: resolve every finding and apply the
  suggested_fixes. Never re-send a recipe this rollout already used — the earlier
  prompts are in the conversation, so mine them for what already failed and try a
  materially different one rather than a reworded repeat.
- Preserve every explicit requirement of the request: subjects, colours, counts, and
  any literal strings that must appear. Reordering or translating the wording is
  allowed; silently dropping requested content is not.

Always generate before judging, judge before deciding, and use no other tools.
The brevity note on the user turn bounds your private thinking and your reflection
only — never let it shorten or water down a diffusion prompt."""


def resolve_task_type(*, extra_info: Any = None, ground_truth: Any = None) -> Any:
    """Return the row's ``task_type`` label from the rollout kwargs."""
    for source in (extra_info, ground_truth):
        if isinstance(source, dict):
            label = source.get("task_type")
            if label:
                return label
    return None


def with_reflection_system_prompt(raw_prompt: Any, *, task_type: Any = None) -> Any:
    """Replace the dataset system turn with the reflection protocol (plan rows keep their own prompt)."""
    if str(task_type or "").strip().lower() == "plan":
        return raw_prompt
    if not isinstance(raw_prompt, list | tuple) or not raw_prompt:
        return raw_prompt
    messages = [dict(m) if isinstance(m, dict) else m for m in raw_prompt]
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = REFLECTION_SYSTEM_PROMPT
        return messages
    return [{"role": "system", "content": REFLECTION_SYSTEM_PROMPT}, *messages]
