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

"""Shared OpenAI-chat POST helper for the frozen agentic image-judge sidecar."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

__all__ = ["post_vllm_chat"]


def post_vllm_chat(
    *,
    vllm_url: str,
    image_b64: str,
    prompt_text: str,
    max_tokens: int,
    model: str = "",
    timeout: float = 120.0,
    enable_thinking: bool = False,
) -> tuple[str | None, str | None]:
    """POST OpenAI ``/v1/chat/completions`` for a single image + text prompt.

    Args:
        vllm_url: Base URL of the OpenAI-compatible server.
        image_b64: PNG as base64 (no data-URL prefix).
        prompt_text: User text sent with the image.
        max_tokens: Completion budget.
        model: Optional model id; omitted from the payload when empty.
        timeout: HTTP timeout in seconds.
        enable_thinking: Passed through ``chat_template_kwargs``.

    Returns:
        ``(raw_text, None)`` on success, or ``(None, error)`` on failure.
    """
    payload: dict = {
        "model": str(model or "").strip(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if not payload["model"]:
        del payload["model"]
    try:
        req = Request(
            f"{vllm_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=float(timeout)) as resp:  # noqa: S310 - operator-configured
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    choices = data.get("choices") or []
    raw_text = ""
    if choices:
        raw_text = str(choices[0].get("message", {}).get("content", "") or "")
    if not raw_text:
        return None, "empty_response"
    return raw_text, None
