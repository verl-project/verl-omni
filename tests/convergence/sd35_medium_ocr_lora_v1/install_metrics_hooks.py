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
"""Driver-side Tracking hook that appends trainer metrics to a JSONL file.

L4 does not dump tensors. The driver already emits the consistency and
stability scalars; this hook only persists them for ``collect_report.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_WRAPPED_ATTR = "__l4_metrics_wrapped__"


def _json_default(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def install_metrics_hooks() -> None:
    """Wrap ``verl.utils.tracking.Tracking.log`` so each step is appended to JSONL."""
    try:
        from verl.utils.tracking import Tracking
    except Exception:
        return

    original_log = Tracking.log
    if getattr(original_log, _WRAPPED_ATTR, False):
        return

    def log_with_jsonl(self, data, step, *args, **kwargs):
        metrics_path = os.environ.get("L4_METRICS_JSONL") or os.environ.get("DEBUG_METRICS_JSONL")
        if metrics_path:
            path = Path(metrics_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"step": int(step), "data": data}
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        return original_log(self, data, step, *args, **kwargs)

    setattr(log_with_jsonl, _WRAPPED_ATTR, True)
    Tracking.log = log_with_jsonl
