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

import base64
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from io import BytesIO

from PIL import Image

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded data URI string.

    Args:
        image: The PIL Image to convert.

    Returns:
        A base64-encoded PNG data URI string (e.g. ``data:image/png;base64,...``).
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_image_text}"
    return base64_image


# Isolates math_verify's ``signal.alarm()`` calls that raise inside reward
# manager worker threads.  Worker function must live in an importable module
# (this file) so ``ProcessPoolExecutor`` can pickle it — reward scorers
# loaded via ``importlib.util.spec_from_file_location`` (e.g. mmk12.py) get
# a synthetic module name like ``custom_module_xxx`` that the subprocess
# cannot re-import.

_math_verify_pool = None
_math_verify_pool_lock = threading.Lock()


def _get_math_verify_pool() -> ProcessPoolExecutor:
    global _math_verify_pool
    if _math_verify_pool is None:
        with _math_verify_pool_lock:
            if _math_verify_pool is None:
                _math_verify_pool = ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _math_verify_pool


def _math_verify_worker(gt_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess with String+Latex+Expr configs.

    Extraction setup matches MM-EUREKA's ``accuracy_reward_func``:
    ``StringExtractionConfig`` handles single letters (A/B/C/D/E),
    ``LatexExtractionConfig`` handles ``\\boxed{}`` / ``$…$``,
    ``ExprExtractionConfig`` handles bare numbers.
    """
    from math_verify.grader import verify
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
        parse,
    )

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (
        StringExtractionConfig(),
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    )

    gold = parse(gt_boxed, gold_targets)
    pred = parse(model_output, pred_targets)
    if gold and pred:
        return max(1.0 if any(verify(g, p) for g in gold) else 0.0 for p in pred)
    return 0.0


def math_verify_score(model_output: str, ground_truth: str, timeout: float = 20.0) -> float:
    """Return 1.0 / 0.0 based on math_verify equivalence (subprocess-isolated).

    Wraps ``ground_truth`` in ``\\boxed{}`` before parsing (matches verl's
    ``math_verify.compute_score`` convention).  Returns 0.0 on timeout /
    parse error.
    """
    gt_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_math_verify_pool().submit(_math_verify_worker, gt_boxed, model_output)
        return future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        return 0.0
    except Exception as e:
        print(f"math_verify_score error: {e}", flush=True)
        return 0.0
