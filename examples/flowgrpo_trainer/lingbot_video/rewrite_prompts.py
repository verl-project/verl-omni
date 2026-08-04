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

"""Rewrite plain-text video prompts into LingBot structured JSON captions.

LingBot-Video DiT consumes structured JSON captions, not casual natural-language
prompts (see the model card).  This driver streams a list of plain prompts
through the official LingBot two-stage rewriter (step1 EXPAND = base VLM; step2
MAP = base VLM + rewriter LoRA) and writes one JSONL record per prompt in the
schema expected by ``prepare_structured_captions.py``.

Two interchangeable backends select how the 27B VLM is served:

* ``--backend vllm`` (default, high throughput): talk to a standalone vLLM
  OpenAI-compatible server (see ``serve_rewriter.sh``), picking the LoRA adapter
  per request via the ``model`` field.  Many prompts run concurrently through a
  producer/consumer worker pool so vLLM continuous-batches them; a tqdm bar shows
  progress.  This is the recommended production path.
* ``--backend transformers`` (reference/fallback, low throughput): load the base
  VLM + LoRA in-process via the bundled ``rewriter/inference.py`` and stream one
  prompt at a time.  Shard across GPUs with ``--num-shards N --shard i`` and
  ``CUDA_VISIBLE_DEVICES=i`` (see ``run_rewrite.sh`` with ``BACKEND=transformers``).

Both backends share one output schema and are resumable: prompts already present
in ``--output`` are skipped, so a killed run can be relaunched with the same
arguments.  The reward ground truth is the ORIGINAL plain prompt (HPSv3 scores
video/text alignment against the human intent), while ``caption`` carries the
structured rewrite fed to the DiT.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import requests
from tqdm import tqdm


def _load_prompts(path: str, num_shards: int, shard: int) -> list[str]:
    if num_shards <= 0:
        raise ValueError(f"`num_shards` must be positive, got {num_shards}.")
    if not 0 <= shard < num_shards:
        raise ValueError(f"`shard` must be in [0, {num_shards}), got {shard}.")

    prompts: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            # The official rewriter explicitly supports Chinese prompts, including
            # the Chinese duration suffix in its step-1 instruction.  Do not drop
            # them during dataset preparation.
            if not s or s in seen:
                continue
            seen.add(s)
            prompts.append(s)
    return [p for i, p in enumerate(prompts) if i % num_shards == shard]


def _already_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["prompt_raw"])
            except (json.JSONDecodeError, KeyError, TypeError):
                # TypeError: a line that is valid JSON but not an object (a bare
                # scalar/array) is not subscriptable; skip it like a malformed line
                # instead of aborting the whole resumable run.
                continue
    return done


def build_record(prompt: str, caption: dict, args: argparse.Namespace) -> dict:
    """Assemble the JSONL record shared by both backends."""
    return {
        "prompt_raw": prompt,
        "caption": caption,
        "duration": int(round(args.duration)),
        "data_source": args.data_source,
        "reward_model": {"style": "model", "ground_truth": prompt},
    }


class VLLMBackend:
    """OpenAI-chat backend mirroring the reference ``TransformersBackend``.

    ``generate(text, image, use_lora)`` posts one chat completion.  ``use_lora``
    picks the served model id: the LoRA adapter (step2 MAP) or the base VLM
    (step1 EXPAND).  A thread-local ``requests.Session`` keeps this safe under a
    thread pool.
    """

    def __init__(
        self,
        base_url: str,
        base_model: str = "qwen3_5_27b",
        lora_model: str = "rewriter",
        max_new_tokens: int = 6144,
        timeout: float = 900.0,
    ):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.base_model = base_model
        self.lora_model = lora_model
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def generate(self, text, image, use_lora):
        if image is not None:
            raise NotImplementedError("VLLMBackend supports text-only (T2V/T2I) prompts; no first frame.")
        payload = {
            "model": self.lora_model if use_lora else self.base_model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            # Mirror the reference backend's apply_chat_template(enable_thinking=False).
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = self._session().post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _process_one(rewriter, prompt: str, args: argparse.Namespace, neg_editor=None) -> dict | None:
    """Run both rewrite stages; return the JSONL record, or None if step2 output
    is not a structured JSON object (counted as ``empty``)."""
    result = rewriter.rewrite(prompt, mode=args.mode, duration=args.duration)
    caption = result["json"]
    if not isinstance(caption, dict):
        return None
    record = build_record(prompt, caption, args)
    if neg_editor is not None:
        record["negative_caption"] = neg_editor.edit(caption, mode=args.mode)["negative_json"]
    return record


def run_vllm(rewriter, todo: list[str], out_path: str, args: argparse.Namespace, neg_editor=None) -> dict:
    """Producer/consumer over a bounded queue: one producer feeds prompts, a fixed
    pool of ``--concurrency`` consumers each pull the next prompt the instant they
    are free.  Exactly ``concurrency`` prompts are in flight at steady state (the
    server continuous-batches them), the queue never starves while work remains,
    and memory stays bounded — no 45k-future cold-start burst."""
    concurrency = args.concurrency
    q: queue.Queue = queue.Queue(maxsize=args.queue_size or concurrency * 2)
    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    counters = {"ok": 0, "empty": 0, "err": 0}
    pbar = tqdm(
        total=len(todo),
        desc=f"rewrite[vllm shard {args.shard}/{args.num_shards}]",
        mininterval=5.0,
        dynamic_ncols=True,
        smoothing=0.05,
    )
    out = open(out_path, "a", encoding="utf-8")

    def consumer():
        while True:
            prompt = q.get()
            if prompt is None:  # sentinel: no more work
                q.task_done()
                return
            status = "ok"
            try:
                record = _process_one(rewriter, prompt, args, neg_editor=neg_editor)
                if record is None:
                    status = "empty"
                else:
                    line = json.dumps(record, ensure_ascii=False)
                    with write_lock:
                        out.write(line + "\n")
                        out.flush()
            except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the run
                status = "err"
                tqdm.write(f"[shard {args.shard}] ERR: {type(exc).__name__}: {exc}", file=sys.stderr)
            with stats_lock:
                counters[status] += 1
                pbar.update(1)
                pbar.set_postfix(ok=counters["ok"], empty=counters["empty"], err=counters["err"], refresh=False)
            q.task_done()

    workers = [threading.Thread(target=consumer, name=f"rw-{i}", daemon=True) for i in range(concurrency)]
    for w in workers:
        w.start()
    try:
        for prompt in todo:  # producer (main thread); blocks when the queue is full
            q.put(prompt)
        for _ in range(concurrency):
            q.put(None)  # one sentinel per consumer
        for w in workers:
            w.join()
    finally:
        pbar.close()
        out.close()
    return counters


def run_transformers(rewriter, todo: list[str], out_path: str, args: argparse.Namespace, neg_editor=None) -> dict:
    """Single-threaded loop: the in-process 27B model serves one request at a time,
    so concurrency comes from multiple sharded processes, not threads."""
    counters = {"ok": 0, "empty": 0, "err": 0}
    with open(out_path, "a", encoding="utf-8") as out:
        for prompt in tqdm(
            todo,
            desc=f"rewrite[hf shard {args.shard}/{args.num_shards}]",
            mininterval=5.0,
            dynamic_ncols=True,
        ):
            try:
                record = _process_one(rewriter, prompt, args, neg_editor=neg_editor)
                if record is None:
                    counters["empty"] += 1
                else:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    counters["ok"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the shard
                counters["err"] += 1
                tqdm.write(f"[shard {args.shard}] ERR: {type(exc).__name__}: {exc}", file=sys.stderr)
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shared
    ap.add_argument("--input", required=True, help="Plain-text prompts, one per line.")
    ap.add_argument("--output", required=True, help="Output JSONL (appended, resumable).")
    ap.add_argument("--rewriter-dir", required=True, help="Path to lingbot-video/rewriter.")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--mode", default="t2v", choices=["t2v", "ti2v", "t2i"])
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--data-source", default="dance_grpo/hpsv3")
    ap.add_argument(
        "--with-negative",
        action="store_true",
        help="Also run auto-negative per caption (adds one 27B generate per prompt). "
        "Off by default: the agent loop falls back to the official DEFAULT_NEGATIVE_PROMPT.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Rewrite at most N prompts (0 = all); for smoke tests.")
    # vLLM backend
    ap.add_argument("--base-url", default="http://127.0.0.1:8137")
    ap.add_argument("--base-model", default="qwen3_5_27b", help="Served model id for step1 (base, no LoRA).")
    ap.add_argument("--lora-model", default="rewriter", help="Served model id for step2 (rewriter LoRA).")
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--concurrency", type=int, default=256, help="In-flight prompts (vllm backend).")
    ap.add_argument("--queue-size", type=int, default=0, help="Bounded work-queue size (0 = 2*concurrency).")
    # transformers backend
    ap.add_argument("--base", default=os.environ.get("REWRITER_BASE_MODEL", ""), help="Base VLM path (hf backend).")
    ap.add_argument("--adapter", default=os.environ.get("REWRITER_ADAPTER", ""), help="LoRA adapter path (hf backend).")
    args = ap.parse_args()

    if args.backend == "vllm" and args.concurrency <= 0:
        ap.error("--concurrency must be positive")
    if args.queue_size < 0:
        # queue.Queue treats maxsize <= 0 as unbounded, which would defeat the
        # producer/consumer backpressure; only 0 (== "2*concurrency") is allowed.
        ap.error("--queue-size must be >= 0 (0 selects the default 2*concurrency)")

    sys.path.insert(0, args.rewriter_dir)
    from rewriter_core import Rewriter  # noqa: E402

    prompts = _load_prompts(args.input, args.num_shards, args.shard)
    done = _already_done(args.output)
    todo = [p for p in prompts if p not in done]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(
        f"[shard {args.shard}/{args.num_shards}] backend={args.backend} {len(prompts)} assigned, "
        f"{len(done)} already done, {len(todo)} to rewrite"
        + (f", concurrency={args.concurrency}" if args.backend == "vllm" else ""),
        flush=True,
    )
    if not todo:
        print("[done] nothing to do", flush=True)
        return

    if args.backend == "vllm":
        backend = VLLMBackend(
            args.base_url,
            base_model=args.base_model,
            lora_model=args.lora_model,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        from inference import make_backend  # noqa: E402

        backend = make_backend("transformers", args.base, args.adapter)
    rewriter = Rewriter(backend)

    neg_editor = None
    if args.with_negative:
        from auto_negative import NegativePromptEditor  # noqa: E402

        neg_editor = NegativePromptEditor(backend)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    runner = run_vllm if args.backend == "vllm" else run_transformers
    counters = runner(rewriter, todo, args.output, args, neg_editor)

    print(
        f"[shard {args.shard}] DONE backend={args.backend} ok={counters['ok']} "
        f"empty={counters['empty']} err={counters['err']} in {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
