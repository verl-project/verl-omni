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

"""Worker RPCs reach the diffusion runner that owns conditioning-cache state."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[4] / "verl_omni/workers/rollout/vllm_rollout/utils.py"


def worker_method(name):
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMOmniColocateWorkerExtension"
    )
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert nodes, f"Worker must expose runner RPC {name}"
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_clear_and_stats_use_same_runner_cache(enabled):
    class Runner:
        def __init__(self):
            self.entries = {"prompt": "conditioning from old weights"} if enabled else None
            self.calls = []

        def clear_prompt_embed_cache(self):
            self.calls.append("clear")
            if self.entries is not None:
                self.entries.clear()

        def get_prompt_embed_cache_stats(self):
            self.calls.append("stats")
            return None if self.entries is None else {"size": len(self.entries)}

    runner = Runner()
    worker = SimpleNamespace(model_runner=runner)
    stats = worker_method("get_prompt_embed_cache_stats")
    clear = worker_method("clear_prompt_embed_cache")
    assert stats(worker) == ({"size": 1} if enabled else None)
    assert clear(worker) is None
    assert stats(worker) == ({"size": 0} if enabled else None)
    assert runner.calls == ["stats", "clear", "stats"]


def test_worker_does_not_report_success_when_runner_flush_fails():
    def fail():
        raise RuntimeError("cache flush failed")

    worker = SimpleNamespace(model_runner=SimpleNamespace(clear_prompt_embed_cache=fail))
    with pytest.raises(RuntimeError, match="cache flush failed"):
        worker_method("clear_prompt_embed_cache")(worker)
