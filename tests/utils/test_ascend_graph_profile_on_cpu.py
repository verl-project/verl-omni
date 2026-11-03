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
"""Exercise the graph profile through a CPU-only replacement for the base recipe."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAME = "run_qwen3_omni_thinker_gspo_avqa_npu_graph_v1.sh"
BASH = os.environ.get("BASH_BINARY") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required for the shell recipe contract tests")
class TestAscendGraphProfile(unittest.TestCase):
    """Check the actual argument and exit-status boundary without importing the trainer."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "repository with spaces"
        self.recipe = self.root / "examples/gspo_trainer/qwen3_omni"
        self.recipe.mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "examples/gspo_trainer/qwen3_omni" / PROFILE_NAME, self.recipe / PROFILE_NAME)
        self.record = self.root / "record.json"
        self.capture = self.root / "capture.py"
        self.capture.write_text(
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "keys=['DEVICE','PYTHONUNBUFFERED','NUM_GPUS_ACTOR_ROLLOUT_REWARD','ROLLOUT_TP','MODEL_PATH']\n"
            "Path(os.environ['STUB_RECORD']).write_text(json.dumps({'args':sys.argv[1:],"
            "'env':{k:os.environ.get(k) for k in keys},'cwd':os.getcwd()}))\n"
            "sys.exit(int(os.environ.get('STUB_EXIT','0')))\n",
            encoding="utf-8",
        )
        (self.recipe / "run_qwen3_omni_thinker_gspo_avqa_v1.sh").write_text(
            '#!/usr/bin/env bash\n"${STUB_PYTHON}" "${STUB_CAPTURE}" "$@" | cat\n', encoding="utf-8"
        )
        self.environment = os.environ.copy()
        for key in ["NPU_GRAPH_MODE", "NUM_GPUS_ACTOR_ROLLOUT_REWARD", "ROLLOUT_TP"]:
            self.environment.pop(key, None)
        self.environment.update(
            STUB_PYTHON=sys.executable,
            STUB_CAPTURE=str(self.capture),
            STUB_RECORD=str(self.record),
            MSYS2_ARG_CONV_EXCL="*",
            MSYS2_ENV_CONV_EXCL="MODEL_PATH",
        )

    def run_profile(self, *arguments, **environment):
        """Run the wrapper from a different directory to check recipe path resolution."""
        return subprocess.run(
            [BASH, str(self.recipe / PROFILE_NAME), *arguments],
            cwd=self.root.parent,
            env=self.environment | environment,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_default_profile_preserves_workload_and_sets_graph_budget(self):
        result = self.run_profile()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(self.record.read_text())
        arguments = record["args"]
        self.assertIn("actor_rollout_ref.rollout.enable_chunked_prefill=true", arguments)
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=1536", arguments)
        self.assertIn(
            "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode=PIECEWISE",
            arguments,
        )
        self.assertIn(
            "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_capture_sizes=[1,16,128,1536]",
            arguments,
        )
        self.assertFalse(any(arg.startswith("data.") or arg.startswith("trainer.") for arg in arguments))
        self.assertEqual(record["env"]["DEVICE"], "npu")
        self.assertEqual(record["env"]["PYTHONUNBUFFERED"], "1")
        self.assertEqual(record["env"]["NUM_GPUS_ACTOR_ROLLOUT_REWARD"], "8")
        self.assertEqual(record["env"]["ROLLOUT_TP"], "2")
        self.assertEqual(Path(record["cwd"]), self.root)

    def test_full_profile_preserves_caller_overrides_and_spaced_paths(self):
        overrides = [
            "data.max_response_length=12288",
            "trainer.test_freq=10",
            "actor_rollout_ref.model.path=/a path/model",
        ]
        result = self.run_profile(
            *overrides,
            NPU_GRAPH_MODE="FULL_AND_PIECEWISE",
            NUM_GPUS_ACTOR_ROLLOUT_REWARD="4",
            ROLLOUT_TP="2",
            MODEL_PATH="/a path/model",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(self.record.read_text())
        self.assertEqual(record["args"][-len(overrides) :], overrides)
        self.assertTrue(any(arg.endswith("cudagraph_mode=FULL_AND_PIECEWISE") for arg in record["args"]))
        self.assertEqual(record["env"]["MODEL_PATH"], "/a path/model")
        self.assertEqual(record["env"]["NUM_GPUS_ACTOR_ROLLOUT_REWARD"], "4")

    def test_pipeline_failure_is_not_hidden_by_the_base_recipe_tee(self):
        result = self.run_profile(STUB_EXIT="7")
        self.assertEqual(result.returncode, 7)

    def test_invalid_profiles_fail_before_the_trainer_starts(self):
        cases = [
            {"NPU_GRAPH_MODE": "FULL"},
            {"NPU_GRAPH_MODE": "unrecognized"},
            {"NUM_GPUS_ACTOR_ROLLOUT_REWARD": "0"},
            {"NUM_GPUS_ACTOR_ROLLOUT_REWARD": "08"},
            {"NUM_GPUS_ACTOR_ROLLOUT_REWARD": "3", "ROLLOUT_TP": "2"},
            {"ROLLOUT_TP": "0"},
            {"ROLLOUT_TP": "two"},
        ]
        for environment in cases:
            with self.subTest(environment=environment):
                result = self.run_profile(**environment)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
