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

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestBooguImageRecipe(unittest.TestCase):
    def test_v0_recipe_resolves_hub_model_before_appending_processor(self):
        repo_root = Path(__file__).parents[2]
        script = repo_root / "examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            captured_args = tmp_path / "python-args.txt"

            hf = bin_dir / "hf"
            hf.write_text("#!/usr/bin/env bash\nprintf '%s\\n' /tmp/boogu-snapshot\n")
            hf.chmod(0o755)

            python = bin_dir / "python3"
            python.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$CAPTURED_ARGS"\n')
            python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "CAPTURED_ARGS": str(captured_args),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "WORKSPACE": str(tmp_path),
                }
            )
            env.pop("MODEL_PATH", None)
            env.pop("REWARD_MODEL_PATH", None)

            result = subprocess.run(["bash", str(script)], cwd=repo_root, env=env, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            args = captured_args.read_text().splitlines()
            self.assertIn("actor_rollout_ref.model.path=/tmp/boogu-snapshot", args)
            self.assertIn("actor_rollout_ref.model.tokenizer_path=/tmp/boogu-snapshot/processor", args)


if __name__ == "__main__":
    unittest.main()
