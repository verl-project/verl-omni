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
"""Check metadata precedence, version-file filtering and source fingerprint boundaries."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/collect_ascend_environment.py"
SPEC = importlib.util.spec_from_file_location("collect_ascend_environment", SCRIPT)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


class TestAscendEnvironmentManifest(unittest.TestCase):
    """Verify reproducibility data without importing any accelerator package."""

    def test_duplicate_metadata_uses_the_default_resolver_not_the_last_record(self):
        distributions = [
            Mock(metadata={"Name": "torch"}, version="2.10.0"),
            Mock(metadata={"Name": "torch"}, version="stale-vendored-record"),
            Mock(metadata={"Name": "numpy"}, version="1.26.4"),
        ]
        with (
            patch.object(COLLECTOR.importlib.metadata, "distributions", return_value=distributions),
            patch.object(
                COLLECTOR.importlib.metadata, "version", side_effect={"torch": "2.10.0", "numpy": "1.26.4"}.get
            ),
        ):
            manifest = COLLECTOR.collect_packages()
        self.assertEqual(manifest["selected"]["torch"], "2.10.0")
        self.assertEqual(manifest["duplicate_metadata"]["torch"], ["2.10.0", "stale-vendored-record"])

    def test_version_files_omit_installer_identity_and_sensitive_build_fields(self):
        content = (
            "Version=1.2.3\nBuildDate=2026-01-01\nUserName=private-user\n"
            "BuildPath=/a/private/path\nVersionToken=private-token\nBuildURL=https://internal.example\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version.info"
            path.write_bytes(content.encode())
            manifest = COLLECTOR.collect_version_file(path)
        self.assertEqual(manifest["fields"], {"Version": "1.2.3", "BuildDate": "2026-01-01"})
        self.assertEqual(manifest["sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_cli_runs_without_site_packages_and_records_git_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

            def git(*arguments):
                return subprocess.check_output(
                    ["git", "-C", str(root), *arguments], env=environment, stderr=subprocess.PIPE
                )

            git("init")
            git(
                "-c",
                "user.name=Manifest Test",
                "-c",
                "user.email=manifest@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "--no-verify",
                "-m",
                "test fixture",
            )
            output = root / "manifest.json"
            result = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), "--output", str(output), "--repo", f"source={root}"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text())
            self.assertEqual(manifest["packages"]["selected"], {})
            self.assertEqual(manifest["repositories"]["source"]["commit"], git("rev-parse", "HEAD").decode().strip())
            self.assertNotIn("path", manifest["repositories"]["source"])
            self.assertNotIn("remote", manifest["repositories"]["source"])
            (root / "untracked.txt").write_text("private data", encoding="utf-8")
            snapshot = COLLECTOR.collect_repository(root)
            self.assertTrue(snapshot["dirty"])
            self.assertNotIn("private data", json.dumps(snapshot))

    def test_invalid_or_duplicate_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in [["--repo", "missing-separator"], ["--repo", "a=x", "--repo", "a=y"]]:
                with self.subTest(options=options):
                    output = Path(directory) / "manifest.json"
                    result = subprocess.run(
                        [sys.executable, "-S", str(SCRIPT), "--output", str(output), *options],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
