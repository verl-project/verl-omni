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
"""Command-line entrypoint for offline diffusion model publishing."""

import json

from .base_model_merger import MergeResult, generate_config_from_args, parse_args, run_model_merger


def main() -> None:
    """Parse one operation, run its backend merger, and print the result."""
    config = generate_config_from_args(parse_args())
    result = run_model_merger(config)
    if isinstance(result, MergeResult):
        output = {"output_dir": str(result.output_dir), "manifest_path": str(result.manifest_path)}
    else:
        output = {"integrity": "passed", "architecture": result["architecture"]}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
