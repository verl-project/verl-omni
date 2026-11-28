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
"""Ray worker setup for standalone rollout groups in single-host smoke tests."""

import os
import re


def configure_gloo_file_store() -> None:
    group = os.environ.get("WG_PREFIX", "")
    if re.fullmatch(r"rollout_(?:reward_)?standalone_[0-9]+", group):
        # Each replica has its own rank zero and needs a separate rendezvous.
        root = os.environ["VERL_SMOKE_FILESTORE_DIR"]
        os.environ.setdefault("DIST_INIT_METHOD", f"file://{root}/{group}")
