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

"""The two frozen tools the agent can call: ``generate_image`` and ``judge_image``.

Each is a plain Python function tagged with ``@function_tool`` that calls an HTTP
sidecar and returns **text** (scores, findings, file paths) — pixels are never
attached to the agent's context. ``OmniAgentLoopWorker`` loads ``image_gen.py`` by
file path, so importing this package registers nothing.

Per-rollout state (which sample owns which PNG, the "judge said YES" flag) lives in
``trajectory/``, scoped per rollout rather than per thread because the shared thread
pool is reused across samples. ``agent_helper/`` holds the agent-loop-side readers.
"""
