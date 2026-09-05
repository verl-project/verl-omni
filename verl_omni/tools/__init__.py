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

"""Frozen ``generate_image`` / ``judge_image`` tools, loaded by file path.

``OmniAgentLoopWorker`` binds ``image_gen.py`` as ``function_tool_path``.
Rollout ContextVars and artifact registry live in ``verl_omni.tools.trajectory``.
Importing this package does not register the tools.
"""
