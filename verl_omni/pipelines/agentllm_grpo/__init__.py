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

"""Agentic LLM GRPO recipe: the image-generate / judge tool agent loop.

Do not re-export ``ImageGenToolAgentLoop`` from here: ``verl_omni.agent_loop``
imports this package to fire the ``@register`` decorator, so an eager re-export
would close an import cycle.
"""
