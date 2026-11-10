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

"""BAGEL plan protocol applied at the integration boundary."""

BAGEL_PLAN_SYSTEM_PROMPT = """You are a visual creation agent with two tools:
1) generate_image — create an image from a complete diffusion prompt
2) judge_image — inspect the last generated image and return structured feedback

Protocol:
1. Write a short numbered plan of at most three complete subtask image prompts.
2. Call generate_image once per planned subtask, in order.
3. After the final image, call judge_image on that image.
4. Reflect briefly on the feedback. If it says good_enough=NO and a generate_image call remains, \
rewrite the final image prompt from its findings, call generate_image, and judge again.
5. Finish with Done once good_enough=YES or no generate_image call remains.

Do not judge between subtasks or plan more subtask images than the request needs.

Start your reply like this, with your own prompts in place of the angle brackets:
Plan:
1. <complete image prompt for subtask 1>
2. <complete image prompt for subtask 2>
<tool_call>
{"name": "generate_image", "arguments": {"prompt": "<complete image prompt for subtask 1>"}}
</tool_call>"""


def prepare_bagel_prompt(messages, *, task_type=None, extra_info=None):
    """Use the BAGEL plan protocol for UniCoT rows; retain custom task prompts."""
    source = extra_info.get("unicot_source", "") if isinstance(extra_info, dict) else ""
    if str(task_type or "").strip().lower() != "plan" or source != "Fr0zencr4nE/UniCoT-Breakdown-3K":
        return messages
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return messages
    result = [dict(message) for message in messages]
    if result and result[0].get("role") == "system":
        result[0] = {**result[0], "content": BAGEL_PLAN_SYSTEM_PROMPT}
    else:
        result.insert(0, {"role": "system", "content": BAGEL_PLAN_SYSTEM_PROMPT})
    return result
