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
"""通过已有 agent registry 接入 Qwen-Image 2.1 的原生分词与参考图预处理。"""

from verl.experimental.agent_loop.agent_loop import register

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop

from .prompt import prepare_reference_media, tokenize_prompt


@register("qwen_image_21_diffusion_single_turn_agent")
class QwenImage21DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """复用通用 run()，仅替换现有媒体准备和 Continuous Token 扩展点。"""

    async def process_multi_modal_info(self, messages):
        """解码后缩放参考图；同一批图传给正负提示词和 rollout。"""
        media = await super().process_multi_modal_info(messages)
        return await self.loop.run_in_executor(None, prepare_reference_media, media)

    async def ct_build_initial_tokens(self, messages, tools=None, images=None, videos=None, audios=None):
        """使用模型原生模板分词，不截断图像 slot。"""
        if tools:
            raise ValueError("Qwen-Image 2.1 does not accept tool messages.")
        media = {"images": images, "videos": videos, "audios": audios}
        ids = await self.loop.run_in_executor(None, tokenize_prompt, messages, media, self.processor)
        if len(ids) > self.rollout_config.prompt_length:
            raise ValueError("Model prompt exceeds rollout.prompt_length; image slots cannot be truncated.")
        return ids

    async def apply_chat_template(
        self,
        messages,
        tools=None,
        images=None,
        videos=None,
        audios=None,
        mm_processor_kwargs=None,
        remove_system_prompt=False,
    ):
        """保持旧入口与 Continuous Token 使用同一条模型分词路径。"""
        del mm_processor_kwargs, remove_system_prompt
        return await self.ct_build_initial_tokens(messages, tools, images, videos, audios)
