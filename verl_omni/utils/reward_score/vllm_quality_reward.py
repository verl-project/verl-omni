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

"""
自定义 vLLM 奖励函数示例：使用 VLM 评估图像质量和提示词匹配度

这个示例展示如何编写一个自定义奖励函数，通过异步调用 vLLM 推理服务
来评估生成图像的质量。可以用于：
1. 文本到图像（T2I）任务
2. 图像编辑（I2I）任务
3. 自定义评估标准

特性：
- 异步 API 调用（高效并行处理）
- 结构化 JSON 解析
- 容错机制
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _to_pil(image) -> Image.Image:
    """将张量/数组转换为 PIL 图像。"""
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "必须是 HWC 格式"
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image


async def _call_vllm_api(
    session: aiohttp.ClientSession,
    router_address: str,
    messages: list[dict],
    model_name: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Optional[str]:
    """异步调用 vLLM OpenAI 兼容 API。

    Args:
        session: aiohttp 客户端会话
        router_address: vLLM 服务地址，格式为 "host:port"
        messages: OpenAI 格式的消息列表
        model_name: 模型名称
        temperature: 采样温度（低值更稳定）
        max_tokens: 最大生成 token 数

    Returns:
        生成的文本内容，失败返回 None
    """
    url = f"http://{router_address}/v1/chat/completions"

    request_data = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }

    try:
        async with session.post(url, json=request_data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                logger.warning(f"vLLM API 返回状态码 {resp.status}")
                return None
            result = await resp.json()
            return result["choices"][0]["message"]["content"]
    except asyncio.TimeoutError:
        logger.error("vLLM API 调用超时")
        return None
    except Exception as e:
        logger.error(f"vLLM API 调用失败: {e}")
        return None


def _extract_score_from_json(response_text: str) -> Optional[float]:
    """从 vLLM 响应中提取分数。

    尝试多种方式解析分数：
    1. JSON 格式：{"score": 0.8, ...}
    2. 直接数字：8/10, 0.8, 8 等
    3. 关键词：excellent=1.0, good=0.8, fair=0.5, poor=0.2

    Args:
        response_text: vLLM 返回的原始文本

    Returns:
        分数（0-1），失败返回 None
    """
    if not response_text:
        return None

    response_text = response_text.strip()

    # 方法 1：尝试 JSON 解析
    try:
        json_match = response_text.find("{")
        if json_match != -1:
            json_end = response_text.rfind("}") + 1
            if json_end > json_match:
                json_str = response_text[json_match:json_end]
                result = json.loads(json_str)
                if "score" in result:
                    score = float(result["score"])
                    return min(1.0, max(0.0, score))  # 裁剪到 [0, 1]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # 方法 2：查找数字分数（如 8/10, 8.0, 8）
    import re

    # 匹配 X/10 格式
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response_text)
    if match:
        try:
            score = float(match.group(1)) / 10.0
            return min(1.0, max(0.0, score))
        except ValueError:
            pass

    # 匹配 0.X 或 X 格式
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", response_text)
    if match:
        try:
            num = float(match.group(1))
            # 如果数字是 1-10 范围，假设是 10 分制
            if 1 <= num <= 10:
                return min(1.0, max(0.0, num / 10.0))
            # 如果数字是 0-1 范围，直接使用
            elif 0 <= num <= 1:
                return num
        except ValueError:
            pass

    # 方法 3：关键词匹配
    response_lower = response_text.lower()
    if any(word in response_lower for word in ["excellent", "perfect", "very good", "很好", "完美"]):
        return 0.9
    elif any(word in response_lower for word in ["good", "ok", "acceptable", "不错", "可以"]):
        return 0.7
    elif any(word in response_lower for word in ["fair", "average", "mediocre", "一般", "还可以"]):
        return 0.5
    elif any(word in response_lower for word in ["poor", "bad", "very bad", "不好", "差"]):
        return 0.3

    # 默认返回 None（调用者会使用备用值）
    return None


async def compute_score_vllm_quality(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
) -> dict[str, float | str | int]:
    """使用 vLLM 评估图像质量的奖励函数。

    通过异步调用 vLLM 推理服务，使用 VLM 模型评估生成图像与提示词的匹配度。

    Args:
        data_source: 数据源标识（如 "image_edit", "text2image"）
        solution_image: 生成的图像，形状为 (C, H, W) 或 (N, C, H, W)
        ground_truth: 参考文本（用户指令或提示词）
        extra_info: 额外信息字典，可能包含：
            - frame_interval: 视频帧间隔（如果是视频）
            - num_turns: 对话轮数
            - source_image_path: 源图像路径（用于图像编辑任务）
        reward_router_address: vLLM 服务地址 "host:port"
        reward_model_tokenizer: 分词器（保留用于接口一致性）
        model_name: VLM 模型名称（如 "qwen-vl-7b"）

    Returns:
        dict: 包含以下字段：
            - "score": float (0-1)，最终综合分数
            - "quality_score": float，图像质量分数
            - "match_score": float，与提示词匹配度分数
            - "avg_score": float，平均分数
            - "num_images": int，评估的图像数量
            - "response": str，VLM 的原始评估文本
            - "model": str，使用的模型名称

    Example:
        >>> score_result = await compute_score_vllm_quality(
        ...     data_source="image_edit",
        ...     solution_image=image_tensor,  # (C, H, W)
        ...     ground_truth="Make the object red",
        ...     extra_info={"source_image_path": "/path/to/source.jpg"},
        ...     reward_router_address="127.0.0.1:8000",
        ...     model_name="qwen-vl-7b",
        ... )
        >>> print(score_result["score"])  # 0.75
    """
    from verl.utils.ray_utils import get_event_loop

    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    if extra_info is None:
        extra_info = {}

    # 确保图像是 4D 张量 (N, C, H, W)
    if solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)

    # 提取参数
    frame_interval = extra_info.get("frame_interval", 1)
    if solution_image.ndim == 4:
        solution_image = solution_image[::frame_interval]

    model_name = model_name or "qwen-vl-7b"
    reward_router_address = reward_router_address or "127.0.0.1:8000"

    loop = get_event_loop()
    quality_scores = []
    match_scores = []
    responses = []

    # 创建共享的 aiohttp 会话
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for image_idx, image in enumerate(solution_image):
            # 转换为 PIL 图像并编码为 base64
            try:
                pil_image = _to_pil(image)
                image_base64 = await loop.run_in_executor(None, pil_image_to_base64, pil_image)
            except Exception as e:
                logger.error(f"图像转换失败: {e}")
                quality_scores.append(0.5)
                match_scores.append(0.5)
                continue

            # 构建评估提示词
            if data_source == "image_edit":
                eval_prompt = (
                    f"你是一个专业的图像编辑评估专家。\n"
                    f"请评估这张编辑后的图像的质量。\n\n"
                    f"编辑指令：{ground_truth}\n\n"
                    f"请从以下两个方面评分（每个方面 1-10 分）：\n"
                    f"1. 质量：图像的清晰度、真实感、是否有明显的伪影\n"
                    f"2. 匹配度：编辑结果是否符合指令要求\n\n"
                    f"返回格式：JSON {{"
                    f'"quality": <1-10>, "match": <1-10>, '
                    f'"reason": "<简短说明>"}}'
                )
            else:
                eval_prompt = (
                    f"你是一个专业的图像质量评估专家。\n"
                    f"请评估这张生成图像与提示词的匹配程度。\n\n"
                    f"提示词：{ground_truth}\n\n"
                    f"请从以下两个方面评分（每个方面 1-10 分）：\n"
                    f"1. 质量：图像的清晰度、真实感、是否有明显的伪影\n"
                    f"2. 匹配度：图像是否符合提示词描述\n\n"
                    f"返回格式：JSON {{"
                    f'"quality": <1-10>, "match": <1-10>, '
                    f'"reason": "<简短说明>"}}'
                )

            # 构建 vLLM 请求消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_base64},
                        },
                        {
                            "type": "text",
                            "text": eval_prompt,
                        },
                    ],
                }
            ]

            # 调用 vLLM API
            logger.info(f"评估图像 {image_idx + 1}/{len(solution_image)}")
            response_text = await _call_vllm_api(
                session=session,
                router_address=reward_router_address,
                messages=messages,
                model_name=model_name,
                temperature=0.3,
                max_tokens=256,
            )

            responses.append(response_text or "")

            if response_text:
                # 尝试从 JSON 中提取质量和匹配分数
                try:
                    json_match = response_text.find("{")
                    if json_match != -1:
                        json_end = response_text.rfind("}") + 1
                        if json_end > json_match:
                            json_str = response_text[json_match:json_end]
                            result = json.loads(json_str)
                            quality = float(result.get("quality", 5)) / 10.0
                            match = float(result.get("match", 5)) / 10.0
                            quality = min(1.0, max(0.0, quality))
                            match = min(1.0, max(0.0, match))
                            quality_scores.append(quality)
                            match_scores.append(match)
                            logger.info(f"  质量: {quality:.2f}, 匹配度: {match:.2f}")
                            continue
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass

                # 备用：使用关键词匹配
                logger.warning("无法解析 JSON，使用关键词匹配")
                quality_score = _extract_score_from_json(response_text) or 0.5
                quality_scores.append(quality_score)
                match_scores.append(quality_score)  # 使用相同的分数
            else:
                logger.error("vLLM API 调用失败，使用默认分数")
                quality_scores.append(0.5)
                match_scores.append(0.5)

    # 计算综合分数
    avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.5
    avg_match = float(np.mean(match_scores)) if match_scores else 0.5
    final_score = (avg_quality + avg_match) / 2.0  # 质量和匹配度均等权重

    logger.info(f"最终分数: {final_score:.2f} (质量: {avg_quality:.2f}, 匹配度: {avg_match:.2f})")

    return {
        "score": final_score,
        "quality_score": avg_quality,
        "match_score": avg_match,
        "avg_score": final_score,
        "num_images": len(quality_scores),
        "response": responses[0] if responses else "",
        "model": model_name,
    }


# 为了向后兼容，也导出同步包装版本
def compute_score_vllm_quality_sync(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, float | str | int]:
    """同步包装版本的 vLLM 奖励函数。

    在同步上下文中使用时使用此函数。系统会自动在线程池中执行。
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        compute_score_vllm_quality(
            data_source=data_source,
            solution_image=solution_image,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    )
