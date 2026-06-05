# 在 VeRL-Omni 中使用自定义 vLLM 奖励函数

Last updated: 06/05/2026

本指南说明如何在 VeRL-Omni 中集成自定义的 vLLM 奖励函数进行训练。

## 快速开始

### 1. 编写自定义奖励函数

在 `verl_omni/utils/reward_score/` 目录下创建你的奖励函数文件，例如 `my_custom_reward.py`：

```python
# verl_omni/utils/reward_score/my_custom_reward.py

import asyncio
import json
from typing import Optional

import aiohttp
import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizer


async def _call_vllm_api(
    router_address: str,
    messages: list,
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
) -> str:
    """异步调用 vLLM OpenAI 兼容 API。
    
    Args:
        router_address: vLLM 服务地址，格式为 "host:port"
        messages: OpenAI 格式的消息列表
        model_name: 模型名称
        temperature: 采样温度
        top_p: nucleus 采样参数
        max_tokens: 最大生成 token 数
        
    Returns:
        生成的文本内容
    """
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    
    request_data = {
        "messages": messages,
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=request_data) as resp:
            response = await resp.text()
            result = json.loads(response)
            return result["choices"][0]["message"]["content"]


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


async def compute_score_my_reward(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
):
    """自定义奖励函数：使用 vLLM 评估图像质量。
    
    Args:
        data_source: 数据源标识
        solution_image: 生成的图像，形状为 (C, H, W) 或 (N, C, H, W)
        ground_truth: 参考文本（如提示词）
        extra_info: 额外信息字典
        reward_router_address: vLLM 路由地址 "host:port"
        reward_model_tokenizer: 分词器（可选）
        model_name: 模型名称（从配置获取）
        
    Returns:
        dict: 包含 "score" 键和其他元数据
    """
    from verl.utils.ray_utils import get_event_loop
    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64
    
    # 确保图像是 4D 张量 (N, C, H, W)
    if solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)
    
    loop = get_event_loop()
    scores = []
    
    for image in solution_image:
        # 转换为 PIL 图像并编码为 base64
        pil_image = _to_pil(image)
        image_base64 = await loop.run_in_executor(None, pil_image_to_base64, pil_image)
        
        # 构建 vLLM 请求
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的图像质量评估专家。请评估生成图像的质量。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    },
                    {
                        "type": "text",
                        "text": f"请评估这张图像与提示词 '{ground_truth}' 的匹配程度。" + 
                                "回复格式：JSON {\"score\": 0.0-1.0, \"reason\": \"原因\"}"
                    }
                ]
            }
        ]
        
        # 调用 vLLM API
        response_text = await _call_vllm_api(
            router_address=reward_router_address,
            messages=messages,
            model_name=model_name or "qwen-vl-max",
            temperature=0.3,
            max_tokens=512,
        )
        
        # 从响应中提取分数
        try:
            # 尝试解析 JSON 响应
            json_match = response_text.find("{")
            if json_match != -1:
                json_str = response_text[json_match:]
                json_str = json_str[:json_str.rfind("}")+1]
                result = json.loads(json_str)
                score = float(result.get("score", 0.5))
                score = min(1.0, max(0.0, score))  # 裁剪到 [0, 1]
            else:
                score = 0.5
        except (json.JSONDecodeError, ValueError, KeyError):
            # 备用：使用字符串匹配
            if "很好" in response_text or "excellent" in response_text.lower():
                score = 0.8
            elif "一般" in response_text or "fair" in response_text.lower():
                score = 0.5
            else:
                score = 0.3
        
        scores.append(score)
    
    final_score = np.mean(scores)
    
    return {
        "score": float(final_score),
        "num_images": len(scores),
        "response": response_text,
    }
```

### 2. 启动 vLLM 服务

在训练前，启动一个 vLLM 推理服务：

```bash
# 启动 vLLM 服务（例如使用 Qwen-VL-Max）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-VL-7B-Chat \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    --served-model-name qwen-vl-7b
```

或者用脚本包装（`scripts/start_vllm_custom_reward.sh`）：

```bash
#!/bin/bash
# scripts/start_vllm_custom_reward.sh

MODEL=${1:-"Qwen/Qwen-VL-7B-Chat"}
PORT=${2:-8000}
TP_SIZE=${3:-2}

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.9 \
    --port $PORT \
    --served-model-name qwen-vl-7b
```

### 3. 在训练脚本中配置

在训练脚本中指定自定义奖励函数：

```bash
# examples/my_nft_training.sh

WORKSPACE=${WORKSPACE:-$HOME}
train_path=$WORKSPACE/data/train.parquet
test_path=$WORKSPACE/data/test.parquet

# vLLM 奖励服务地址
REWARD_ROUTER_ADDRESS="127.0.0.1:8000"

# 自定义奖励函数路径和名称
custom_reward_path="verl_omni/utils/reward_score/my_custom_reward.py"
custom_reward_name="compute_score_my_reward"

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=16 \
    algorithm.trainer_type=direct_preference \
    actor_rollout_ref.model.algorithm=nft \
    actor_rollout_ref.model.path=Qwen/Qwen-Image-Edit-2511 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=nft \
    \
    # 关键：配置自定义奖励函数
    reward.custom_reward_function.path=$custom_reward_path \
    reward.custom_reward_function.name=$custom_reward_name \
    reward.num_workers=4 \
    \
    # vLLM 奖励模型配置（用于异步调用）
    reward.reward_model.enable=True \
    reward.reward_model.model_path="qwen-vl-7b" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=2 \
    reward.reward_model.rollout.gpu_memory_utilization=0.9 \
    \
    trainer.experiment_name=my_custom_reward_training \
    "$@"
```

## 配置参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `reward.custom_reward_function.path` | 奖励函数文件路径 | `verl_omni/utils/reward_score/my_custom_reward.py` |
| `reward.custom_reward_function.name` | 函数名称（async 或 sync） | `compute_score_my_reward` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `reward.num_workers` | 并行评分的 worker 数 | 8 |
| `reward.reward_model.enable` | 是否启用奖励模型服务 | False |
| `reward.reward_model.model_path` | 模型名称（传给奖励函数） | null |
| `reward.reward_model.rollout.name` | 推理后端 | vllm |
| `reward.reward_model.rollout.tensor_model_parallel_size` | 张量并行大小 | 2 |

## 奖励函数接口规范

### 异步函数（推荐用于 vLLM 调用）

```python
async def compute_score_my_reward(
    data_source: str,                    # 数据源标识
    solution_image: np.ndarray | torch.Tensor,  # 生成图像 (C,H,W) 或 (N,C,H,W)
    ground_truth: str,                   # 参考文本/提示词
    extra_info: dict,                    # 额外信息（如 num_turns, rollout_reward_scores）
    reward_router_address: str,          # vLLM 服务地址 "host:port"
    reward_model_tokenizer: PreTrainedTokenizer = None,  # 分词器（可选）
    model_name: Optional[str] = None,    # 模型名称（从配置获取）
) -> dict | float:
    """
    返回值：
    - dict: {"score": float, ...其他字段}
    - float: 直接返回分数
    """
    pass
```

### 同步函数

```python
def compute_score_my_reward(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict | float:
    """同步奖励函数（自动在线程池中执行）"""
    pass
```

## 完整工作流示例

### 步骤 1：创建奖励函数

```python
# verl_omni/utils/reward_score/quality_scorer.py

import asyncio
import json
import aiohttp
import torch
import numpy as np
from PIL import Image


async def compute_score_quality(
    data_source: str,
    solution_image: torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    model_name: str = None,
    **kwargs,
):
    """评估生成图像的质量和清晰度。"""
    from verl.utils.ray_utils import get_event_loop
    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64
    
    if solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)
    
    loop = get_event_loop()
    scores = []
    
    for image in solution_image:
        # 图像转 base64
        pil_img = Image.fromarray(
            (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        img_base64 = await loop.run_in_executor(
            None, pil_image_to_base64, pil_img
        )
        
        # 调用 vLLM 评估
        url = f"http://{reward_router_address}/v1/chat/completions"
        
        payload = {
            "model": model_name or "qwen-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base64}},
                        {"type": "text", "text": f"按 1-10 评分图像质量，仅回复数字。提示词：{ground_truth}"}
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 10,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                text = result["choices"][0]["message"]["content"].strip()
                
                # 提取分数
                try:
                    score_int = int("".join(c for c in text if c.isdigit())[:2])
                    score = min(10, max(1, score_int)) / 10.0  # 归一化到 [0, 1]
                except:
                    score = 0.5
                
                scores.append(score)
    
    return {
        "score": float(np.mean(scores)),
        "individual_scores": scores,
    }
```

### 步骤 2：启动 vLLM 服务

```bash
# 终端 1：启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-VL-7B-Chat \
    --tensor-parallel-size 2 \
    --port 8000 \
    --served-model-name qwen-vl
```

### 步骤 3：运行训练

```bash
# 终端 2：运行训练
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=/path/to/train.parquet \
    algorithm.trainer_type=direct_preference \
    actor_rollout_ref.model.algorithm=nft \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/quality_scorer.py \
    reward.custom_reward_function.name=compute_score_quality \
    reward.num_workers=4 \
    trainer.experiment_name=quality_guided_nft \
    "$@"
```

## 常见问题

### Q1：我的奖励函数报错 `aiohttp.ClientConnectError`

**A：** vLLM 服务未启动或地址错误。检查：
- vLLM 服务是否运行：`curl http://127.0.0.1:8000/v1/models`
- 配置的地址是否正确（应为 `host:port`，不含 `http://`）

### Q2：奖励分数总是相同的值

**A：** 可能是：
1. 奖励函数中的异常被吞掉了 — 添加日志
2. vLLM 返回的格式与解析逻辑不符 — 调整 JSON 解析
3. 模型输出不稳定 — 降低 `temperature` 参数

### Q3：如何使用本地模型而不是 API？

**A：** 直接在奖励函数中加载模型：

```python
async def compute_score_local_model(
    data_source, solution_image, ground_truth, extra_info, **kwargs
):
    """使用本地加载的模型评分。"""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    
    model = AutoModelForImageClassification.from_pretrained("your-model-path")
    processor = AutoImageProcessor.from_pretrained("your-model-path")
    
    # ... 处理图像 ...
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    score = outputs.logits.softmax(-1)[0, 1].item()  # 假设 class 1 是"好"
    return {"score": score}
```

### Q4：训练期间内存不足

**A：** 减少以下参数：
- `reward.num_workers` — 减少并行评分的 worker 数
- `reward.reward_model.rollout.gpu_memory_utilization` — 从 0.9 改为 0.5
- 增加 `reward.reward_model.rollout.max_num_seqs` 缩小 batch size

### Q5：如何同时使用多个奖励函数？

**A：** 使用 multi-reward 配置：

```bash
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=/path/to/train.parquet \
    reward.reward_functions.quality.path=verl_omni/utils/reward_score/quality.py \
    reward.reward_functions.quality.name=compute_score_quality \
    reward.reward_functions.quality.weight=0.6 \
    reward.reward_functions.diversity.path=verl_omni/utils/reward_score/diversity.py \
    reward.reward_functions.diversity.name=compute_score_diversity \
    reward.reward_functions.diversity.weight=0.4 \
    reward.aggregation=weighted_sum \
    "$@"
```

## 性能优化

### 1. 异步并行化

使用 `async` 函数可以并行处理多张图像：

```python
async def compute_score_batch(
    data_source, solution_images, ground_truth, extra_info, 
    reward_router_address, **kwargs
):
    """批量评分多张图像。"""
    tasks = [
        _score_single_image(img, reward_router_address)
        for img in solution_images
    ]
    scores = await asyncio.gather(*tasks)
    return {"score": np.mean(scores)}
```

### 2. 缓存 vLLM 连接

复用 HTTP 连接而不是每次创建新的：

```python
import aiohttp

# 全局连接
_session = None

async def get_session():
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
    return _session

async def compute_score_cached(
    data_source, solution_image, ground_truth, extra_info, 
    reward_router_address, **kwargs
):
    session = await get_session()
    # ... 使用 session ...
```

### 3. 批量请求

如果 vLLM 支持，可以批量发送请求：

```python
async def compute_score_batch_api(
    data_source, solution_images, ground_truth, extra_info,
    reward_router_address, **kwargs
):
    """一次 API 调用评估多张图像。"""
    # 构建批量请求...
    pass
```

## 参考资源

- [VeRL-Omni 奖励系统文档](../../reward_loop/README.md)
- [vLLM 部署指南](https://docs.vllm.ai/en/latest/)
- [OpenAI API 兼容性](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Flow-Factory 奖励模型](../../../third_party/flow-factory/src/flow_factory/rewards/)
