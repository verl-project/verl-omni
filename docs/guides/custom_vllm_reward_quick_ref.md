# VeRL-Omni 自定义 vLLM 奖励函数 — 快速参考

Last updated: 06/05/2026

## 三个关键配置参数

在 CLI 中指定自定义奖励函数只需要这三个参数：

```bash
reward.custom_reward_function.path=<文件路径>        # 奖励函数所在的 Python 文件
reward.custom_reward_function.name=<函数名>          # 函数名称（async 或 sync）
reward.num_workers=<并行 worker 数>                  # 默认值：8
```

## 最小化完整工作流

### 1. 编写奖励函数（复制此模板）

创建 `my_reward.py`：

```python
import aiohttp
import json
from typing import Optional
import torch

async def my_vllm_reward(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str = "127.0.0.1:8000",
    model_name: str = "qwen-vl",
    **kwargs
):
    """简单的 vLLM 奖励函数示例"""
    from verl.utils.ray_utils import get_event_loop
    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64
    from PIL import Image
    import numpy as np
    
    # 确保图像是 4D
    if solution_image.ndim == 3:
        solution_image = solution_image.unsqueeze(0)
    
    loop = get_event_loop()
    scores = []
    
    async with aiohttp.ClientSession() as session:
        for image in solution_image:
            # 转换图像为 base64
            pil_img = Image.fromarray(
                (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            )
            img_b64 = await loop.run_in_executor(
                None, pil_image_to_base64, pil_img
            )
            
            # 调用 vLLM
            url = f"http://{reward_router_address}/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_b64}},
                        {"type": "text", "text": f"按 1-10 评分。提示词：{ground_truth}"}
                    ]
                }],
                "temperature": 0.3,
                "max_tokens": 10,
            }
            
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                text = result["choices"][0]["message"]["content"]
                
                # 提取分数
                import re
                match = re.search(r'\d+', text)
                score = int(match.group()) / 10 if match else 0.5
                scores.append(min(1.0, score))
    
    return {"score": float(np.mean(scores))}
```

### 2. 启动 vLLM

```bash
# 单个 GPU
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-VL-7B-Chat \
    --port 8000

# 多个 GPU（张量并行）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen-VL-7B-Chat \
    --tensor-parallel-size 2 \
    --port 8000
```

### 3. 运行训练

```bash
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=/path/to/train.parquet \
    algorithm.trainer_type=direct_preference \
    actor_rollout_ref.model.algorithm=nft \
    \
    # ← 关键三行
    reward.custom_reward_function.path=my_reward.py \
    reward.custom_reward_function.name=my_vllm_reward \
    reward.num_workers=4 \
    \
    trainer.experiment_name=my_experiment \
    "$@"
```

## 完整生产级例子

使用预实现的高级奖励函数 `vllm_quality_reward.py`：

```bash
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    data.train_batch_size=16 \
    algorithm.trainer_type=direct_preference \
    algorithm.adv_estimator=nft \
    actor_rollout_ref.model.algorithm=nft \
    actor_rollout_ref.model.path=Qwen/Qwen-Image-Edit-2511 \
    \
    # 使用 vllm_quality_reward
    reward.custom_reward_function.path=verl_omni/utils/reward_score/vllm_quality_reward.py \
    reward.custom_reward_function.name=compute_score_vllm_quality \
    reward.num_workers=4 \
    \
    trainer.experiment_name=nft_with_vllm_reward \
    trainer.total_epochs=10 \
    "$@"
```

## API 接口

### 异步函数（推荐）

```python
async def compute_score_xxx(
    data_source: str,                       # 数据源 ID
    solution_image,                         # 生成图像 (C,H,W) 或 (N,C,H,W)
    ground_truth: str,                      # 参考文本
    extra_info: dict = None,                # 额外信息
    reward_router_address: str = None,      # vLLM 服务地址
    reward_model_tokenizer = None,          # 分词器（可选）
    model_name: str = None,                 # 模型名称
    **kwargs
) -> dict:
    """
    返回必须包含 "score" 字段的字典：
    {
        "score": 0.8,           # 必需
        "reason": "...",        # 可选
        "debug_info": {...}     # 可选
    }
    """
    pass
```

### 同步函数

```python
def compute_score_xxx(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> dict:
    """同步版本（会在线程池中运行）"""
    pass
```

## 调试技巧

### 检查 vLLM 是否运行

```bash
curl http://127.0.0.1:8000/v1/models
```

### 手动测试奖励函数

```python
import asyncio
from my_reward import my_vllm_reward
import torch

image = torch.randn(3, 512, 512)
result = asyncio.run(my_vllm_reward(
    data_source="image_edit",
    solution_image=image,
    ground_truth="test prompt",
    extra_info={},
    reward_router_address="127.0.0.1:8000",
))
print(result)
```

### 查看训练日志

```bash
# 实时看日志
tail -f checkpoints/diffusion_nft/my_experiment/training.log

# 搜索奖励相关日志
grep -i "reward\|score" checkpoints/diffusion_nft/my_experiment/training.log
```

## 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| `aiohttp.ClientConnectError` | vLLM 未运行 | 启动 vLLM 服务 |
| `Cannot import module` | 文件路径错误 | 检查 path 是否正确且相对于项目根目录 |
| `score is None` | JSON 解析失败 | 添加日志打印响应内容 |
| `CUDA OOM` | 奖励模型内存不足 | 降低 reward.num_workers |
| `TypeError: cannot unpack` | 返回值格式错误 | 确保返回 dict 且有 "score" 键 |

## 多奖励函数

使用多个奖励函数的配置：

```bash
python3 -m verl_omni.trainer.main_diffusion \
    ... \
    # 定义多个奖励函数
    reward.reward_functions.quality.path=verl_omni/utils/reward_score/vllm_quality_reward.py \
    reward.reward_functions.quality.name=compute_score_vllm_quality \
    reward.reward_functions.quality.weight=0.7 \
    \
    reward.reward_functions.diversity.path=my_diversity_reward.py \
    reward.reward_functions.diversity.name=my_diversity_score \
    reward.reward_functions.diversity.weight=0.3 \
    \
    reward.aggregation=weighted_sum \
    ... \
    "$@"
```

## 性能优化

### 1. 异步并行处理

确保函数是 `async`，可以并行处理多个图像：

```python
async def my_reward(...):
    async with aiohttp.ClientSession() as session:
        tasks = [process_image(img) for img in images]
        results = await asyncio.gather(*tasks)
    return results
```

### 2. 连接复用

使用共享的 aiohttp Session 而不是每次创建新的。

### 3. 批量请求

如果 vLLM 支持，批量发送多个请求。

## 实现示例

- **VQA 评分**：`vllm_quality_reward.py`（已提供）
- **OCR 评分**：`genrm_ocr.py`
- **JPEG 压缩性**：`jpeg_compressibility.py`

## 更多资源

- 完整文档：`docs/guides/custom_vllm_reward.md`
- Flow-Factory 奖励模型：`third_party/flow-factory/src/flow_factory/rewards/`
- vLLM 文档：https://docs.vllm.ai/
