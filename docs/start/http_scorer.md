(http_scorer)=
# Using an External HTTP Scorer Service

Last updated: 08/09/2026

VeRL-Omni ships a generic HTTP reward client (`verl_omni.utils.reward_score.http_scorer_client`) that sends generated images to an external scorer service over HTTP and returns the score. This is useful when your reward model is too large to co-locate with training, needs a different runtime (e.g., a separate GPU pool), or is shared across multiple experiments.

## How it works

```text
┌──────────────┐        pickle payload         ┌──────────────────┐
│  VeRL-Omni   │  ──── POST (bytes) ────────►  │  Scorer Service  │
│  (training)  │                               │  (Flask/Gunicorn)│
│              │  ◄─── pickle response ──────  │                  │
└──────────────┘                               └──────────────────┘
```

1. During reward computation, the client converts the generated image tensor to JPEG bytes (offloaded to a thread pool to avoid blocking the async event loop).
2. The JPEG bytes and prompt are packed into a pickle payload and sent via HTTP POST.
3. The scorer service runs inference and returns scores in a pickle response.

Since `compute_score` is an `async` function and `RewardLoopWorker.compute_score_batch` uses `asyncio.gather`, all samples in a batch hit the server concurrently — no serial bottleneck.

## Scorer service protocol

The HTTP scorer client (`http_scorer_client.py`) communicates with external reward services using a pickle-based protocol, following the interface defined in [flow_grpo](https://github.com/yifan123/flow_grpo#3-reward-preparation). A reference implementation is available at [deepgen_rl/ocr_scorer_service](https://github.com/deepgenteam/deepgen_rl/tree/main/rewards_services/api_services/ocr_scorer_service).

### Request

The client sends a **POST** request with body = `pickle.dumps(payload)` where:

```python
payload = {
    "images": [bytes, ...],   # List of JPEG-encoded image bytes
    "prompts": [str, ...],    # List of prompt strings (same length as images)
    "metadata": {},           # Reserved for future use
}
```

### Response

The service must return `pickle.dumps(response)` where:

```python
# Success (HTTP 200):
response = {"scores": [float, ...]}  # One score per image, typically in [0, 1]

# Error (HTTP 200 with error key, or HTTP 5xx):
response = {"error": "description of what went wrong"}
```

Any service that implements this interface can be used as a reward function — PaddleOCR, HPSv3, aesthetic scorers, CLIP-based scorers, etc. The service runs independently and can use any framework (PaddlePaddle, PyTorch, ONNX, etc.) without conflicting with the training environment.

## Setting up a scorer service

A reference implementation is available at [deepgen_rl/ocr_scorer_service](https://github.com/deepgenteam/deepgen_rl/tree/main/rewards_services/api_services/ocr_scorer_service). Each service follows the same Flask + Gunicorn pattern:

```bash
# Clone and start the OCR scorer service
cd rewards_services/api_services/ocr_scorer_service
pip install -r requirements.txt
gunicorn -c gunicorn.conf.py 'app:create_app()'
```

The default port is configured in each service's `gunicorn.conf.py`. You can also write your own service — just implement the pickle-based protocol above (see [flow_grpo reward preparation](https://github.com/yifan123/flow_grpo#3-reward-preparation) for the specification).

## Configuring VeRL-Omni to use the HTTP scorer

In your training launch script, configure the reward function to point to `http_scorer_client` and pass the `server_url`:

```bash
python3 -m verl_omni.trainer.main_diffusion \
    ...
    "+reward.reward_functions.my_reward.path=pkg://verl_omni.utils.reward_score.http_scorer_client" \
    '+reward.reward_functions.my_reward.name=compute_score' \
    '+reward.reward_functions.my_reward.weight=1.0' \
    '+reward.reward_functions.my_reward.required=true' \
    "+reward.reward_functions.my_reward.server_url=http://<scorer-host>:<port>" \
    '+reward.reward_functions.my_reward.max_retries=2' \
    '+reward.reward_functions.my_reward.retry_backoff=0.5' \
    ...
```

Key points:

- **`path`**: Module path using the `pkg://` prefix.
- **`name`**: The async function to call (`compute_score`).
- **`weight`**: Reward weight when combining multiple reward functions.
- **`required`**: Controls what happens when this scorer fails. Set it to `true` to stop training. If it is `false` (the default), the reward manager sets the score to `0` and continues.
- **`server_url`**: Full URL of your scorer service (no trailing slash).
- **`max_retries`**: Number of retries after the initial attempt (default: `2`, for at most 3 attempts total).
- **`retry_backoff`**: Initial retry delay in seconds (default: `0.5`). The delay doubles after each failed attempt.

The client retries connection and response-transfer failures, timeouts, and HTTP 408/429/5xx responses. It fails immediately without retrying for other 4xx responses, HTTP 200 responses containing an `error` field, and malformed success responses. Once the retry budget is exhausted, the final error is passed to the reward manager, where `required` determines whether training stops or continues with a zero score.

Any extra key-value pairs added under the same reward function config are forwarded as `**kwargs` to `compute_score`.

## Full example

See the example script: `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_reward_server.sh`

This script trains Qwen-Image with FlowGRPO using an external OCR reward server.

```bash
# 1. Start external OCR reward server (separate process/machine)
# See: https://github.com/deepgenteam/deepgen_rl/tree/main/rewards_services/api_services/ocr_scorer_service
# The service interface follows: https://github.com/yifan123/flow_grpo#3-reward-preparation
cd rewards_services/api_services/ocr_scorer_service
bash run.sh  # Starts on port 19082

# 2. Prepare data (stores full prompt as ground_truth for HTTP service)
python examples/flowgrpo_trainer/data_process/qwenimage_ocr_http_service.py \
    --input_dir ~/dataset/ocr/ --output_dir ~/data/ocr_http

# 3. Run training
OCR_REWARD_SERVER_URL=http://<server-ip>:19082 \
    bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_reward_server.sh
```

## Notes

- The HTTP client reuses a single `aiohttp.ClientSession` across calls to avoid per-request connection overhead.
- Image serialization (tensor to PIL to JPEG) is offloaded to a thread pool via `asyncio.loop.run_in_executor` so it does not block the reward manager's async event loop.
- Each attempt has a 120-second timeout. With the default retry settings, a persistent retryable failure can make three attempts before it is reported.
- Reward-server profiling (`reward.reward_model.rollout.profiler.*`, see the [profiler guide](../perf/profiler.md)) only covers model-backed rewards served by VeRL-Omni. With an HTTP scorer, reward inference runs inside your external service — profile it there.
