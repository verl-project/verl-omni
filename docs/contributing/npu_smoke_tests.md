# NPU Smoke Tests

Last updated: 09/24/2026.

我们在 verl-omni 上增加基于华为昇腾设备的CI用例添加指导。

verl-omni 仓库使用 GitHub Actions 作为 CI 平台，通过分层测试架构保障代码质量与系统稳定性。
NPU 相关的工作流主要包括：

GitHub Actions 入口为
[`.github/workflows/npu_smoke.yml`](../../.github/workflows/npu_smoke.yml)，
本地统一入口为
[`tests/npu_smoke/run_npu_smoke_tests.sh`](../../tests/npu_smoke/run_npu_smoke_tests.sh)。

## 测试用例

| ID | 名称 | 测试入口 | 默认 NPU 数量 | 状态 |
|---|---|---|---:|---|
| 0 | vLLM-Omni rollout + sleep/wake-up | `tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py` | 8 | 启用 |
| 1 | Qwen-Image FlowGRPO trainer e2e | `tests/special_e2e/run_flowgrpo_qwen_image_npu.sh` | 8 | 暂时跳过 |

Test 1 仍保留在 runner 和 workflow 中，但
`run_npu_smoke_tests.sh` 当前通过 `RUN_TEST[1]=0` 强制将其标记为
`SKIP`。待 FlowGRPO 运行时问题解决后，删除该临时设置即可恢复。

## 增加一个新的 NPU Smoke 测试

### 1. 添加最小测试入口

根据测试范围选择文件位置：

| 测试类型 | 推荐位置 |
|---|---|
| rollout、worker、engine 或 sleep/wake-up | `tests/workers/` 下的 pytest |
| trainer 端到端流程 | `tests/special_e2e/` 下的 Shell 脚本 |
| NPU smoke 公共逻辑 | `tests/npu_smoke/` |

测试应尽量小且可重复：

- 使用 tiny-random 或 CI 已缓存的模型权重。
- 使用最少的数据、batch size 和训练步数。
- 不依赖运行期间从公网下载模型或数据集。
- 命令成功时返回 0，失败时保留原始非零退出码。
- 不复用其他任务遗留的 Ray 集群或 worker 进程。

例如，新增 pytest：

```python
def test_my_npu_feature():
    # Arrange the smallest NPU workload.
    ...
```

或者新增端到端脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

export DEVICE_NAME=npu
python -m verl_omni.trainer.main_ppo \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1
```

### 2. 在统一 runner 中注册测试

打开
[`run_npu_smoke_tests.sh`](../../tests/npu_smoke/run_npu_smoke_tests.sh)，
为新测试分配下一个未使用的数字 ID。

首先在 `RUN_TEST` 中注册 ID。假设新用例 ID 为 2：

```bash
declare -A RUN_TEST=([0]=1 [1]=1 [2]=1)
```

然后在测试执行区域添加 `run_selected_test`：

```bash
cleanup_runtime
run_selected_test 2 "my NPU smoke test" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
        NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/test_my_npu_feature.py
```

端到端脚本则写为：

```bash
cleanup_runtime
run_selected_test 2 "my trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
        NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_my_npu_smoke.sh
```

同时更新脚本的 `--help` 输出：

```text
Tests:
  0  vllm-omni rollout + sleep/wake_up
  1  FlowGRPO trainer e2e
  2  my NPU smoke test
```

不要复用已有 ID。测试名称应能说明覆盖的组件和行为。

### 3. 配置 NPU 数量与可见设备

统一 runner 默认使用 8 张 NPU，并根据 `--num-npus` 生成
`ASCEND_RT_VISIBLE_DEVICES`。测试命令必须传递：

```bash
env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
    NUM_NPUS="${NUM_NPUS}" \
    <test-command>
```

如果测试只能使用固定卡数，应在测试脚本中检查并给出清晰错误，而不是静默使用
错误的并行配置。

### 4. 更新 workflow 测试分组

如果新用例需要由现有标签触发，在
[`npu_smoke.yml`](../../.github/workflows/npu_smoke.yml) 的
`resolve-groups` 步骤中加入测试 ID。例如让完整模式执行 Test 2：

```yaml
if [[ "${mode}" == "all" ]]; then
  plan_json='{
    "test_ids": ["0", "1", "2"],
    "group_count": 3,
    "num_npus": 8,
    "runner_size": 8
  }'
fi
```

如需单独运行新测试，可以新增 mode 和 PR label，并在
`Resolve NPU CI mode` 与 `resolve-groups` 两处同时配置。

如果测试文件不在现有 workflow 的 path filters 中，也要添加对应路径。当前
主要过滤路径包括：

- `verl_omni/**`
- `tests/npu_smoke/**`
- `tests/workers/**`
- `tests/special_e2e/**`
- `pyproject.toml`
- `.github/workflows/npu_smoke.yml`
- `.github/vllm_omni_pin.txt`

### 5. 处理模型和数据

Qwen-Image tiny-random 权重默认放在：

```text
${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image
```

workflow 仅在目录不存在时构建：

```bash
MODEL_PATH="${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image"
if [[ ! -d "${MODEL_PATH}" ]]; then
  python tests/special_e2e/build_qwen_image_tiny_random.py \
    --output-dir "${MODEL_PATH}"
fi
```

新增模型时，优先使用 runner 已缓存的权重。若必须生成 tiny checkpoint，应：

- 仅在目标目录不存在时生成。
- 通过环境变量允许覆盖默认路径。
- 避免覆盖 CI 机器上的共享缓存。
- 在测试完成后只清理本次测试产生的临时文件。

### 6. 本地验证

在安装好 CANN、vLLM、vLLM-Ascend 和 vLLM-Omni 的 Ascend 环境中，
从仓库根目录运行测试。开始前可以先确认设备和停止残留的 Ray runtime：

```bash
npu-smi info
ray stop --force
```

运行默认测试集合：

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh
```

只运行 Test 0：

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 0
```

指定 NPU 数量运行 Test 0：

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 4 0
```

显式选择设备：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,2,4,6 NUM_NPUS=4 \
  bash tests/npu_smoke/run_npu_smoke_tests.sh 0
```

当前请求 Test 1 时，runner 会将其报告为 `SKIP`，不会启动 FlowGRPO：

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 1
```

新增 Test 2 后，可以单独运行：

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 2
```

测试日志默认保存在 `logs/npu_smoke/<timestamp>/`。先查看
`summary.log`，再根据失败 ID 查看对应的 `test_<id>.log`。

提交前至少检查：

```bash
bash -n tests/npu_smoke/run_npu_smoke_tests.sh
pre-commit run --all-files
```

## CI 触发方式

workflow 在相关文件发生变化时响应以下事件：

- push 到 `main` 或 `v0.*`。
- 针对 `main` 或 `v0.*` 的 pull request 被创建、同步、重新打开或添加标签。

PR 标签与请求范围：

| 标签 | 模式 | 请求的测试 |
|---|---|---|
| `ready-for-ci` | all | Test 0、Test 1 |
| `ci-npu` | all | Test 0、Test 1 |
| `ci-npu-rollout` | rollout | Test 0 |
| `ci-npu-flowgrpo` | flowgrpo | Test 1 |

Test 1 暂时禁用期间，即使 workflow 请求 Test 1，runner 也会将其报告为
`SKIP`。

## 运行环境

当前 CI 使用：

- Runner：`linux-aarch64-a2b4-8`
- NPU 数量：8
- 超时时间：120 分钟
- 共享内存：16 GiB
- 容器镜像：
  `swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-verl-omni-a2:latest`

测试开始前，workflow 会打印 CANN 安装信息、`npu-smi info`、Ascend
相关环境变量以及 vLLM、vLLM-Ascend、vLLM-Omni 版本。

## 进程清理与日志

每个测试开始前，统一 runner 会：

1. 执行 `ray stop --force`。
2. 终止残留的 `DiffusionWorker`、`VLLMWorker` 和
   `vLLMOmniHttpServer`。
3. 等待 5 秒后强制终止仍然存在的相关进程。
4. 打印 `npu-smi info`。

不要在同一台机器上同时运行名称匹配上述规则的其他任务。

日志默认写入：

```text
logs/npu_smoke/<timestamp>/
```

每个实际执行的用例会生成 `test_<id>.log`，`summary.log` 记录
`PASS`、`FAIL` 或 `SKIP` 以及用时。排查失败时应先检查测试日志、
NPU 显存占用、CANN 环境和 vLLM 组件版本。
