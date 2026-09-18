# Monitor Training with RL-Insight

Last updated: 08/27/2026.

[RL-Insight](https://github.com/verl-project/rl-insight) provides online observability for RL training. In VeRL-Omni, it can receive trainer scalar metrics, vLLM-Omni rollout engine metrics, TransferQueue metrics, and rollout state traces, then display them in Grafana dashboards managed by the RL-Insight server.

## When to Use

Use RL-Insight when you want one monitoring view for:

- trainer metrics such as rewards, losses, and throughput
- vLLM-Omni rollout engine metrics across all replicas
- TransferQueue metrics when the V1 trainer is enabled
- rollout generation state timelines
- CPU, memory, network, and Ascend NPU hardware metrics

## Step 1: Install and Start RL-Insight

Install RL-Insight in the environment where the monitoring server runs. This integration requires
[RL-Insight commit `dd1d0a8`](https://github.com/verl-project/rl-insight/commit/dd1d0a8f6d3235989fd662264dcc96cd6e442be4)
or later. The following command pins the earliest compatible revision:

```bash
pip install "git+https://github.com/verl-project/rl-insight.git@dd1d0a8f6d3235989fd662264dcc96cd6e442be4"
```

Alternatively, clone the repository and install the latest source version:

```bash
git clone https://github.com/verl-project/rl-insight.git
cd rl-insight
pip install -e .
```

The Python package must also be installed in the VeRL-Omni training environment because trainer and rollout processes import the RL-Insight client when monitoring is enabled.

### Install monitoring services

`rl-insight server install` downloads Prometheus, Tempo, and Grafana into `~/.rl-insight/services`. The machine running this command needs access to GitHub release assets and `dl.grafana.com`.

For a machine with network access:

```bash
rl-insight server install
rl-insight server start
```

For an air-gapped or restricted cluster, download the three archives on a networked machine and copy them into one directory on the RL-Insight host. The default `linux-amd64` installer versions are:

| Service | Version | Download URL |
| --- | --- | --- |
| Prometheus | `2.54.1` | https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz |
| Tempo | `2.6.1` | https://github.com/grafana/tempo/releases/download/v2.6.1/tempo_2.6.1_linux_amd64.tar.gz |
| Grafana | `13.0.0` | https://dl.grafana.com/oss/release/grafana-13.0.0.linux-amd64.tar.gz |

For `linux-arm64`, replace `amd64` with `arm64` in the filenames and URLs. Tempo uses `linux_arm64` in its archive name. Keep the downloaded filenames unchanged.

Install and start the services from that directory:

```bash
rl-insight server install --local-archive /path/to/archives
rl-insight server start
```

`rl-insight server start` prints the detected server IP, Grafana URL, and related endpoints. Use that server IP in the training configuration below.

| Service | Default port | Purpose |
| --- | --- | --- |
| RL-Insight server | `18080` | Receives metrics and trace registrations |
| Prometheus | `9090` | Stores and queries metrics |
| Tempo | `3200` | Stores traces |
| Grafana | `3000` | Displays dashboards |

## Step 2: Enable RL-Insight in VeRL-Omni

Set the RL-Insight server URL before starting training. `<server-ip>` must be reachable from the controller and every Ray worker:

```bash
export RL_INSIGHT_SERVER_URL="http://<server-ip>:18080"
```

Add `rl_insight` to `trainer.logger`. VeRL-Omni then enables the RL-Insight client before Ray starts and propagates the setting to its worker processes.

### Diffusion training

```bash
python3 -m verl_omni.trainer.main_diffusion \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl_omni \
    trainer.experiment_name=flowgrpo_rl_insight \
    actor_rollout_ref.rollout.disable_log_stats=False \
    ...
```

### Diffusion V1 training

```bash
python3 -m verl_omni.trainer.main_diffusion_v1 \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl_omni \
    trainer.experiment_name=flowgrpo_v1_rl_insight \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    ...
```

### Omni training

```bash
python3 -m verl_omni.trainer.main_omni \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name=verl_omni \
    trainer.experiment_name=omni_rl_insight \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    ...
```

Trainer scalar metrics are automatically sent through the `rl_insight` logger backend. Keep `actor_rollout_ref.rollout.disable_log_stats=False` when monitoring rollout metrics. VeRL-Omni fails fast during rollout server initialization if RL-Insight is enabled while engine statistics are disabled.

## Step 3: Configure Multi-Node Training

For a multi-node Ray cluster, add the RL-Insight server address to the runtime environment file submitted with the job:

```yaml
env_vars:
  RL_INSIGHT_SERVER_URL: "http://<server-ip>:18080"
```

If the launch command uses another file through `ray job submit --runtime-env`, add the variable to that file instead. Do not use `localhost` unless the RL-Insight server runs inside every worker node; use an IP or hostname reachable across the cluster.

The `trainer.logger` setting automatically propagates `VERL_RL_INSIGHT_ENABLE=1` to Ray actors. Users should set `RL_INSIGHT_SERVER_URL`, but do not need to set `VERL_RL_INSIGHT_ENABLE` manually.

## Step 4: Monitor Rollout and TransferQueue Metrics

Rollout engine statistics are exposed by each vLLM-Omni server and registered with RL-Insight. Each rollout replica uses a separate lane such as `replica_0` or `replica_1` for `vllm_generate`, `vllm_sleep`, `vllm_wake_up`, `vllm_release_kv_cache`, and `vllm_resume_kv_cache` traces. Actor workers emit `update_weights` on their own `rank_<rank>` lanes.

The following setting is required for rollout metrics:

```bash
actor_rollout_ref.rollout.disable_log_stats=False
```

For V1 training, expose the TransferQueue metrics endpoint with:

```bash
transfer_queue.metrics.enabled=True
```

This is independent of `transfer_queue.enable`: the V1 entrypoint enables TransferQueue itself, while `transfer_queue.metrics.enabled` controls whether its Prometheus endpoint is exposed and registered.

## Step 5: Add Hardware Metrics (Optional)

To monitor CPU, memory, network, or Ascend NPU metrics, follow the [RL-Insight Hardware Monitoring guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/hardware/index.md). It explains how to install or reuse exporters and register their endpoints with RL-Insight.

Hardware monitoring is configured on the monitored nodes and RL-Insight server. It is not enabled through the VeRL-Omni trainer arguments above.

## View Dashboards

1. Open the Grafana URL printed by `rl-insight server start`. The default is `http://<server-ip>:3000`.
2. Log in with the default username `admin` and password `admin`.
3. Open **Dashboards** from the left navigation, then open the **RL-Insight** folder.
4. Select the dashboard matching the rollout backend and trainer mode.
5. While training is running, select a recent time range such as **Last 5 minutes** or **Last 15 minutes**.

The dashboards can include trainer metrics, rollout metrics for every replica, TransferQueue metrics, rollout state timelines, and optional hardware metrics.

Example views from the RL-Insight integration:

**RL state timeline (synchronous training)**

![Synchronous RL state timeline](https://github.com/mengchengTang/verl-data/raw/master/sync_timeline.png)

**RL state timeline (separate asynchronous training)**

![Separate asynchronous RL state timeline](https://github.com/mengchengTang/verl-data/raw/master/separate_async_timeline.png)

**Inference engine metrics across replicas**

![Inference engine metrics](https://github.com/mengchengTang/verl-data/raw/master/infer_engine_metric_of_all_replica.png)

**TransferQueue metrics**

![TransferQueue metrics](https://github.com/mengchengTang/verl-data/raw/master/transfer_queue_metric.png)

## Troubleshooting

### Trainer metrics do not appear

- Verify that `trainer.logger` contains `rl_insight`.
- Verify that `RL_INSIGHT_SERVER_URL` points to the machine running `rl-insight server start`.
- Confirm that `rl-insight` is installed in the training Python environment.

### Rollout metrics do not appear

- Set `actor_rollout_ref.rollout.disable_log_stats=False`.
- Confirm that the rollout server addresses printed by `LLMServerManager` are reachable from the RL-Insight/Prometheus host.
- Check that every rollout replica uses a unique reachable metrics endpoint.

### TransferQueue metrics do not appear

- Use the V1 trainer.
- Set `transfer_queue.metrics.enabled=True`.
- Confirm that the TransferQueue metrics endpoint is reachable from the RL-Insight/Prometheus host.

### Traces are missing but scalar metrics are visible

- Confirm that `VERL_RL_INSIGHT_ENABLE=1` is present in the rollout server process environment.
- For Ray jobs, check that the training entrypoint initialized the current cluster rather than attaching to a cluster that was started without the required runtime environment.
- Check Tempo availability on port `3200` from the RL-Insight server.

### Multi-node workers cannot connect

- Put `RL_INSIGHT_SERVER_URL` in the runtime environment passed to `ray job submit`.
- Use the monitoring host's cluster-reachable IP or hostname instead of `127.0.0.1` or `localhost`.
- Check firewall rules for the RL-Insight, Prometheus, Tempo, and Grafana ports.

### Service installation cannot download packages

Use `rl-insight server install --local-archive /path/to/archives` with the three archives described in Step 1.

For additional server installation and operation details, see the [RL-Insight server installation guide](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/server_installation.md) and [quick start](https://github.com/verl-project/rl-insight/blob/main/docs/monitor/quick_start.md).
