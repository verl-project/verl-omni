# AVQA graph profiles on Ascend

Last updated: 11/03/2026.

The graph-profile launcher wraps the existing full-parameter Qwen3-Omni AVQA
recipe. It selects chunked prefill, a 1536-token budget and capture sizes
`[1,16,128,1536]`. It preserves the base recipe's batch size, rollout count,
prompt/response limits, reward, validation and checkpoint settings. Caller Hydra
overrides remain last.

## Prerequisites

Activate your installed Python/CANN environment and provide model/data paths.
The launcher installs no packages and patches no dependencies. The base recipe
sources its CANN and ATB environment using `ASCEND_HOME_PATH`.

The default `PIECEWISE` mode is the baseline profile. To test FULL decode, set
`NPU_GRAPH_MODE=FULL_AND_PIECEWISE`. This mode still uses PIECEWISE for mixed
prefill/decode. It requires a compatible vllm-omni NPU runner that binds its
attention backend for FULL graph capture/replay. An accepted configuration string
does not prove that the backend supports correct FULL replay: check actual
capture logs and compare deterministic multimodal outputs before benchmarking.

The vllm-omni runner on current upstream main includes that binding; older pinned
revisions may require a backend patch. Do not upgrade only one dependency to
main without checking the vllm, vllm-ascend, PyTorch and Transformers constraints.

FSDP2 model offload must also preserve parameter values. Upstream
[verl#8187](https://github.com/verl-project/verl/pull/8187) fixes uneven-shard
corruption by using blocking device-to-CPU copies, including initialization.
Synchronizing only after an asynchronous `model.to()` has finished is not an
equivalent fix. A pinned-memory OOM fallback alone does not address that race.
[verl-omni#826](https://github.com/verl-project/verl-omni/pull/826) tracks delivery
of the corrected dependency pin; check its current status before selecting a
training stack.

These profiles do not add or replace the GMM/model patches tracked separately in
[verl-omni#819](https://github.com/verl-project/verl-omni/pull/819).

## Run matched profiles

Use absolute paths for model/data inputs. Replace the CANN path below with the
layout required by the base recipe:

```bash
export ASCEND_HOME_PATH=/path/to/cann/toolkit
export MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_FILE=/path/to/avqa/train.parquet
export VAL_FILE=/path/to/avqa/validation.parquet
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=8 ROLLOUT_TP=2

NPU_GRAPH_MODE=PIECEWISE bash \
  examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_avqa_npu_graph_v1.sh \
  ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  trainer.experiment_name=avqa_piecewise

# Run separately, after checking the FULL replay prerequisite above.
NPU_GRAPH_MODE=FULL_AND_PIECEWISE bash \
  examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_avqa_npu_graph_v1.sh \
  ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  trainer.experiment_name=avqa_full_decode
```

Change topology through both environment variables consistently. If caller Hydra
overrides change topology directly, keep the worker count consistent with the
final tensor-parallel and device counts. The wrapper validates its environment
defaults before starting the base recipe and propagates trainer failures through
the base recipe's logging pipeline.

## Record component versions

The collector uses only the Python standard library and does not import torch,
torch_npu, vllm or initialize an accelerator. Run it from the same activated
environment as training:

```bash
python scripts/collect_ascend_environment.py --output results/environment.json \
  --repo verl-omni="$PWD" \
  --repo verl=/path/to/verl \
  --repo vllm=/path/to/vllm \
  --repo vllm-ascend=/path/to/vllm-ascend \
  --repo vllm-omni=/path/to/vllm-omni \
  --version-file driver=/usr/local/Ascend/driver/version.info \
  --version-file toolkit=/path/to/toolkit/version.info
```

It records package metadata, duplicate dist-info versions, source commits,
tracked-diff hashes, selected environment flags and version/build fields from
explicitly selected CANN/driver files. It omits remotes, source diff contents and
installer user/path/token fields. Review version/build identifiers before sharing
manifests from a prerelease or internal stack.

Distribution metadata can differ from runtime build identifiers such as
`torch.__version__`. Keep runtime module identifiers and hardware/firmware records
alongside the manifest when those distinctions matter. A dirty repository also
needs its reviewed patch; a hash alone cannot reconstruct it. Untracked files are
reported by the dirty flag but excluded from the tracked-diff hash.

## Interpret results

Use the same prompt/response caps, batch size, rollout count, model/data files and
step interval. Record effective trainer and engine configurations, graph capture
and replay evidence, generated token counts, cache warmup and any compatibility
fallbacks. Compare training, rollout generation, actor update and weight-sync
times separately. V1 trainer validation runs outside its training-step timer;
retain validation time and total wall time separately.

For aggregate throughput use total valid prompt-plus-response tokens divided by
total training-step seconds and by the number of accelerators. This differs from
the arithmetic mean of per-step throughput. Report the first step and a warmed
window separately. Stochastic response lengths can differ despite identical caps;
do not shorten limits or silently discard slow steps to improve a comparison.

CPU launcher tests validate argument quoting, overrides, topology checks and exit
propagation. They do not establish FULL replay correctness, full-model performance
or convergence on a different dependency stack.
