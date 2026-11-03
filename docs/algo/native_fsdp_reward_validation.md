# Native FSDP reward validation

Last updated: 11/03/2026.

## Acceptance boundary

Functional acceptance applies to the exact configurations below. Single-replica PickScore, multiple single-node
replica groups, CLIPScore and CPU/Gloo are separate changes. Streaming and shared-GPU training are independent
integrations. A pass on one integration does not establish another integration's acceptance.

The [RFC](native_fsdp_reward_rfc.md) defines execution, lifecycle, failure and configuration contracts.
The tested runtime used Python 3.12, PyTorch 2.13.0+cu130, Ray 2.59.0, Transformers 5.14.1 and Diffusers 0.41.0.
The pinned verl revision was `fefb080262e1c015a0ea05f958822a6a512dc795`; vLLM-Omni was
`12e9280010afc41ce5c6a1f46900fc329cf08377`. CPU/Gloo acceptance is specific to this PyTorch stack.

## Single-replica foundation

The original upstream integration used `main` at `aaaea177e80dd55b6d90a3d83ef08984bafc3a46`, followed by the
single-replica implementation published at `6f1c14d1ab7eb1ee37fdc34a60b0a6c49ae3b18e`.

| Check | Result |
| --- | --- |
| Related CPU regressions | 1571 passed, including real-Ray CPU-native tests. |
| Pretrained PickScore v1 | 20 load/score/unload cycles; maximum score absolute error `1.7881393432617188e-7`. |
| Rank/NCCL faults | Ten actual cases, including missing/corrupt weights, constructor/forward errors, rank exits, a held rank and an NCCL watchdog timeout. |
| Facade faults | Four actual exit/SIGSTOP cases while idle or scoring; independent owner confirmed facade/rank termination. |
| Dedicated-pool training | Real SD3.5 Medium FlowGRPO, rank-4 LoRA, 128×128, four denoising steps, two updates, validation/checkpoints each step. |
| Training tensors | Adam steps 1→2; 484 changed LoRA tensors; gradient norms `0.11865234375` and `0.1669921875`. |
| Cleanup | Owned reward and queue actors confirmed dead before Ray shutdown. |

A separate earlier development baseline completed ten updates at 256×256/eight denoising steps, resume 10→11,
resident execution and independent-venv checks. Those historical runs are not executions of the later extension
candidates and do not establish their training acceptance.

## Independent inference extensions

### Single-node replica groups

Two replicas use four actual RTX A6000 GPUs and parent-pool groups `[[3,1],[0,2]]`. Each group has independent
rank order, process group and facade; each pair's local DTensor element counts sum to the complete model.
Production dispatch checks preserve sample order across batches 1/3/16/17, reordered/repeated samples and long text.

Actual rank or facade death invalidates the entire deployment and terminates both groups before Ray shutdown.
A CPU contract test covers failure while constructing a second group after a first group exists. The equivalent
real multi-group constructor-failure/rollback case remains a GPU acceptance gate.

### CLIPScore and PickScore parity

| Configuration | Maximum raw-feature absolute error | Maximum score absolute error |
| --- | ---: | ---: |
| Two-GPU PickScore, 20 cycles | Recorded by validator | `1.7881393432617188e-7` |
| Two-GPU CLIP ViT-B/32, 20 cycles | `3.3080577850341797e-6` | `1.4901161193847656e-7` |
| Four-GPU CLIP ViT-B/32, two groups, 20 cycles | `3.814697265625e-6` | `2.980232238769531e-7` |
| Four-GPU PickScore, two groups | `5.900859832763672e-6` | `1.1920928955078125e-7` |

These historical extension runs used production named deployment, rank execution and existing scorer formulas.
CLIP has 151,277,313 parameters; PickScore has 986,109,441. Real OpenAI CLIP projection layouts exercised the
adapter's non-contiguous frozen-parameter normalization. GPU parity inputs came from real diffusion-training images.
Four source images were reused to form 17 samples; they were not 17 independent rollouts.
Assertions use `rtol=1e-4, atol=2e-5` on raw outputs and scores. Observed errors do not redefine scorer tolerances.

### CPU/Gloo

The real CPU validator uses four Ray rank actors, Gloo collectives, FSDP2/DTensor parameters, production named
scoring and the existing unsharded reference. It reserves zero GPUs and checks actual actor death before shutdown.

| Model / case | Result |
| --- | --- |
| Pretrained CLIP ViT-B/32, two groups, two cycles | Maximum score error `2.980232238769531e-7`. |
| Pretrained PickScore v1, two groups, one cycle | Maximum score error `1.1920928955078125e-7`. |
| Tiny CLIP, 20 cycles and actual rank exit | Correct score parity, deployment invalidation and complete termination. |
| Inherited accelerator visibility override | Real zero-GPU execution passed with the parent override enabled. |

CPU image inputs are deterministic RGB tensors. Tiny-model execution validates mechanisms; pretrained runs validate
actual checkpoint loading and model parity. Neither establishes CPU diffusion-training or NPU support.

## Candidate verification

The split candidates are checked independently using their own source snapshots. The combined candidate additionally
checks feature interaction. Test and execution manifests identify source/helper hashes, commands and results;
historical tests retain their original execution identities.

| Candidate | Verification on its own source |
| --- | --- |
| Foundation update | Ten actual GPU faults, 20 residency cycles, infer/sleep and infer/close drain, actual five-second NCCL all-gather watchdog; all 26 owned ranks dead. |
| Replica groups | 82 focused CPU tests; real four-GPU PickScore, two cycles, two groups, facade exit and complete termination. Score error `1.1920928955078125e-7`. |
| CLIPScore | 89 focused CPU tests; real four-GPU CLIP on this isolated source, 20 cycles and actual rank exit. Raw error `3.814697265625e-6`, score error `2.980232238769531e-7`; all actors dead. |
| CPU/Gloo | 97 focused CPU tests; real pretrained four-rank CLIP, two cycles and actual rank exit. Score error `2.980232238769531e-7`; all actors dead. |
| Combined inference candidate | 1597 related CPU tests; real four-GPU CLIP, two groups, 20 cycles and actual rank exit. Raw error `3.814697265625e-6`, score error `2.980232238769531e-7`; all actors dead. |

All 12 repository pre-commit hooks passed on the foundation update, the three feature snapshots and the combined candidate, including generated-config verification.

All accelerator checks require actual idle devices and confirm their physical UUIDs. Manual validators write
results outside the repository. A skipped or mocked accelerator path is not counted as an accelerator pass.

## Independent training integrations

These results concern the separately implemented integrations; their code is outside the single-replica PR.
All used real pretrained SD3.5 Medium FlowGRPO, rank-4 LoRA, Pick-a-Pic-derived prompts, two prompts × two responses,
256×256 images, eight denoising steps, validation/checkpoints, and checkpoint 2→3 continuation.

| Integration | Reward | Actor ranks | Changed LoRA tensors | Resumed gradient norm |
| --- | --- | ---: | ---: | ---: |
| Shared two-GPU pool, one reward group | PickScore + JPEG | 2 | 968 | `0.01904296875` |
| Shared four-GPU pool, two reward groups | PickScore + JPEG | 4 | 1935 | `0.018310546875` |
| Dedicated resident streaming, one reward group | CLIPScore + JPEG | 1 | 484 | `0.0302734375` |

Every actor rank had finite model/Adam state. Supplemental four-rank checkpoint inspection found 484/484/483/484
changed LoRA tensors, zero changed frozen tensors on every rank, and exact Adam steps 2→3 on every rank.
Shared-phase observation confirmed all reward ranks had zero local parameter elements before actor update;
unloaded allocations were 8,519,680 bytes per rank.

Streaming additionally resumed 3→4 with `max_inflight=2`. Six real generation requests, including validation,
had peak concurrency two in the single-agent-worker run. Ray timelines showed generation-request/reward-inference
overlap. This establishes neither a deployment-wide bound with multiple clients nor GPU-kernel overlap/speedup.

Actual streaming cancellation, timeout and close drained held forwards; rank/facade exit and SIGSTOP terminated
the deployment. Shared training OOM, missing weights and rank exit between scoring and unload produced zero
actor-update entries, preserved the latest checkpoint and terminated owned actors. Those negative training runs
covered one reward group; the two-group training fault path remains unaccepted.

The actual GPU fault matrix also passed ten cases, 20 residency cycles and infer/sleep/close draining. A rank's
stderr confirmed the five-second NCCL all-gather watchdog; Ray exposed a generic actor exit. All 26 owned rank
actors were dead before Ray shutdown. Test-only injection is absent from production configuration.

## Reproduction

Install the project's pinned dependencies. Run from the repository root and supply local immutable checkpoints,
processor directories, actual generated-media JSONL and output directories. The general GPU/CPU validators belong
to their explicit acceptance-tooling/CPU increments; the foundation supplies its PickScore validator.

```bash
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export RAY_WORKER_NICENESS=0

CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  "$PYTHON" -m pytest -p pytest_asyncio.plugin \
  tests/reward_loop/*_on_cpu.py tests/utils/reward_score/*_on_cpu.py \
  tests/trainer/diffusion/*_on_cpu.py \
  tests/trainer/omni/test_omni_sync_resume_bridge_on_cpu.py \
  -q --asyncio-mode=auto

CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PYTHON" -m tests.reward_loop.validate_native_fsdp \
  --model-path "$PICKSCORE_MODEL" --processor-path "$PICKSCORE_PROCESSOR" \
  --media-jsonl "$MEDIA_JSONL" --residency-cycles 20 --output "$PARITY_OUTPUT"

CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON" -m tests.reward_loop.validate_native_fsdp_gpu \
  --adapter clipscore --model-path "$CLIP_MODEL" --processor-path "$CLIP_PROCESSOR" \
  --media-jsonl "$MEDIA_JSONL" --replicas 2 --cycles 20 \
  --fault rank_exit --output "$GPU_OUTPUT"

CUDA_VISIBLE_DEVICES='' "$PYTHON" -m tests.reward_loop.validate_native_fsdp_cpu \
  --adapter clipscore --model-path "$CLIP_MODEL" --processor-path "$CLIP_PROCESSOR" \
  --replicas 2 --cycles 2 --output "$CPU_OUTPUT"

CUDA_VISIBLE_DEVICES='' "$PYTHON" -m tests.reward_loop.validate_native_fsdp_cpu \
  --model-path unused --processor-path "$CLIP_PROCESSOR" \
  --replicas 2 --cycles 20 --tiny --fault rank_exit --output "$CPU_FAULT_OUTPUT"

# TINY_CHECKPOINT is produced by a preceding --tiny validation run.
CUDA_VISIBLE_DEVICES="$GPU_TRIPLE" "$PYTHON" -m tests.reward_loop.validate_native_fsdp_faults \
  --model-path "$TINY_CHECKPOINT" --processor-path "$PICKSCORE_PROCESSOR" \
  --bundles 2,0 --output "$FAULT_OUTPUT"

pre-commit run --all-files --show-diff-on-failure
```

Use `--adapter pickscore` with matching model/processor paths for the other adapter. `--resident` checks dedicated
residency. The general GPU validator requires exactly two visible GPUs per replica and uses reversed/interleaved
logical bundles. `--fault facade_exit` and `--fault facade_hang` exercise facade death/unresponsiveness.
`MEDIA_JSONL` uses normal trainer `input` and image `output` fields; all images must share a shape for tensor batching.

## Remaining acceptance gates

| Scope | Gate before claiming support |
| --- | --- |
| Multi-group construction | Real second-group constructor failure and rollback of all previously created actors. |
| GPU streaming | Two groups, deployment-wide faults, multiple agent workers and measured aggregate pending work. |
| Shared two-group training | Actual scoring/unload failure; no actor-update entry; all groups terminated; checkpoint unchanged. |
| Other training combinations | Reference/KL and FSDP1 actor acceptance if included; DPO requires its own algorithm acceptance. |
| Multi-node | Two physical hosts, node-local group placement, node loss and network-failure handling. |
| NPU / trainable rewards | Real NPU stack; independently defined training objective and optimizer/checkpoint contracts. |
| Production stability | Intended resolution/denoising workload and longer runs with GPU/host memory, pending-work and process monitoring. |

Multi-rank training tooling must persist per-rank assertions and reject actor-update entry for every fatal reward
fault mode. Final executions must capture the full candidate and all test helpers, including rollout changes where
applicable. Historical launcher metadata omitted some helper/rollout hashes; later source manifests cannot retroactively
supply execution-time identity for them.

The short runs establish real functional updates and continuation for their recorded configurations. They establish
neither convergence, reward quality, throughput improvement, broad runtime compatibility nor general release readiness.
