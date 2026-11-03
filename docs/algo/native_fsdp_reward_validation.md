# Native FSDP reward: validation report

Last updated: 11/02/2026.

## Scope and validation baselines

This change supports one two-rank FSDP2 logical replica in a dedicated single-node reward GPU pool,
frozen FP32 PickScore and synchronous V1 diffusion training.
The [RFC](native_fsdp_reward_rfc.md) defines the proposed configuration and execution contract.

Upstream integration used `main` at `aaaea177e80dd55b6d90a3d83ef08984bafc3a46`.
It preserves upstream CPU-native placement, named-media validation and component aggregation,
reuses `clip_feature_tensor`, and explicitly declares `media_kind=image` in GPU validation inputs.
Model-specific executor validation stays in the reward adapter; shared worker configuration stays model-agnostic.
The branch does not include the two unrelated colocate-async commits from #742.

The original development baseline was `bfa310517315a5cda454d2482c3e84ba165deac1` with uncommitted implementation sources.
Its longer training and independent-venv results are recorded separately below.
They are not presented as reruns on upstream main.
Raw logs, source/evidence hash manifests, environment snapshots and local dependency locks are retained locally.
They are not repository deliverables; the PR contains the implementation, relevant tests, RFC and this report.

## Results on the upstream integration

| Check | Result |
| --- | --- |
| Scoped CPU regression suite | **1571 passed, 32 warnings**; reward/scorer, diffusion trainer and related omni resume suites, including real-Ray CPU-native tests |
| Real pretrained PickScore | 20 load/score/unload cycles; maximum score absolute error `1.7881393432617188e-7` |
| Rank/NCCL failures | 10 cases passed, including a real `_ALLGATHER_BASE` watchdog timeout |
| Facade failures | Four exit/SIGSTOP cases, both idle and during scoring; owner confirmed facade death and ranks/queue were dead before Ray shutdown |
| Real training | SD3.5 Medium FlowGRPO, 128×128, 4 denoising steps, `offload=true`, two full updates; optimizer 1→2, 484 changing LoRA tensors, gradient norms 0.11865234375 / 0.1669921875; validation/checkpoints and pre-shutdown cleanup passed |
| Repository checks | All applicable pre-commit hooks passed, including generated-config verification |

The scoped CPU suite was rerun after removing two unrelated rollout test-fixture edits from the PR.
The production source and GPU validation entry points are unchanged from the upstream GPU acceptance runs.
Documentation and removal of local validation artifacts do not introduce new GPU behavior.

The upstream GPU reruns used the existing validation environment with Diffusers 0.41.0 and an explicit worktree `PYTHONPATH`.
Separate visible GPU sets were assigned to concurrent experiments. These runs were not performance measurements.
Setup failures and interrupted runs were retained locally and excluded from passing results.

## Original-baseline acceptance

| Check | Result |
| --- | --- |
| CPU regression | Existing environment and independent venv each passed **1107 tests, 35 warnings**, with GPUs hidden |
| Real model sharding | 986,109,441 parameters; ranks held 493,055,873 / 493,053,568 parameters, all represented as DTensors |
| Raw-output and score parity | Batches 1/3/16/17, reordering, repeated prompts and long text; maximum score absolute error `2.384185791015625e-7`; raw outputs and scores asserted at `rtol=1e-4, atol=2e-5` |
| Existing scorer/manager | Training and validation paths each processed 17 samples with uint8 media, PickScore/JPEG weights 0.7/0.3 and diagnostics |
| Validation before allocation | Rejected mutually exclusive devices/groups, duplicate/boolean/out-of-range indices, invalid timeouts, unsupported model arguments and non-required terms; rejected shared pools, streaming, unsupported V1 modes and multiple nodes |
| Parent-pool placement | Used logical bundles `[2,0]` from a three-bundle pool; asserted raw Ray `GPU_group_2_*` / `GPU_group_0_*` allocations and distinct GPU UUIDs |
| Rank/NCCL failures | Missing/corrupt weights, initialization/forward failures on either rank, either rank killed during forward, a stuck rank and the NCCL watchdog: 10 cases, all invalidated the group and confirmed both actors dead |
| Actual forward timeout | Both ranks passed FSDP root pre-forward/all-gather and wrote markers before the stall; inference hit a 5-second controller deadline with Ray `GetTimeoutError`, separately from the 120-second loading budget |
| Residency cycles | Pretrained PickScore and tiny CLIP each completed 20 load/score/unload cycles; reloaded scores were bitwise equal, no model parameters remained after sleep and allocations were stable |
| NCCL watchdog | Both ranks entered real forward; 5-second collective timeout and 40-second controller budget produced `_ALLGATHER_BASE` watchdog logs and `ActorDiedError`; only diagnostic-dump waiting was shortened to 1 second in the test |
| Facade failures | Exit/SIGSTOP while idle or scoring, four cases; hung facade detected in about 3 seconds; rank/facade/queue actors were DEAD before Ray shutdown without calling GC |
| Concurrent sleep/close | Submitted lifecycle operations after both ranks entered real forward; verified lock waiting and completion only after forward was released |
| Controlled CPU failures | Partial submission, repeated cancellation, draining, duplicate-output rejection, failed termination confirmation, construction rollback, multiple cleanup failures and result writing |
| Continuous real RL | Ten full updates at 256×256 with 8 denoising steps, validation/checkpoint every step, finite nonzero gradients throughout, optimizer 9→10 and 484 changing LoRA tensors |
| Resume | Restored step 10 and completed step 11; optimizer 10→11, 484 changing LoRA tensors, gradient norm 0.0003643035888671875; each of 11 steps had 4 training and 2 validation images |
| Resident mode | Two `offload=false` updates, optimizer 1→2 and 484 changing LoRA tensors; facade/rank/queue actors DEAD before normal return |
| Independent-venv training | Two full 128×128 updates with validation, optimizer 1→2 and 484 changing LoRA tensors; parameter verifier and pre-shutdown resource checks passed |
| Fit-failure cleanup | Resume from step 11 with missing reward weights caused a real fit error; ranks, facade, TransferQueueController and SimpleStorageUnit were DEAD while the driver and Ray remained live; checkpoint 11 remained intact and no checkpoint 12 was created |

CPU mocks validate control flow. GPU experiments validate actual sharding, NCCL/Ray execution and actor death.
Early runs that timed out during wake or observed placement through a Ray API that removed bundle suffixes were corrected and rerun;
those earlier runs are not counted as acceptance.

A second review exposed a real ownership gap: killing the CPU facade made its close RPC fail while both ranks stayed alive
for at least 13 seconds, until explicit GC. `NativeFSDPManagedRewardModel` fixes that gap by retaining independent termination
handles in the trainer, supervising RPC ownership and facade heartbeats, and confirming death without GC.
Normal collective ordering remains in the facade. CPU tests cover continuing cleanup after one actor fails and preserving original errors.

## Environment and input provenance

- NVIDIA RTX A6000, CUDA runtime 13.0; Python 3.12.4, PyTorch 2.13.0+cu130, Ray 2.59.0,
  Transformers 5.14.1, Diffusers 0.41.0 and vLLM 0.30.0 in the existing validation environment.
- verl commit `fefb080262e1c015a0ea05f958822a6a512dc795` and vLLM-Omni commit
  `12e9280010afc41ce5c6a1f46900fc329cf08377` matched the project pins.
- Real PickScore_v1 checkpoint SHA-256:
  `ef31ef6fc5ff4d9bb90dd232df4e145887ba62c5a03aa2841415f8c25f18d52e`.
- Original-baseline continuous training used SD3.5 Medium, rank-4 LoRA, FlowGRPO, 256×256, 8 denoising steps,
  2 prompts × 2 responses per step, 2 validation samples and PickScore/JPEG weights 0.7/0.3.
  One GPU hosted actor/rollout and two dedicated reward GPUs hosted the shards.
- Prompts were 8 training and 2 validation short prompts selected from flow_grpo's Pick-a-Pic train/test splits.
  The legacy `data_source=pickscore_sfw` label does not mean the prompts came from an SFW dataset.
- Final parity inputs reused four real images from training step 1 to form 17 test samples, including one extended prompt
  that triggered truncation. They were not 17 independently generated images or a quality/convergence evaluation.

The original baseline's independent venv used `include-system-site-packages=false`, installed project GPU/dev and OCR dependencies,
and checked module origins for Torch, Ray, Transformers, Diffusers, verl, vLLM, vLLM-Omni, TransferQueue and TensorDict.
It used Diffusers 0.40.0 under that baseline's dependency constraints; current upstream requires >=0.41.0.
Its local dependency lock must not be reused directly for the upstream branch.
The upstream rerun used the existing 0.41.0 environment and does not establish a fresh upstream environment installation.
Three `pip check` metadata conflicts matched the repository's declared Transformers/kernels/accelerate overrides;
no FSDP-specific version exception was introduced. No fresh OS/container validation is claimed.

## Reproduction entry points

Run from the repository root with the project's pinned dependencies installed.
Set `PYTHON` to the validation interpreter and supply local checkpoint, processor, generated-media and output paths.
Set `GPU_PAIR`, `GPU_TRIPLE` and `TRAINING_GPUS` to available physical GPU lists; the placement indices remain logical bundles.
The manual accelerator validators require real GPUs and local model checkpoints.

```bash
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export RAY_WORKER_NICENESS=0

CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  "$PYTHON" -m pytest -p pytest_asyncio.plugin \
  tests/reward_loop/*_on_cpu.py tests/utils/reward_score/*_on_cpu.py \
  tests/trainer/diffusion/*_on_cpu.py \
  tests/trainer/omni/test_omni_sync_resume_bridge_on_cpu.py \
  -q --asyncio-mode=auto

CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PYTHON" -m tests.reward_loop.validate_native_fsdp \
  --model-path "$PICKSCORE_MODEL" --processor-path "$PICKSCORE_PROCESSOR" \
  --media-jsonl "$MEDIA_JSONL" --residency-cycles 20 --output "$PARITY_OUTPUT"

# First use validate_native_fsdp --tiny to generate a tiny checkpoint.
CUDA_VISIBLE_DEVICES="$GPU_TRIPLE" "$PYTHON" -m tests.reward_loop.validate_native_fsdp_faults \
  --model-path "$TINY_CHECKPOINT" --processor-path "$PICKSCORE_PROCESSOR" \
  --bundles 2,0 --output "$FAULT_OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PYTHON" -m tests.reward_loop.validate_native_fsdp_owner \
  --overrides "$OVERRIDES" --model-path "$TINY_CHECKPOINT" \
  --processor-path "$PICKSCORE_PROCESSOR" --output "$FACADE_OUTPUT"
```

Automatic pytest plugin discovery was disabled to avoid a local ROS `launch_testing` conflict; project assertions still ran.
GPU validators create isolated Ray clusters and clean up only their owned actors and placement groups.

For training, use the existing `verl_omni.trainer.main_diffusion_v1` entry point with a synchronous SD3.5 FlowGRPO configuration
and the dedicated reward-pool configuration in the [RFC](native_fsdp_reward_rfc.md).
Save the complete Hydra override argument list as a JSON array in `OVERRIDES`, using local dataset/model/output paths.
Set validation and checkpoint frequency to 1. The upstream acceptance run used 128×128, 4 denoising steps and 2 training steps;
the original continuous run used 256×256, 8 denoising steps and 10 training steps.

```bash
CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" "$PYTHON" -m tests.special_e2e.validate_native_fsdp_training_failure \
  --overrides "$OVERRIDES" --checkpoint-root "$CHECKPOINT_ROOT" \
  --expected-step 2 --expect-success --output "$EXIT_RESULT"

"$PYTHON" -m tests.special_e2e.verify_native_fsdp_training \
  --output "$RUN_DIR" --log "$TRAIN_LOG" --steps 2
```

For the original continuous run, use `--expected-step 10` and verify with `--steps 10 --image-size 256`.
To resume, retain the same paths, set `trainer.resume_mode=auto` and `trainer.total_training_steps=11`,
then verify with `--steps 11 --resume-from 10 --image-size 256`.
For fit-failure cleanup, resume from step 11, target step 12 and point reward `model_path` to a directory containing
valid config but no weights. Run the failure validator with `--expected-step 11` and omit `--expect-success`.
It catches the expected trainer error, checks GCS actor states before Ray shutdown and confirms checkpoint 11 was preserved.
A passing negative test does not mean training with missing weights succeeded.

Shared-host cold loading required explicit `timeout=300` and `collective_timeout=120` for the original continuous run;
production defaults remain 120/30. Setup OOM, missing OCR dependencies and an incorrect local output-path join were corrected
and fully rerun; those setup attempts were not counted as passes.

## Limits

Multiple groups/nodes, streaming, shared-GPU scheduling, CPU/NPU FSDP, other native model adapters and trainable reward weights
require separate work. Real RL covered both offload modes at the two sizes above; larger resolutions, long-term convergence
and matched-workload performance require further experiments.
External GPU load on the shared host prevents a throughput or strict peak-memory benefit claim.
Sleep releases model weights; it does not promise zero accelerator-context or communication-buffer allocations.

This is not a whole-repository green claim. Four non-CPU `test_visual_reward_manager.py` cases required unavailable
`~/models/tiny-random/{Qwen-Image,qwen3-vl}` checkpoints and reproduced the same failures on the clean original baseline.
Those tests were not changed or supplied with fake models to hide the missing dependencies.
