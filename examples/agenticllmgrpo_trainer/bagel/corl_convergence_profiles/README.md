# Bagel Co-RL controlled convergence profiles

These profiles are proposed experiments, **not measured results**. CPU tooling tests
do not establish serving parity, FSDP correctness, image improvement, or convergence.
No training-host checkpoints, installed serving probability semantics, or GPU results
were available while preparing these files. The checkpoint preceding run gk3opfg7
step 26 and untracked runtime helpers still require recovery. `reproduction` records
its effective published settings; a fresh model start is not an exact reproduction.

Run from the repository root on the configured Linux/CUDA training host. The launcher
requires explicit starting model/checkpoint and existing fixed dataset paths; it
disables automatic resume and dataset regeneration. `START_CHECKPOINT` must be a
Bagel model directory usable by `actor_rollout_ref.model.path`; restoring an FSDP
training-state checkpoint instead needs the separately reviewed trainer resume path
and matching optimizer/scheduler/RNG state. Never substitute a new model silently.

```bash
export START_CHECKPOINT=/absolute/path/to/verified/bagel-model
export UNICOT_TRAIN=/absolute/path/to/fixed-small-train.parquet
export UNICOT_VAL=/absolute/path/to/fixed-heldout-panel.parquet
export EXPERIMENT_PREFIX=corl_convergence_20261023_fresh
export EXPERIMENT_ROOT=/absolute/path/to/new-experiment-root
python examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py B1 --seed 11 --dry-run
python examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py B1 --seed 11 --resolve-only
```

`--dry-run` works on CPU/Windows. `--resolve-only` invokes the real Bash recipe with
Hydra `--cfg job --resolve`, without model execution, and writes
`<experiment>/<profile>/provenance/run_manifest.json` and
`resolved-config.allowlisted.json`. The actual directory basename is
`<EXPERIMENT_PREFIX>_<profile>_seed<seed>`. Only allowlisted config fields are persisted;
the manifest records source/dependency/data/checkpoint hashes and labels configured
peak LRs separately from effective runtime LRs. After optimizer initialization and
on resume, supply `effective_optimizer_group_lrs` when calling `write_run_manifest`.
The launcher does not manufacture resolved configs or claim host composition passed.

Before training, independently verify all Gate A requirements (including two-rank
CUDA gradient/collective checks, one joint optimizer step, exact publication and
frozen-policy parity on every replica). Record their evidence paths in a host gate
JSON with `gate_A_passed: true` and `frozen_policy_parity_passed: true`, then run:

```bash
python examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py B1 --seed 11 --gates /absolute/path/to/verified-gates.json
```

Every profile fixes 512x512, 10-step SDE, noise .8, SDE window `[0,2]`, GEN clip .02,
CFG text/image scales 4.0/1.5, and the data-loader/FSDP/rollout seeds explicitly,
UND KL .01 enabled, entropy coefficient zero, std normalization true, four prompt
groups, microbatch one, and no implicit checkpoint resume. B0-B4 use trusted numeric
V2 reward consistently so B2/B3 optimize the same image target and B3-to-B4 isolates
extra passes. This is a declared experimental choice; reproduction retains V1.
Validation uses the whole fixed panel with 50-step ODE. Separately capture/evaluate
training-kernel SDE under the same fixed render seeds; do not pool kernels.

| Profile | UND LR / GEN LR | N / S | Passes | Update behavior |
| --- | --- | --- | --- | --- |
| reproduction | 3e-5 / 3e-5 | 8 / 1 | 3 | 200 updates; V1 runtime baseline settings |
| B0 | 1e-5 / 3e-5 | 4 / 2 | 1 | Optimizer and scheduler steps disabled |
| B1 | 1e-5 / 3e-5 | 4 / 2 | 1 | UND frozen and weight zero; original request fixed as GEN prompt |
| B2-1e-6 | 1e-6 / 3e-5 | 8 / 1 | 1 | GEN frozen and weight zero |
| B2-1e-5 | 1e-5 / 3e-5 | 8 / 1 | 1 | GEN frozen and weight zero |
| B3 | 1e-5 / 3e-5 | 4 / 2 | 1 | Joint, 50 updates |
| B4 | 1e-5 / 3e-5 | 4 / 2 | 3 | Joint V2 final-quality/call-cost, 50 updates |

Run B0 and promising B3/B4 arms independently at training/sampling seeds 11, 22, 33.
B0 still performs rollout/diagnostic backwards but neither optimizer nor scheduler
steps; verify before/after parameter hashes. B1 uses
`+actor_rollout_ref.model.corl_freeze_und=True`,
`+actor_rollout_ref.rollout.agent.corl_fixed_gen_prompt=__user_request__`, and one pass.
Confirm UND parameter hashes stay unchanged, including GEN text-conditioning paths.
B2 uses `+actor_rollout_ref.model.corl_freeze_gen=True`; verify GEN hashes unchanged
and UND prompts/actions actually change. Loss weights alone do not freeze a group.
Common image seeds must be captured and checked across paired frozen/learning runs;
the profile's `data.seed` alone is not evidence of exact render-seed pairing.

All one-pass arms render 32 images per four-prompt batch (B1/B3: 4x4x2;
B2: 4x8x1). Confirm actual counts after rejection/refill. B4 has variable image cost;
report calls and wall time, and match any comparison by actual rendered-image and
optimizer-update budgets. Retain starting/last healthy checkpoints and full training
state. B5 objective changes and std-normalization/episode-balanced arms require a
separate measured failure; these profiles do not implement those unmotivated arms.

Build the balanced fixed panel from the native UniCoT parquet or normalized JSONL:

```bash
python scripts/bagel_corl/build_eval_panel.py --dataset /data/val.parquet --training-dataset /data/train.parquet --output /data/panel.parquet --per-task 200 --seed 42 --render-seeds 42 43
python scripts/bagel_corl/write_run_manifest.py --config /runs/resolved-secret-free.json --checkpoint "$START_CHECKPOINT" --output /runs/provenance
python scripts/bagel_corl/check_policy_parity.py --config /runs/parity-calibration.json --artifact /runs/frozen-replays.jsonl --output /runs/parity
python scripts/bagel_corl/compare_checkpoints.py --baseline /runs/base-sde.jsonl --candidate /runs/step50-sde.jsonl --output /runs/comparison-sde.json
```

Parquet output retains the original trainer-readable rows and adds evaluation seeds
and hashes to `extra_info`, with a `.panel.json` metadata sidecar. JSON output is
also supported as a standalone evaluation specification. The host evaluator must
consume and verify the recorded render seeds; setting a data-loader seed alone does
not ensure paired images.
Duplicate IDs/prompts, train overlap, or fewer than 200 per task fail. A smaller
`--per-task` is a smoke fixture only. Native task labels `plan`/`reflect` map to
`breakdown`/`reflection`. The comparer requires both tasks for every training seed,
identical prompt/render pairs, and constant kernel/scorer/evaluation config hashes.
It reports paired prompt-cluster effects/CIs per seed/task and a fixed 50/50 task
aggregate with a hierarchical seed/prompt bootstrap. Quality/protocol gates require
at least 200 prompts per task and three training seeds. Independent frozen-scorer or
blinded-review evidence and useful rewrite gains at reported cost remain separate
requirements before any generalization claim.

Parity config requires `policy_version`, `expected_replicas` (either one shared ID
list or separate lane lists such as `{"und": [0, 1], "gen": [10]}`),
`actor_repeat_tolerance` (float32 reference: 1e-5),
`gen_ratio_p95_tolerance` (provisional .005), `und_mean_gap_baseline`,
`und_mean_gap_tolerance` (provisional .01), `logprob_convention` (`raw_policy` or
`exact_behavior_transforms`) and `calibration_id`. Calibrate bf16/cross-engine
tolerances on the installed serving version and verify its processed-logprob
semantics before writing that declaration. Capture identical token/context/mask
inputs and repeat actor replay at frozen weights before/after two publications.
The validator consumes captures and exits 2 on a failed numerical gate; it does
not capture models or prove that hashes correspond to exact tokens/masks.

Pending host work also includes inspecting at least 20 plan traces with real
reference subtasks before proposing a parser change, offline V1/V2 trace rescoring,
recovering step-25/step-200 checkpoints if available, and full held-out experiments.
