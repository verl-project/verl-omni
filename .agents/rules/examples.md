---
paths:
  - examples/**
---

# Example Recipes & Layout

Conventions and automated gates for `examples/`. Match the canonical layout and
reference recipes instead of copying older, non-uniform scripts.

## Canonical layout and naming

```text
examples/<algorithm>_trainer/
├── README.md                              # Algorithm overview (symlinked in docs/examples/)
├── data_process/                          # Shared data prep scripts (never dataprocess/)
│   └── <dataset_or_task>.py
└── <model>/
    ├── README.md                          # Model recipe guide (never README_*.md)
    ├── run_<model>_<task>[_<variant>].sh  # Primary recipe
    └── run_<model>_<task>_<variant>.sh    # Thin wrapper reusing the base recipe
```

- Avoid duplicate nested folders (e.g. `qwen3_omni/qwen3_omni/`) and do not dump
  shared converters directly inside `<model>/` when sibling recipes can share
  `data_process/`.
- Keep generated images, videos, audio, and training curves in PR evidence, not
  committed inside `examples/`.

## Reference patterns

Inspect these exemplars before adding a new recipe:

- Standalone trainer launcher: `run_qwen3_omni_thinker_gspo_veomni.sh` (clean
  env overrides) or `run_minicpm_omni_preference_lora.sh` (with fail-fast
  parquet existence checks).
- Variant / NPU / benchmark wrapper: `crd_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh`
  (resolves `script_dir`, validates positive integer degrees, and delegates via
  `exec bash` with differential Hydra overrides).
- Model README: `dpo_trainer/minicpm/README.md` or
  `diffusionnft_trainer/boogu_image/README.md` (`Last updated:`, `Prerequisites`,
  `Data`, `Launch`, and `Key Constraints`).

## Shell recipe skeleton

```bash
#!/usr/bin/env bash
# <Model> <algorithm> <task> recipe (<trainer / rollout engine>).
# Prepare data using the referenced README or data_process script.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# External overrides in UPPER_CASE; keep defaults free of local paths.
WORKSPACE=${WORKSPACE:-${HOME}}
MODEL_PATH=${MODEL_PATH:-org/model}
TRAIN_FILE=${TRAIN_FILE:-${WORKSPACE}/data/task/train.parquet}
VAL_FILE=${VAL_FILE:-${WORKSPACE}/data/task/test.parquet}
NUM_GPUS=${NUM_GPUS:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
```

## Shell, config, and docs invariants

- Always start `run_*.sh` with `#!/usr/bin/env bash` and enable `set -e`
  (`set -euo pipefail` recommended). Preserve trainer exit status, including
  through `tee`; handle intentionally recoverable commands explicitly.
- Group overridable defaults (`${VAR:-default}`) near the top; quote every
  path-bearing argument and forward caller overrides with `"$@"` last. Never
  hardcode `/home/`, `/Users/`, `/nfs/`, `/mnt/`, `/root/`, credentials, or
  `.venv/bin/python`.
- Use plain Hydra overrides for declared config fields; reserve `+` for new keys
  ([config.md](config.md)).
- Validate positive parallel degrees and GPU divisibility before shell arithmetic;
  keep worker/replica counts consistent with runtime topology. Keep output paths
  recipe-attributable and overridable.
- Let a parent scheduler own log redirection; cleanup must target only the
  launched job, not all host Ray or GPU processes.
- Verify scripts with `bash -n` and stub-trainer argv checks without launching
  GPUs; use [testing_guide.md](../../docs/contributing/testing_guide.md) for
  required L1/L2 runtime validation.
- In `README.md`, include `Last updated: MM/DD/YYYY` (`check-docs-time-info`),
  add the `docs/examples/*.md` git symlink, and use full
  `https://github.com/verl-project/verl-omni/blob/main/examples/...` URLs for
  links into `examples/**` so `check_example_docs_symlinks.py` passes on Sphinx.

## Automated gates vs review conventions

| Check / Rule | Scope | Enforced |
| --- | --- | --- |
| `EX001` | Valid UTF-8 + `bash -n` syntax (no execution) | Incremental (`check-pr-lint`) |
| `EX002` | `run_*.sh` forwards `"$@"` (excl. comments/heredocs) | Incremental (`check-pr-lint`) |
| `EX003` | No committed media under `examples/` | Incremental (`check-pr-lint`) |
| `EX004` | Use `README.md`, not `README_*.md` | Incremental (`check-pr-lint`) |
| `EX005` | `run_*.sh` starts with `#!/usr/bin/env bash` + `set -e` | Incremental (`check-pr-lint`) |
| `EX006` | No `dataprocess/`, duplicate dirs, or `.venv/bin/python` | Incremental (`check-pr-lint`) |
| `VO014` | No hardcoded `/home/`, `/Users/`, `/nfs/`, `/mnt/`, `/root/` | Incremental (`check-pr-lint`) |
| Symlinks / `Last updated` | `docs/examples/` symlinks + docs timestamps | `check_example_docs_symlinks.py` / `check-docs-time-info` |
