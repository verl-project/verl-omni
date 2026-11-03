#!/usr/bin/env bash
# Run SD3.5-Medium OCR once, then repeat cached stages within that run (V1 sync).
#
#   bash examples/stage_skip/run_sd35_medium_ocr_replay.sh
#
# Hydra overrides are forwarded to the base recipe, e.g.
#   bash examples/stage_skip/run_sd35_medium_ocr_replay.sh trainer.nnodes=1
#
# Environment:
#   OCR_WORKSPACE / WORKSPACE  dataset root of the base recipe (default: $HOME)
#   STAGE_SKIP_RUN_DIR         dumps, metrics, checkpoints (default: $WORKSPACE/stage_skip/sd35_medium_ocr)
#   NUM_STEPS                  training steps (default: 3)
#   STAGES                     omni_skip roles to replay (default: "reward old_log_prob")
#
# See README.md in this directory.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}
RUN_DIR=${STAGE_SKIP_RUN_DIR:-$WORKSPACE/stage_skip/sd35_medium_ocr}
NUM_STEPS=${NUM_STEPS:-3}
read -r -a STAGES <<< "${STAGES:-reward old_log_prob}"

DUMP_DIR=$RUN_DIR/dump
METRICS_DIR=$RUN_DIR/metrics

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$NUM_STEPS" =~ ^[1-9][0-9]*$ ]] || die "NUM_STEPS must be a positive integer."
STEP_LIST="[$(seq -s, 1 "$NUM_STEPS")]"

if [ -n "$(ls -A "$DUMP_DIR" 2>/dev/null)" ]; then
    die "$DUMP_DIR already holds a dump. Set STAGE_SKIP_RUN_DIR to a fresh directory to record step 1."
fi

OVERRIDES=(
    'trainer.logger=["console","file"]'
    trainer.project_name=stage_skip
    trainer.experiment_name=sd35_medium_ocr
    trainer.default_local_dir="$RUN_DIR/checkpoints"
    trainer.resume_mode=disable
    trainer.total_training_steps="$NUM_STEPS"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    skip.rollout_tq.enable=True
    skip.rollout_tq.dump_dir="$DUMP_DIR"
    skip.rollout_tq.steps="$STEP_LIST"
    skip.rollout_tq.action=repeat
)

for stage in "${STAGES[@]}"; do
    OVERRIDES+=(
        "omni_skip.$stage.enable=True"
        "omni_skip.$stage.dump_dir=$DUMP_DIR"
        "omni_skip.$stage.steps=$STEP_LIST"
        "omni_skip.$stage.action=repeat"
    )
done

echo "Stage skip: record step 1, then repeat rollout_tq and ${STAGES[*]} through step $NUM_STEPS."
mkdir -p "$METRICS_DIR"
VERL_FILE_LOGGER_PATH=$METRICS_DIR/repeat.jsonl bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh \
    "${OVERRIDES[@]}" "$@"
[ -s "$METRICS_DIR/repeat.jsonl" ] || die "run left no metrics at $METRICS_DIR/repeat.jsonl. VERL_FILE_LOGGER_PATH does not reach \
the trainer on a pre-started Ray cluster: run without RAY_ADDRESS so the job starts its own."
