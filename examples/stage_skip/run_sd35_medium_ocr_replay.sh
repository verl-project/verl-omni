#!/usr/bin/env bash
# Replay SD3.5-Medium OCR training steps with per-stage skip (V1 sync trainer).
#
#   bash examples/stage_skip/run_sd35_medium_ocr_replay.sh          # record, replay, compare
#   bash examples/stage_skip/run_sd35_medium_ocr_replay.sh profile  # training phase only, from an existing dump
#
# Hydra overrides after the mode reach every pass, e.g.
#   bash examples/stage_skip/run_sd35_medium_ocr_replay.sh replay trainer.nnodes=1
#
# Environment:
#   OCR_WORKSPACE / WORKSPACE  dataset root of the base recipe (default: $HOME)
#   STAGE_SKIP_RUN_DIR         dumps, metrics, checkpoints (default: $WORKSPACE/stage_skip/sd35_medium_ocr)
#   NUM_STEPS                  training steps per pass (default: 3)
#   STAGES                     omni_skip roles to replay (default: "reward old_log_prob")
#
# See README.md in this directory.
set -euo pipefail

MODE=${1:-replay}
[ $# -gt 0 ] && shift
EXTRA_OVERRIDES=("$@")

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}
RUN_DIR=${STAGE_SKIP_RUN_DIR:-$WORKSPACE/stage_skip/sd35_medium_ocr}
NUM_STEPS=${NUM_STEPS:-3}
read -r -a STAGES <<< "${STAGES:-reward old_log_prob}"

DUMP_DIR=$RUN_DIR/dump
METRICS_DIR=$RUN_DIR/metrics
STEP_LIST="[$(seq -s, 1 "$NUM_STEPS")]"

# The base recipe seeds rollout and data; the actor needs this flag so the LoRA init and the update reproduce.
BASE_OVERRIDES=(
    actor_rollout_ref.actor.fsdp_config.full_determinism=True
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
)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

run_pass() {
    local name=$1
    shift
    local metrics=$METRICS_DIR/$name.jsonl
    local overrides=("${BASE_OVERRIDES[@]}")
    local stage
    for stage in "$@"; do
        overrides+=(
            "omni_skip.$stage.enable=True"
            "omni_skip.$stage.dump_dir=$DUMP_DIR"
            "omni_skip.$stage.steps=$STEP_LIST"
        )
    done
    echo "=== Stage skip pass '$name': rollout_tq ${*:+and $*} over steps $STEP_LIST ==="
    mkdir -p "$METRICS_DIR"
    VERL_FILE_LOGGER_PATH=$metrics bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh \
        "${overrides[@]}" "${EXTRA_OVERRIDES[@]}"
    [ -s "$metrics" ] || die "pass '$name' left no metrics at $metrics. VERL_FILE_LOGGER_PATH does not reach \
the trainer on a pre-started Ray cluster: run without RAY_ADDRESS so the job starts its own."
}

check_stage_dumps() {
    local stage dumped
    for stage in "${STAGES[@]}"; do
        dumped=$(find "$DUMP_DIR" -path "*/$stage/*" -name stage_payload.pt | wc -l)
        [ "$dumped" -ge "$NUM_STEPS" ] ||
            die "stage '$stage' dumped $dumped payloads over $NUM_STEPS steps: the record pass did not reach it."
    done
}

case "$MODE" in
    replay)
        if [ -n "$(ls -A "$DUMP_DIR" 2>/dev/null)" ]; then
            die "$DUMP_DIR already holds a dump, so a record pass would replay it. \
Remove it or set STAGE_SKIP_RUN_DIR to a fresh directory."
        fi
        run_pass record "${STAGES[@]}"
        check_stage_dumps
        run_pass replay "${STAGES[@]}"
        python3 examples/stage_skip/compare_replay_metrics.py \
            "$METRICS_DIR/record.jsonl" "$METRICS_DIR/replay.jsonl" \
            --num-steps "$NUM_STEPS" --replayed-stages "${STAGES[@]}"
        ;;
    profile)
        [ -n "$(find "$DUMP_DIR" -name tq_batch.pt 2>/dev/null)" ] ||
            die "no rollout dump under $DUMP_DIR: run the default replay mode first."
        run_pass profile
        echo "Compare timing_s/* in $METRICS_DIR/record.jsonl and $METRICS_DIR/profile.jsonl (see README.md)."
        ;;
    *)
        die "unknown mode '$MODE': expected replay or profile."
        ;;
esac
