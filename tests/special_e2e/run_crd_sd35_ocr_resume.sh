#!/usr/bin/env bash
# Real SD3.5-M OCR validation: fresh two-iteration run, then resume to iteration four.
# Requires prepared OCR data and two GPUs; run from the repository root.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export GROUP_SIZE=${GROUP_SIZE:-2}
export IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-256}
export OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/validation/crd/smoke"}
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Use a new OUTPUT_DIR so validation cannot overwrite an earlier run: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
export TOTAL_TRAINING_STEPS=2
bash "$repo_root/examples/crd_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh" \
    trainer.resume_mode=disable trainer.save_freq=1 trainer.test_freq=1 \
    data.val_max_samples=4 trainer.logger='["console"]' "$@" \
    2>&1 | tee "$OUTPUT_DIR/fresh.log"
python3 tests/special_e2e/check_crd_run.py --log "$OUTPUT_DIR/fresh.log" --output-dir "$OUTPUT_DIR" --steps 1 2
export TOTAL_TRAINING_STEPS=4
bash "$repo_root/examples/crd_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh" \
    trainer.resume_mode=resume_path "trainer.resume_from_path=$OUTPUT_DIR/global_step_2" \
    trainer.save_freq=1 trainer.test_freq=1 data.val_max_samples=4 \
    trainer.logger='["console"]' "$@" \
    2>&1 | tee "$OUTPUT_DIR/resume.log"
python3 tests/special_e2e/check_crd_run.py --log "$OUTPUT_DIR/resume.log" --output-dir "$OUTPUT_DIR" --steps 3 4
