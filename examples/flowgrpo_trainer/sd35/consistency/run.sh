#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run.sh manifest|admit|capture|replay}
if (($# != 1)); then
  echo 'expected exactly one mode' >&2
  exit 2
fi
: "${SOURCE_ROOT:?absolute deployed source checkout required}"
: "${PYTHON:?Python executable required}"
recipe="$SOURCE_ROOT/examples/flowgrpo_trainer/sd35/consistency"
case "$mode" in
  manifest)
    : "${SOURCE_MANIFEST:?fresh source manifest path required}"
    args=(--source "$SOURCE_ROOT" --output "$SOURCE_MANIFEST")
    if [[ -n ${SOURCE_HEAD:-} ]]; then args+=(--head "$SOURCE_HEAD"); fi
    "$PYTHON" "$recipe/source_manifest.py" "${args[@]}"
    ;;
  admit|capture)
    rollout_offload=${SD35_ROLLOUT_OFFLOAD:-none}
    : "${MODEL_DIR:?local SD3.5 model directory required}"
    : "${OUTPUT_ROOT:?capture parent required}"
    : "${RUN_ID:?safe fresh run identifier required}"
    : "${SOURCE_MANIFEST:?source manifest required}"
    : "${SOURCE_MANIFEST_SHA256:?independently retained manifest SHA256 required}"
    : "${SOURCE_HEAD:?expected source Git revision required}"
    if [[ $mode == capture ]]; then : "${PIN_FILE:?fresh pin file outside capture root required}"; fi
    cd -- "$SOURCE_ROOT"
    export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    common=(--source "$SOURCE_ROOT" --model "$MODEL_DIR" --output-root "$OUTPUT_ROOT"
      --run-id "$RUN_ID" --source-manifest "$SOURCE_MANIFEST"
      --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" --expected-source-head "$SOURCE_HEAD"
      --rollout-offload "$rollout_offload")
    if [[ $mode == admit ]]; then
      CUDA_VISIBLE_DEVICES='' "$PYTHON" "$recipe/capture.py" --role admit "${common[@]}"
    else
      "$PYTHON" "$recipe/capture.py" --role rollout "${common[@]}"
      actor_status=0
      "$PYTHON" "$recipe/capture.py" --role actor "${common[@]}" \
        --rollout-root "$OUTPUT_ROOT/$RUN_ID/rollout" || actor_status=$?
      if ((actor_status != 0 && actor_status != 10)); then exit "$actor_status"; fi
      "$PYTHON" "$recipe/pin_producer.py" --root "$OUTPUT_ROOT/$RUN_ID" --output "$PIN_FILE"
      if ((actor_status == 10)); then exit 10; fi
    fi
    ;;
  replay)
    : "${OUTPUT_ROOT:?capture parent required}"
    : "${RUN_ID:?run identifier required}"
    : "${PIN_FILE:?external producer pin file required}"
    : "${TRUSTED_PIN_SHA256:?digest retained independently after capture required}"
    : "${REPLAY_OUTPUT:?fresh replay output path required}"
    cd -- "$SOURCE_ROOT"
    export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" scripts/replay_sd35_consistency.py --root "$OUTPUT_ROOT/$RUN_ID" \
      --pins "$PIN_FILE" --pins-sha256 "$TRUSTED_PIN_SHA256" --output "$REPLAY_OUTPUT"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
