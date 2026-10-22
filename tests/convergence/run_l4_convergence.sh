#!/usr/bin/env bash
# L4 convergence entrypoint.
#
# Runs the declarative L4 recipes through the runner, which validates each
# recipe, checks its declared hardware/data/checkpoint preconditions, optionally
# runs the real recipe, scores the reward/loss curve against the reviewed
# release baseline, and writes `release_readiness.json` / `.md`.
#
# Modes:
#   preflight  validate recipes and preconditions only; never starts training
#   baseline   run the recipes and store reviewed baselines (release owners only)
#   verify     run the recipes and compare against the stored baselines
#   report     rebuild the release report from existing per-case results
#
# Environment:
#   MODE                preflight | baseline | verify | report (default: preflight)
#   REGISTRY            recipe directory (default: tests/convergence/recipes)
#   OUTPUT_ROOT         artifact root (default: <repo>/outputs/l4_convergence)
#   L4_CASES            comma-separated case_ids to run (default: every recipe)
#   L4_TIMEOUT_MINUTES  override each recipe's budget (useful for smoke runs)
#   L4_*_PATH / L4_*_DATA_ROOT
#                       asset overrides referenced by the recipes' env_path/env_root
#
# The exit status is the release gate: 0 only when the selected mode's own
# success criterion is met.  A skipped, timed-out, invalid, or incomparable case
# never exits 0 in verify mode.
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODE=${MODE:-preflight}
REGISTRY=${REGISTRY:-${SCRIPT_DIR}/recipes}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/l4_convergence}

CASE_ARGS=()
if [ -n "${L4_CASES:-}" ]; then
    IFS=',' read -r -a cases <<< "${L4_CASES}"
    for case_id in "${cases[@]}"; do
        CASE_ARGS+=(--case "${case_id}")
    done
fi

TIMEOUT_ARGS=()
if [ -n "${L4_TIMEOUT_MINUTES:-}" ]; then
    TIMEOUT_ARGS+=(--timeout-minutes "${L4_TIMEOUT_MINUTES}")
fi

cd "${REPO_ROOT}"

python3 -m tests.convergence.run_convergence \
    --registry "${REGISTRY}" \
    --output-root "${OUTPUT_ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --mode "${MODE}" \
    ${CASE_ARGS[@]+"${CASE_ARGS[@]}"} \
    ${TIMEOUT_ARGS[@]+"${TIMEOUT_ARGS[@]}"}

echo "[L4] artifacts under ${OUTPUT_ROOT}"
