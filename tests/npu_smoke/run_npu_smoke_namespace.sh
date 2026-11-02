#!/usr/bin/env bash
# Execute one smoke allocation with private /tmp and process visibility.
set -euo pipefail

[[ $# -eq 3 && $$ -eq 1 ]] || {
    echo "NPU smoke worker requires unshare --mount --pid --fork --mount-proc." >&2
    exit 2
}
NAMESPACE_TMP="$1"
REPO_ROOT="$2"
GROUP="$3"
[[ -d "${NAMESPACE_TMP}" && -d "${REPO_ROOT}" ]] || {
    echo "Missing namespace backing directory or repository." >&2
    exit 2
}
mount --bind -- "${NAMESPACE_TMP}" /tmp
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp
unset RAY_ADDRESS
cd "${REPO_ROOT}"

case "${GROUP}" in
    --probe) exit 0 ;;
    core|e2e)
        exec bash "tests/npu_smoke/run_npu_smoke_${GROUP}.sh" --num-npus "${NUM_NPUS:?}"
        ;;
    *)
        echo "Unknown NPU smoke group: ${GROUP}" >&2
        exit 2
        ;;
esac
