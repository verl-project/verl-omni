#!/usr/bin/env bash
# Prepare optional audio packages and the pinned tiny source before offline runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
case "${1:-}" in
    '')
        python3 -m pip install --no-deps -r tests/npu_smoke/requirements_tts_smoke.txt
        python3 -m pip install --no-deps --force-reinstall \
            "qwen-tts @ https://github.com/QwenLM/Qwen3-TTS/archive/$(cat .github/qwen_tts_pin.txt).tar.gz"
        if ! command -v sox >/dev/null; then
            apt-get update
            apt-get install -y --no-install-recommends sox
        fi
        ;;
    --check-only) ;;
    *) echo "Usage: $0 [--check-only]" >&2; exit 2 ;;
esac

command -v sox >/dev/null
python3 - <<'PY'
import librosa
import onnxruntime
import soundfile
import sox
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
PY

if [[ "${1:-}" != --check-only ]]; then
    python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "optimum-intel-internal-testing/tiny-random-qwen3-tts",
    revision="6374d605b31381cac6f9577f5e742af2f76ba79c",
)
PY
fi
