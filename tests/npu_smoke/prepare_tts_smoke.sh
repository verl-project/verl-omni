#!/usr/bin/env bash
# Check optional audio prerequisites and cache the pinned tiny source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
case "${1:-}" in
    ''|--check-only) ;;
    *) echo "Usage: $0 [--check-only]" >&2; exit 2 ;;
esac

command -v sox >/dev/null
python3 - <<'PY'
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
declared = project["dependencies"] + [
    requirement for extra in project["optional-dependencies"].values() for requirement in extra
]
for line in Path("tests/npu_smoke/requirements_tts_smoke.txt").read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    pinned = Requirement(line)
    specs = list(pinned.specifier)
    if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
        raise SystemExit(f"Expected an exact audio smoke pin: {line}")
    for requirement in map(Requirement, declared):
        if canonicalize_name(requirement.name) != canonicalize_name(pinned.name):
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if specs[0].version not in requirement.specifier:
            raise SystemExit(f"Audio smoke pin {pinned} conflicts with pyproject.toml: {requirement}")

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
