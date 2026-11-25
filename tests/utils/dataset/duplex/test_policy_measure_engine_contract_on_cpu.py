# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-contract check: ``policy_measure`` against the engine's real stage-0 sampler.

The engine's sampler fixture lives in its own ``tests`` package, which collides
with this repo's ``tests`` package in-process, so the sampler runs in a
subprocess over a scripted episode and the two measures are compared here.
"""

import json
import pathlib
import subprocess
import sys

import pytest

_MAX_ABS_NATS = 1e-5
_STEP_CLASSES = {"listen", "content", "boundary", "turn_eos", "rewrite"}

#: Runs the engine's real MiniCPM-o 4.5 duplex stage-0 sampler with capture on
#: over one scripted row and compares each recorded log-probability with the
#: trainer's measure on the same logits. Prints ``@rows@`` then one JSON row
#: (``step_class``, ``token``, ``kind``, ``engine_logprob``, ``trainer_logprob``,
#: ``delta``) per decode step.
_PROBE_SCRIPT = """
import importlib.util
import json
import sys

import torch

engine_root, verl_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, verl_root)
sys.path.insert(0, engine_root)  # the engine test module imports its sibling via the repo's `tests` package
spec = importlib.util.spec_from_file_location(
    "engine_capture_test", f"{engine_root}/tests/model_executor/models/minicpmo_4_5/test_duplex_action_capture.py"
)
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)

from vllm_omni.protocol.duplex.action_records import decode_unit_payload  # noqa: E402

from verl_omni.utils.dataset.duplex.policy_measure import two_stage_logprob  # noqa: E402

CHUNK_EOS, LISTEN, TTS_PAD = eng.CHUNK_EOS, eng.LISTEN, eng.TTS_PAD
ROW = 0

model, states, md, steps0 = eng._setup(0)
VOCAB = steps0[0].shape[-1]
eng._enable_capture(model)
model._minicpmo45_duplex_row_identity = {ROW: ("sess-contract", 0, 100)}
states[ROW].current_turn_ended = True  # unit A's listen stays listen
for row in range(1, 6):  # keep the other rows inert
    states[row].current_turn_ended = True

# unit A: listen | unit B (spoken): three content tokens, a mid-turn listen draw
# the engine rewrites to tts_bos, then the boundary chunk_eos | unit C: turn_eos
# closing the turn | unit D: listen again (turn closed, no rewrite).
SCRIPT = ["listen", "content", "content", "rewrite", "content", "boundary", "turn_eos", "listen"]
CONTENT_TOKENS = iter(20 + step for step in range(4))


def logits_for(step_class):
    z = torch.zeros(6, VOCAB)
    if step_class in ("listen", "rewrite"):
        z[ROW, LISTEN] = 30.0
    elif step_class == "content":
        z[ROW, next(CONTENT_TOKENS)] = 30.0
    elif step_class == "boundary":
        z[ROW, CHUNK_EOS] = 30.0
    else:
        z[ROW, 8] = 30.0  # turn_eos_token_id in the fixture
    for r in range(1, 6):
        z[r, LISTEN] = 30.0
    return z


temperature = float(md.temperature[ROW])
forbidden = torch.tensor([TTS_PAD, CHUNK_EOS])  # the fixture tokenizer has no bad_token_ids
rows = []
for step, step_class in enumerate(SCRIPT):
    if step_class == "rewrite":
        states[ROW].current_turn_ended = False  # an open turn rewrites a listen draw
    z = logits_for(step_class)
    out = model._sample_minicpmo45_native_duplex_stage0(z.clone(), md, duplex_rows=list(range(6)))
    token = int(out.sampled_token_ids[ROW])
    payload = decode_unit_payload(eng._published(model)[f"req-{ROW}"])
    extension = payload["model"].get("minicpmo_duplex", {})
    # A rewrite row records the emitted token but the raw draw's measure.
    raw_token = extension.get("sampled_token", token)
    mine = two_stage_logprob(
        z[ROW].double().float(), raw_token, chunk_eos_id=CHUNK_EOS, forbidden_index=forbidden, temperature=temperature
    )
    theirs = payload["logprobs"][0]
    rows.append(
        {
            "step_class": step_class,
            "token": token,
            "raw_token": raw_token,
            "kind": payload["action_kinds"][0],
            "reason": payload["reason_codes"][0] or "",
            "engine_logprob": theirs,
            "trainer_logprob": mine,
            "delta": None if theirs is None else abs(theirs - mine),
        }
    )
print("@rows@")
for row in rows:
    print(json.dumps(row))
"""


def test_policy_measure_matches_engine_all_step_classes(tmp_path):
    vllm_omni = pytest.importorskip("vllm_omni")
    engine_root = pathlib.Path(vllm_omni.__file__).resolve().parents[1]
    if not (engine_root / "tests/model_executor/models/minicpmo_4_5/test_duplex_action_capture.py").is_file():
        pytest.skip("the installed vLLM-Omni is not a source checkout: its sampler test fixture is not on disk")
    verl_root = pathlib.Path(__file__).resolve().parents[3]
    script = tmp_path / "engine_measure_probe.py"
    script.write_text(_PROBE_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(engine_root), str(verl_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0 and ("ImportError" in proc.stderr or "ModuleNotFoundError" in proc.stderr):
        pytest.skip(f"the engine sampler fixture does not import in this environment: {proc.stderr.strip()[-400:]}")
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = [json.loads(line) for line in proc.stdout.split("@rows@\n")[-1].splitlines() if line.strip()]

    assert [row["step_class"] for row in rows] == [
        "listen",
        "content",
        "content",
        "rewrite",
        "content",
        "boundary",
        "turn_eos",
        "listen",
    ]
    assert {row["step_class"] for row in rows} == _STEP_CLASSES
    rewrite = next(row for row in rows if row["step_class"] == "rewrite")
    assert rewrite["kind"] == "forced" and rewrite["reason"] == "mid_turn_listen_rewrite"
    assert rewrite["token"] == 7 and rewrite["raw_token"] != 7  # tts_bos emitted, the listen draw measured
    assert all(row["kind"] == "sampled" for row in rows if row is not rewrite)
    # Every row with a draw — the rewrite's raw draw included — agrees with the engine.
    assert all(row["engine_logprob"] is not None for row in rows)
    deltas = {row["step_class"]: row["delta"] for row in rows}
    assert all(delta <= _MAX_ABS_NATS for delta in deltas.values()), deltas
    boundary = next(row for row in rows if row["step_class"] == "boundary")
    assert boundary["token"] == 5 and boundary["engine_logprob"] == pytest.approx(0.0, abs=1e-6)
