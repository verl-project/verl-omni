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
"""CPU tests for the step-metrics parser of scripts/analyze_step_metrics.py.

The parser is pure stdlib and reads three input formats (JSONL, verl console
records, step-prefixed dict literals); its first round shipped three bugs
that only usage exposed. These tests pin the parsing rules -- malformed or
non-integral steps drop the line, the console prefix's step number is kept,
ANSI escapes are stripped -- plus the attribution record's schema and the
CLI's exit code on unparsable input.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_step_metrics.py"
_spec = importlib.util.spec_from_file_location("analyze_step_metrics", _SCRIPT)
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


# -- line parsing -------------------------------------------------------------


def test_jsonl_line_parses_and_keeps_payload_step():
    step, payload = analyze._parse_line('{"training/global_step": 12, "timing_s/step": 41.2}')
    assert step == 12
    assert payload["timing_s/step"] == 41.2


def test_jsonl_line_strips_ansi_escapes():
    line = '[36m{"training/global_step": 3, "timing_s/step": 2.0}[0m'
    assert analyze._parse_line(line) == (3, {"training/global_step": 3, "timing_s/step": 2.0})


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",  # valid JSON, not an object
        '{"timing_s/step": 1.0}',  # no step
        '{"training/global_step": 12.5, "timing_s/step": 1.0}',  # fractional step
        '{"training/global_step": true, "timing_s/step": 1.0}',  # boolean step
        '{"training/global_step": "NaN", "timing_s/step": 1.0}',  # non-finite step
        "",
    ],
)
def test_jsonl_line_rejects_malformed_steps(line):
    assert analyze._parse_line(line) is None


def test_console_line_parses_numeric_fields_and_prefix_step():
    line = "step:7 - training/global_step:9 - timing_s/step:10.5 - state:running - loss:nan-ish"
    step, payload = analyze._parse_line(line)
    # The prefix step wins; non-numeric fields are dropped, not fatal.
    assert step == 7
    assert payload == {"training/global_step": 9.0, "timing_s/step": 10.5}


def test_console_line_without_numeric_payload_is_dropped():
    assert analyze._parse_line("step:7 - state:running") is None
    assert analyze._parse_line("garbage line") is None


def test_dict_literal_line_parses_and_falls_back_to_prefix_step():
    step, payload = analyze._parse_line("step:9 metrics: {'timing_s/step': 41.2}")
    assert step == 9
    assert payload == {"timing_s/step": 41.2}
    # global_step inside the dict takes precedence over the prefix.
    step, _ = analyze._parse_line("step:9 metrics: {'training/global_step': 11}")
    assert step == 11
    assert analyze._parse_line("step:9 metrics: {broken") is None


def test_load_steps_skips_unparsable_lines(tmp_path):
    log = tmp_path / "metrics.log"
    log.write_text(
        "\n".join(
            [
                "garbage",
                '{"training/global_step": 1, "timing_s/step": 2.0}',
                "step:2 - timing_s/step: 3.0 - update_actor: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    assert analyze.load_steps(str(log)) == [
        (1, {"training/global_step": 1, "timing_s/step": 2.0}),
        (2, {"timing_s/step": 3.0, "update_actor": 1.0}),
    ]


# -- attribution --------------------------------------------------------------


def _full_metrics():
    return {
        "timing_s/step": 10.0,
        "timing_s/update_actor": 4.0,
        "timing_s/sample_wait/generation_bound": 5.0,
        "tq/put/bytes": 3e9,
        "tq/put/seconds": 1.5,
    }


def test_attribute_step_ranks_segments_and_derives_context():
    record = analyze.attribute_step(_full_metrics(), step=4)
    assert record["step"] == 4
    assert record["bound"].startswith("wait:generation_bound(rollout) 50%")
    assert record["ranked"][0] == ("wait:generation_bound(rollout)", 5.0, 0.5)
    assert record["context"]["tq_put_GBps"] == pytest.approx(2.0)


def test_attribute_step_without_step_seconds_keeps_one_schema():
    record = analyze.attribute_step({"timing_s/update_actor": 1.0}, step=5)
    assert record["bound"] == "unknown(step_s<=0)"
    # JSON records keep a constant schema: the populated path's keys exist.
    assert record["ranked"] == []
    assert record["context"] == {}


def test_attribute_step_coerces_nonfinite_inputs_to_zero():
    record = analyze.attribute_step(
        {"timing_s/step": float("inf"), "timing_s/update_actor": float("nan")},
        step=1,
    )
    assert record["step_s"] == 0.0
    assert record["bound"] == "unknown(step_s<=0)"


# -- CLI ----------------------------------------------------------------------


def test_cli_json_output_has_constant_schema(tmp_path, capsys):
    log = tmp_path / "metrics.log"
    log.write_text(
        '{"training/global_step": 1, "timing_s/step": 4.0, "timing_s/update_actor": 1.0}\n'
        '{"training/global_step": 2, "timing_s/update_actor": 1.0}\n',
        encoding="utf-8",
    )
    sys.argv = ["analyze_step_metrics.py", str(log), "--json"]
    analyze.main()
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 2
    assert set(records[0]) == set(records[1])


@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_cli_exits_nonzero_on_unparsable_input(tmp_path, capsys, extra):
    log = tmp_path / "metrics.log"
    log.write_text("nothing parsable here\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        sys.argv = ["analyze_step_metrics.py", str(log), *extra]
        analyze.main()
    assert excinfo.value.code == 1
    assert "no parsable" in capsys.readouterr().err
