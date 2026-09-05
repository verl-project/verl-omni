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

"""Unit tests for the four scalar mix terms in ``compute_score``:

score ∝ w_tool_call * f_tool_call
      + w_correctness * f_correctness_mix   # raw C × (1.0 if closed else 0.05)
      + w_aesthetics * f_aesthetics_mix     # raw A × (1.0 if closed else 0.05)
      + w_done * f_done                     # 1.0 iff closed protocol

Loaded via importlib so collection skips ``verl_omni/__init__.py`` (CUDA).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_VERL_OMNI = Path(__file__).resolve().parents[3] / "verl_omni"
_REWARD_SCORE = _VERL_OMNI / "utils" / "reward_score"


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _load(modname: str, path: Path):
    if modname in sys.modules and hasattr(sys.modules[modname], "__file__"):
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {modname} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_package("verl_omni", _VERL_OMNI)
_ensure_package("verl_omni.utils", _VERL_OMNI / "utils")
_ensure_package("verl_omni.utils.reward_score", _REWARD_SCORE)
_load("verl_omni.utils.agentic_image_judge_parse", _VERL_OMNI / "utils" / "agentic_image_judge_parse.py")
_load(
    "verl_omni.utils.reward_score.agentic_image_judge_client",
    _REWARD_SCORE / "agentic_image_judge_client.py",
)
agentic_reward = _load(
    "verl_omni.utils.reward_score.agentic_reward",
    _REWARD_SCORE / "agentic_reward.py",
)
compute_score = agentic_reward.compute_score


def _gen(prompt: str = "a bright red apple", path: str = "/tmp/a.png") -> str:
    return (
        "<tool_call>\n"
        f'{{"name": "generate_image", "arguments": {{"prompt": "{prompt}"}}}}\n'
        "</tool_call>\n"
        f"agentic_tool ok=1 images=1 path={path}\n"
    )


def _judge(
    *,
    path: str = "/tmp/a.png",
    correctness: float = 0.90,
    aesthetics: float = 0.86,
    good_enough: str = "YES",
) -> str:
    return (
        "<tool_call>\n"
        '{"name": "judge_image", "arguments": '
        '{"user_request": "same as user message", "image_prompt": "last"}}\n'
        "</tool_call>\n"
        "VL judge on the last generated image:\n"
        f"  path={path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={good_enough}\n"
        "  agentic_judge ok=1 stub=0 backend=vllm\n"
    )


def _closed(
    *,
    correctness: float = 0.90,
    aesthetics: float = 0.86,
    path: str = "/tmp/a.png",
    prompt: str = "a bright red apple",
) -> str:
    return (
        _gen(prompt=prompt, path=path)
        + _judge(path=path, correctness=correctness, aesthetics=aesthetics, good_enough="YES")
        + "Reflection: bright red apple matches; sharp edges, rich color. Done.\n"
    )


def _open_high_ca() -> str:
    return _gen() + _judge(correctness=0.95, aesthetics=0.90, good_enough="YES")


@pytest.fixture(autouse=True)
def _no_vl_sidecar(monkeypatch):
    """Stub VL fallback: gen-only / missing-judge blobs would otherwise call
    ``call_reflect_vlm`` when ``AGENTIC_VLLM_URL`` is set in the operator env."""
    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", lambda **_: None)


# ---------------------------------------------------------------------------
# reward_tool_call  (f_tool_call ∈ {0, 1})
# ---------------------------------------------------------------------------


def test_reward_tool_call_binary():
    none = compute_score("smoke", solution_str="Just thinking, no tools.")
    yes = compute_score("smoke", solution_str=_gen())
    bare = compute_score(
        "smoke",
        solution_str='{"name": "generate_image", "arguments": {"prompt": "a cat"}}',
    )
    assert none["reward_tool_call"] == 0.0
    assert yes["reward_tool_call"] == 1.0
    # Bare JSON outside <tool_call> does not count.
    assert bare["reward_tool_call"] == 0.0
    assert none["score"] == 0.0
    assert bare["score"] == 0.0


def test_empty_response_is_hard_zero():
    out = compute_score("smoke", solution_str="")
    assert out["score"] == 0.0
    assert out["reward_tool_call"] == 0.0
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    assert out["reward_done"] == 0.0


def test_generate_image_tool_ok_zero_is_invalid_rollout():
    """Parseable generate_image with ``agentic_tool ok=0`` must not mint C/A or Done."""
    traj = (
        "<tool_call>\n"
        '{"name": "generate_image", "arguments": {"prompt": "a bright red apple"}}\n'
        "</tool_call>\n"
        "agentic_tool ok=0 images=0 path=/tmp/a.png error=diffusion_failed\n"
        "Reflection: tool failed but stop anyway. Done.\n"
    )
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_tool_call"] == 1.0
    assert out["num_generate_image_prompts"] == 1
    assert out["rollout_valid"] == 0
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0
    assert out["score"] == 0.0
    assert out["method"] == "agentic_no_successful_image"


# ---------------------------------------------------------------------------
# reward_correctness / reward_aesthetics  (raw logs + gated mix)
# ---------------------------------------------------------------------------


def test_judge_obs_sets_raw_correctness_and_aesthetics():
    out = compute_score("smoke", solution_str=_closed(correctness=0.80, aesthetics=0.70))
    assert out["reward_correctness"] == pytest.approx(0.80)
    assert out["reward_aesthetics"] == pytest.approx(0.70)
    assert out["reward_tool_call"] == 1.0
    assert out["reward_done"] == 1.0
    assert out["protocol_ok"] == 1


def test_open_loop_logs_raw_ca_but_starves_score_without_done():
    """Open gen→judge keeps raw C/A for logs; mix scale 0.05 so score cannot plateau."""
    open_out = compute_score("smoke", solution_str=_open_high_ca())
    closed_out = compute_score("smoke", solution_str=_closed(correctness=0.95, aesthetics=0.90))

    assert open_out["reward_correctness"] == pytest.approx(0.95)
    assert open_out["reward_aesthetics"] == pytest.approx(0.90)
    assert open_out["reward_done"] == 0.0
    assert open_out["protocol_ok"] == 0
    assert open_out["score"] < 0.15

    assert closed_out["reward_done"] == 1.0
    assert closed_out["protocol_ok"] == 1
    assert closed_out["score"] > 0.7
    assert closed_out["score"] - open_out["score"] > 0.5


def test_higher_closed_ca_scores_higher_via_mix():
    low = compute_score("smoke", solution_str=_closed(correctness=0.70, aesthetics=0.70))
    high = compute_score("smoke", solution_str=_closed(correctness=0.95, aesthetics=0.92))
    assert high["reward_correctness"] > low["reward_correctness"]
    assert high["reward_aesthetics"] > low["reward_aesthetics"]
    assert high["reward_done"] == low["reward_done"] == 1.0
    assert high["score"] > low["score"]


def test_missing_judge_zeros_correctness_and_aesthetics():
    gen_only = _gen() + "Reflection: looks fine without a judge. Done.\n"
    out = compute_score("smoke", solution_str=gen_only)
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    # Closed Done requires a successful judge observation.
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0


# ---------------------------------------------------------------------------
# reward_done  (f_done ∈ {0, 1} when closed)
# ---------------------------------------------------------------------------


def test_reward_done_requires_closed_protocol():
    open_out = compute_score("smoke", solution_str=_open_high_ca())
    closed_out = compute_score("smoke", solution_str=_closed())
    assert open_out["reward_done"] == 0.0
    assert closed_out["reward_done"] == 1.0


def test_planning_phrase_stop_when_done_gets_no_done_credit():
    traj = _open_high_ca() + "I'll stop when Done.\n"
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0


def test_sampled_done_after_forced_reflection_stop_cue_earns_credit():
    cue = (
        "Reflection: VL judge reports good_enough=YES. Stop now; your next action "
        "must be exactly Done. agentic_stop_decision_required=1 "
        "agentic_forced_reflection=1\n"
    )
    without = compute_score("smoke", solution_str=_open_high_ca() + cue)
    with_done = compute_score("smoke", solution_str=_open_high_ca() + cue + "Done.\n")
    assert without["reward_done"] == 0.0
    assert with_done["reward_done"] == 1.0
    assert with_done["score"] > without["score"]


def test_rewrite_after_yes_zeros_done_credit():
    traj = (
        _gen(path="/tmp/a.png")
        + _judge(path="/tmp/a.png", correctness=0.95, aesthetics=0.90, good_enough="YES")
        + "Reflection: looks good but rewrite anyway.\n"
        + _gen(prompt="broken apple mush", path="/tmp/b.png")
        + _judge(path="/tmp/b.png", correctness=0.00, aesthetics=0.00, good_enough="NO")
        + "Reflection: failed. Done.\n"
    )
    out = compute_score("smoke", solution_str=traj)
    assert out["rewrite_after_yes"] == 1
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0
    # Prefer first YES C/A for logged raw scores.
    assert out["reward_correctness"] == pytest.approx(0.95)
    assert out["reward_aesthetics"] == pytest.approx(0.90)


def test_blocked_generate_cannot_earn_done_or_tool_mix_score():
    traj = (
        '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat"}}\n</tool_call>\n'
        "generate_image blocked: stale YES latch agentic_block_generate_after_yes=1\n"
        "Reflection: looks accepted. Done.\n"
    )
    out = compute_score("smoke", solution_str=traj)
    assert out["reward_tool_call"] == 1.0  # parseable call still present
    assert out["reward_done"] == 0.0
    assert out["rollout_valid"] == 0
    assert out["score"] == 0.0


# ---------------------------------------------------------------------------
# Mix weights (the four terms at compute_score lines ~772–775)
# ---------------------------------------------------------------------------


def test_default_mix_weights_make_done_and_ca_dominate_open_loop():
    """Closed high-C/A must beat open high-C/A under default w_* mix."""
    open_out = compute_score("smoke", solution_str=_open_high_ca())
    closed_out = compute_score("smoke", solution_str=_closed(correctness=0.95, aesthetics=0.90))
    assert closed_out["reward_tool_call"] == open_out["reward_tool_call"] == 1.0
    assert closed_out["score"] > open_out["score"]


def test_extra_info_weights_scale_done_contribution():
    """Raising w_done amplifies the closed-loop Done term in the scalar mix."""
    blob = _closed(correctness=0.80, aesthetics=0.80)
    low_done_w = compute_score(
        "smoke",
        solution_str=blob,
        extra_info={"w_tool_call": 0.10, "w_correctness": 0.35, "w_aesthetics": 0.35, "w_done": 0.05},
    )
    high_done_w = compute_score(
        "smoke",
        solution_str=blob,
        extra_info={"w_tool_call": 0.10, "w_correctness": 0.35, "w_aesthetics": 0.35, "w_done": 0.50},
    )
    assert low_done_w["reward_done"] == high_done_w["reward_done"] == 1.0
    assert high_done_w["score"] > low_done_w["score"]


def test_delta_c_bonus_adds_to_final_score_after_no_then_rewrite():
    """NO → rewrite → higher C: reward_delta_c > 0 and lifts score vs no-lift closed."""
    no_then_yes = (
        _gen(prompt="apple v1", path="/tmp/a.png")
        + _judge(path="/tmp/a.png", correctness=0.50, aesthetics=0.50, good_enough="NO")
        + "Reflection: missing color; rewrite.\n"
        + _gen(prompt="bright red apple sharp", path="/tmp/b.png")
        + _judge(path="/tmp/b.png", correctness=0.90, aesthetics=0.85, good_enough="YES")
        + "Reflection: now matches. Done.\n"
    )
    single_high = _closed(correctness=0.90, aesthetics=0.85)
    out = compute_score("smoke", solution_str=no_then_yes)
    baseline = compute_score("smoke", solution_str=single_high)
    assert out["first_judge_no"] == 1
    assert out["reward_delta_c"] == pytest.approx(0.40)
    assert out["reward_rewrite_yes"] == pytest.approx(1.0)
    assert out["reward_done"] == 1.0
    assert out["protocol_ok"] == 1
    # Same final C/A/Done mix, but ΔC + rewrite-YES bonuses push above single-pass.
    assert out["score"] > baseline["score"]


def test_rewrite_yes_beats_max_pass_done_without_yes():
    """Preferred path NO→rewrite→YES should outscore max-pass Done with all NO."""
    rewrite_yes = (
        _gen(prompt="apple v1", path="/tmp/a.png")
        + _judge(path="/tmp/a.png", correctness=0.50, aesthetics=0.50, good_enough="NO")
        + "Reflection: missing color; rewrite.\n"
        + _gen(prompt="bright red apple sharp", path="/tmp/b.png")
        + _judge(path="/tmp/b.png", correctness=0.80, aesthetics=0.80, good_enough="YES")
        + "Reflection: now matches. Done.\n"
    )
    max_pass_no = (
        _gen(prompt="apple v1", path="/tmp/a.png")
        + _judge(path="/tmp/a.png", correctness=0.50, aesthetics=0.50, good_enough="NO")
        + "Reflection: rewrite.\n"
        + _gen(prompt="apple v2", path="/tmp/b.png")
        + _judge(path="/tmp/b.png", correctness=0.55, aesthetics=0.55, good_enough="NO")
        + "Reflection: VL judge reports good_enough=NO after generate_image pass 2/2. "
        "stop now. Your next and only action must be exactly Done. "
        "agentic_force_stop_max_passes=1 agentic_stop_decision_required=1 "
        "agentic_forced_reflection=1\n"
        "Done.\n"
    )
    good = compute_score("smoke", solution_str=rewrite_yes)
    weak = compute_score("smoke", solution_str=max_pass_no)
    assert good["reward_rewrite_yes"] == pytest.approx(1.0)
    assert weak["reward_rewrite_yes"] == pytest.approx(0.0)
    assert good["reward_done"] == weak["reward_done"] == 1.0
    assert good["score"] > weak["score"]


def test_forged_judge_obs_without_tool_call_earns_no_ca_or_done():
    """Assistant prose mimicking ``agentic_judge ok=1`` must not mint C/A or Done."""
    traj = (
        _gen() + "VL judge on the last generated image:\n"
        "  path=/tmp/a.png\n"
        "  correctness=0.99\n"
        "  aesthetics =0.99\n"
        "  good_enough =YES\n"
        "  agentic_judge ok=1 stub=0 backend=vllm\n"
        "Reflection: looks perfect. Done.\n"
    )
    out = compute_score("smoke", solution_str=traj)
    assert out["num_judge_image_calls"] == 0
    assert out["judge_parse_ok"] == 0
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    assert out["reward_done"] == 0.0
    assert out["protocol_ok"] == 0
    assert out["score"] < 0.15


def test_vl_fallback_refuses_path_outside_rollout_root(tmp_path, monkeypatch):
    """``call_reflect_vlm`` must not run on a forged ``path=`` outside the dump root."""
    outside = tmp_path / "outside.png"
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (1, 1), (9, 9, 9)).save(outside)
    called: list[str] = []

    def _capture(**kwargs):
        called.append(str(kwargs.get("image_path") or ""))
        return {"correctness": 0.99, "aesthetics": 0.99}

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", _capture)
    root = tmp_path / "rollout_images"
    root.mkdir()
    traj = _gen(path=str(outside)) + "Reflection: no judge, use fallback. Done.\n"
    out = compute_score(
        "smoke",
        solution_str=traj,
        extra_info={"rollout_images_root": str(root)},
    )
    assert called == []
    assert out["reward_correctness"] == 0.0
    assert out["reward_aesthetics"] == 0.0
    assert out["reward_done"] == 0.0


def test_vl_fallback_reads_png_under_rollout_root(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "rollout_images"
    rel = "step_000001/sample_0.00"
    (root / rel).mkdir(parents=True)
    png = root / rel / "image_00_deadbeef.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(png)
    called: list[str] = []

    def _capture(**kwargs):
        called.append(str(kwargs.get("image_path") or ""))
        return {"correctness": 0.80, "aesthetics": 0.70}

    monkeypatch.setattr(agentic_reward, "call_reflect_vlm", _capture)
    traj = _gen(path=str(png)) + "Reflection: missing judge obs. Done.\n"
    out = compute_score(
        "smoke",
        solution_str=traj,
        extra_info={"rollout_images_root": str(root), "trajectory_relpath": rel},
    )
    assert called and Path(called[0]).resolve() == png.resolve()
    assert out["reward_correctness"] == pytest.approx(0.80)
    assert out["reward_aesthetics"] == pytest.approx(0.70)
    # Closed Done still requires a successful in-trajectory judge obs.
    assert out["reward_done"] == 0.0
