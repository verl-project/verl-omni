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
"""CPU tests for the UniCoT agentic RL parquet builder."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl as builder


def _write_snapshot(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    root = tmp_path / name
    snapshot = root / "snapshots" / "0000000000000000000000000000000000000000"
    snapshot.mkdir(parents=True)
    (snapshot / "metadata.json").write_text(json.dumps(rows))
    return root


def _reflection_row(data_id: str, states: int = 2) -> dict:
    inputs = [f"./images/{data_id}_{index}.png" for index in range(states)]
    outputs: list[str | None] = [f"./images/{data_id}_{index + 1}.png" for index in range(states - 1)] + [None]
    return {
        "data_id": data_id,
        "prompt": f"Reflection prompt {data_id}.",
        "eval": [f"Evaluation {index}." for index in range(states)],
        "eval_summary": [f"Summary {index}." for index in range(states)],
        "edit": ["Improve lighting."] * (states - 1) + ["Everything is good. No editing needed."],
        "input_image": inputs,
        "output_image": outputs,
    }


def _breakdown_row(data_id: str, count: int = 2) -> dict:
    subtasks: list[str | None] = [f"Subtask {index}." for index in range(count)]
    subtasks.extend([None] * (3 - count))
    images: list[str | None] = [f"./images/{data_id}_{index}.png" for index in range(count)]
    images.extend([None] * (3 - count))
    return {
        "data_id": data_id,
        "prompt": f"Breakdown prompt {data_id}.",
        "subtasks": subtasks,
        "subtask_images": images,
    }


def _edit_style_breakdown_row(data_id: str) -> dict:
    """A three-slot row worded the way the hub corpus words it, for an *edit* tool."""
    return {
        "data_id": data_id,
        "prompt": f"Breakdown prompt {data_id}.",
        "subtasks": [
            "An African American librarian floats while reading a book in an underwater library.",
            "Keep the outline of the image unchanged and edit with the following details. "
            "The librarian wears a luxurious gold satin gown.",
            "Keep all previously rendered elements unchanged. Apply visual effects that reinforce the ethereal mood.",
        ],
        "subtask_images": [f"./images/{data_id}_{index}.png" for index in range(3)],
    }


def _no_breakdown_row(data_id: str) -> dict:
    return {
        "data_id": data_id,
        "prompt": f"Simple prompt {data_id}.",
        "subtasks": ["No breakdown needed.", None, None],
        "subtask_images": [None, None, None],
    }


def _build(
    tmp_path: Path,
    *,
    reflection_rows: list[dict],
    breakdown_rows: list[dict],
    train_size: int | None = None,
    val_size: int | None = None,
    mix_ratio: float = 0.5,
    val_ratio: float = 0.2,
    seed: int = 7,
) -> Path:
    output = tmp_path / "output"
    reflection_dir = _write_snapshot(tmp_path, "reflection", reflection_rows) if reflection_rows else ""
    breakdown_dir = _write_snapshot(tmp_path, "breakdown", breakdown_rows) if breakdown_rows else ""
    builder.main_cli(
        reflection_dir=str(reflection_dir),
        breakdown_dir=str(breakdown_dir),
        local_save_dir=str(output),
        train_size=train_size,
        val_size=val_size,
        mix_ratio=mix_ratio,
        seed=seed,
        val_ratio=val_ratio,
    )
    return output


def _read(output: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(output / f"{split}.parquet")


def test_builds_expected_agentic_schema_without_reference_leakage(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1 + i % 3) for i in range(30)],
        breakdown_rows=[_breakdown_row(f"b{i}", 1 + i % 3) for i in range(30)],
        train_size=20,
        val_size=8,
    )
    train = _read(output, "train")

    assert set(train.columns) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert {"unicot_reflection", "unicot_breakdown"}.issubset(set(train["data_source"]))
    for messages in train["prompt"]:
        messages = list(messages)
        assert [message["role"] for message in messages] == ["system", "user"]
        prompt_blob = " ".join(message["content"] for message in messages)
        assert "Summary 0." not in prompt_blob
        assert "Subtask 0." not in prompt_blob
        assert messages[1]["content"].endswith("(≤4 sentences).")


def test_references_and_weights_live_only_in_ground_truth(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}") for i in range(20)],
        breakdown_rows=[_breakdown_row(f"b{i}", 3) for i in range(20)],
        train_size=20,
        val_size=4,
    )
    train = _read(output, "train")

    for reward_model, extra_info in zip(train["reward_model"], train["extra_info"], strict=True):
        ground_truth = reward_model["ground_truth"]
        assert all(f"w_{dim}" in ground_truth for dim in builder.DIMS)
        assert not any(key.startswith("w_") for key in extra_info)
        if ground_truth["task_type"] == "plan":
            assert len(ground_truth["reference_subtasks"]) == 3
            assert ground_truth.get("reference_steps") is None
        else:
            assert [step["action"] for step in ground_truth["reference_steps"]] == ["continue", "stop"]
            assert ground_truth.get("reference_subtasks") is None


def test_plan_references_keep_each_part_of_the_stateless_generate_prompt(tmp_path):
    """Plan references must drop the edit framing, because the tool cannot edit.

    The hub subtasks are written for an image-edit tool ("keep the outline unchanged
    and edit with the following details"), and the harness has none. Scoring the
    policy against an edit instruction would reward text the tool cannot act on, so the
    builder strips the framing and keeps the part. ``expected_num_images`` still equals
    the source slot count.
    """
    output = _build(
        tmp_path,
        reflection_rows=[],
        breakdown_rows=[_edit_style_breakdown_row(f"b{i}") for i in range(6)],
        train_size=None,
        val_size=None,
    )
    rows = pd.concat([_read(output, "train"), _read(output, "val")])

    for reward_model in rows["reward_model"]:
        ground_truth = reward_model["ground_truth"]
        references = list(ground_truth["reference_subtasks"])
        assert ground_truth["task_type"] == "plan"
        assert ground_truth["expected_num_images"] == 3
        assert len(references) == 3
        # Each reference is its own part, not the accumulated chain up to it.
        assert "gold satin gown" in references[1]
        assert references[0] not in references[1]
        assert "ethereal mood" in references[2]
        assert references[0] not in references[2]
        assert not any("Keep the outline" in reference for reference in references)
        assert not any("Keep all previously rendered" in reference for reference in references)

    report = json.loads((output / "build_report.json").read_text())
    assert report["plan_reference_transform"] == builder.PLAN_REFERENCE_TRANSFORM


def test_prompts_frame_the_agent_job_as_a_rendering_recipe():
    """Guard the rewrite contract against a silent rebase revert.

    The prompts are the only lever that makes GRPO explore prompt space, so both
    must frame the agent's job as turning a user request into a *rendering recipe*
    for ``generate_image``. Without that framing the observed failure returned:
    each pass appended one legibility assertion, the tool read the same recipe and
    returned the same defect, and the rollout burned all passes without improving.
    """
    for name, prompt in (
        ("reflect", builder.REFLECT_SYSTEM_PROMPT),
        ("plan", builder.PLAN_SYSTEM_PROMPT),
    ):
        # Normalise the hard-wrapped prose so an assertion cannot hinge on where a
        # phrase happens to fall across a line.
        normalized = " ".join(prompt.split())
        assert "rendering recipe" in normalized, name
        assert "generate_image" in normalized, name
        # Typography is the axis the failing rollouts needed most; dropping the
        # concrete guidance reverts to "make the text legible" restatements.
        assert "typography" in normalized, name
    assert "rewrite" in builder.REFLECT_SYSTEM_PROMPT
    assert "one complete diffusion prompt" in " ".join(builder.PLAN_SYSTEM_PROMPT.split())
    assert builder.REFLECT_SYSTEM_PROMPT != builder.PLAN_SYSTEM_PROMPT


def test_plan_prompt_asks_for_an_own_turn_and_a_whole_plan_call():
    """Plan mode needs both halves of the protocol stated explicitly.

    The plan is a turn of its own (the loop only reopens a tool-call-free turn when it
    carries a plan), and every later ``generate_image`` carries the whole list, because
    the items are parts of one prompt rather than one render each. Getting either wrong
    reproduces ``sample_9004``: no plan written, then a reflect-style rewrite loop that
    the protocol forbids.
    """
    normalized = " ".join(builder.PLAN_SYSTEM_PROMPT.split())

    assert "no tool call" in normalized
    assert "carries the whole plan" in normalized
    assert "one item per line" in normalized
    # The list is one image; a single item is not a call. Reflect mode must not inherit
    # the plan-only turn contract.
    assert "never call the tool with a single item" in normalized
    assert "no tool call on this turn" not in " ".join(builder.REFLECT_SYSTEM_PROMPT.split())


def test_plan_prompt_forbids_meta_steps_in_the_plan():
    """Every plan item must describe content, not a process step.

    Observed failure (``sample_9004``): the model returned

        1. Generate a vertical composition flat design poster ... (the request back)
        2. Generate the image.
        3. Judge the image.
        4. Done.

    Items 2-4 are the protocol read back, and ``generate_image`` would have tried to draw
    those words. An earlier revision of this prompt named those exact lines as wrong and
    the model emitted them anyway: quoting the forbidden sentence puts its tokens in
    context, and the numbered protocol the prompt was explained by was itself a template
    for the numbered list the reply had to be. So the rule is stated as a property of an
    item, no procedure-shaped line is printed anywhere, and "one item is a complete plan"
    is stated outright — a reply that must *look* like a list is what the padding served.
    """
    normalized = " ".join(builder.PLAN_SYSTEM_PROMPT.split())

    # The list as a whole is what the tool receives.
    assert "one complete diffusion prompt" in normalized
    # The rule, stated as a property rather than as a phrase to copy.
    assert "is a step and not an item" in normalized
    assert "No item may restate an earlier one word for word" in normalized
    # Nothing plan-shaped is printed: no numbered protocol, no sentence-final "Done.".
    numbered = [line for line in builder.PLAN_SYSTEM_PROMPT.splitlines() if re.match(r"\s*\d+[.)]\s", line)]
    assert len(numbered) == 2, numbered
    assert all("market street" in line or "fishmonger" in line for line in numbered), numbered
    assert "Done." not in builder.PLAN_SYSTEM_PROMPT
    assert "Protocol" not in builder.PLAN_SYSTEM_PROMPT
    # The worked example, and the count contract as a task-sized decomposition that allows
    # one item.
    assert "Both items together are the prompt for one photograph" in normalized
    assert "Three is the ceiling, not a target" in normalized
    assert "One item is a complete plan" in normalized
    assert "never pad the list" in normalized


def test_neither_system_prompt_prints_a_numbered_protocol_plan():
    """The prompt must not hand the model a numbered list in the reply's own grammar.

    The protocol used to be a numbered list — "1. Call generate_image. 2. Judge it." — in
    the same grammar the plan had to be written in, and the model continued it. Reading
    both prompts for numbered lines keeps the two from drifting back together: the reflect
    prompt has none, and the plan prompt has only its worked example.
    """
    numbered = re.compile(r"(?m)^\s*\d+[.)]\s")

    assert numbered.search(builder.REFLECT_SYSTEM_PROMPT) is None
    # The plan prompt carries only the two items of its worked example.
    assert len(numbered.findall(builder.PLAN_SYSTEM_PROMPT)) == 2
    assert not re.search(
        r"(?m)^\s*\d+[.)]\s+(?:Call|Generate the image|Judge the image|Reflect on)",
        builder.PLAN_SYSTEM_PROMPT,
    )


def test_plan_and_reflect_rows_use_task_specific_system_prompts(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}") for i in range(20)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(20)],
        train_size=20,
        val_size=4,
    )
    train = _read(output, "train")
    prompts_by_type = {
        extra["task_type"]: prompt[0]["content"]
        for prompt, extra in zip(train["prompt"], train["extra_info"], strict=True)
    }
    assert prompts_by_type["plan"] == builder.PLAN_SYSTEM_PROMPT
    assert prompts_by_type["reflect"] == builder.REFLECT_SYSTEM_PROMPT
    assert prompts_by_type["plan"] != prompts_by_type["reflect"]


def test_no_breakdown_rows_become_single_image_reflect_tasks(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[],
        breakdown_rows=[_no_breakdown_row(f"n{i}") for i in range(20)],
        train_size=None,
        val_size=None,
    )
    rows = pd.concat([_read(output, "train"), _read(output, "val")])
    for reward_model in rows["reward_model"]:
        ground_truth = reward_model["ground_truth"]
        assert ground_truth["task_type"] == "reflect"
        # No reference trajectory exists for this sentinel, so there is no image
        # budget to derive; the reward must not enforce a fabricated cap.
        assert ground_truth["expected_num_images"] is None
        assert ground_truth["plan_expected"] is False
    assert set(rows["data_source"]) == {builder.REFLECT_DATA_SOURCE}
    for extra in rows["extra_info"]:
        assert extra["unicot_source"] == builder.UNICOT_BREAKDOWN_DATASET_ID


def test_rejections_are_reported_and_dropped(tmp_path):
    bad = _reflection_row("bad")
    bad["edit"][0] = ""
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    all_rows = pd.concat([_read(output, "train"), _read(output, "val")])

    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "bad"
    assert "bad" not in {extra["data_id"] for extra in all_rows["extra_info"]}
    assert report["source_configs"]["reflection"]["unicot_reflection_cleaner"]
    assert report["partition_id"].startswith("partition_")


def test_full_mode_uses_every_valid_row_and_splits_are_disjoint(tmp_path):
    reflection = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(40)]
    breakdown = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(40)]
    output = _build(tmp_path, reflection_rows=reflection, breakdown_rows=breakdown)
    train = _read(output, "train")
    val = _read(output, "val")
    train_ids = {(extra["unicot_source"], extra["data_id"]) for extra in train["extra_info"]}
    val_ids = {(extra["unicot_source"], extra["data_id"]) for extra in val["extra_info"]}

    assert len(train) + len(val) == 80
    assert train_ids.isdisjoint(val_ids)


def test_full_mode_is_order_independent_and_deterministic(tmp_path):
    reflection = [_reflection_row(f"r{i}", 1 + i % 3) for i in range(30)]
    breakdown = [_breakdown_row(f"b{i}", 1 + i % 3) for i in range(30)]
    first = _build(tmp_path / "first", reflection_rows=reflection, breakdown_rows=breakdown)
    second = _build(
        tmp_path / "second",
        reflection_rows=list(reversed(reflection)),
        breakdown_rows=list(reversed(breakdown)),
    )

    assert list(_read(first, "train")["extra_info"]) == list(_read(second, "train")["extra_info"])
    assert list(_read(first, "val")["extra_info"]) == list(_read(second, "val")["extra_info"])


def test_size_caps_apply_requested_mix_ratio(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(40)],
        train_size=20,
        val_size=8,
        mix_ratio=0.25,
    )
    train = _read(output, "train")
    reflect_count = sum(extra["task_type"] == "reflect" for extra in train["extra_info"])
    assert len(train) == 20
    assert reflect_count == 5


def test_requires_a_source_and_valid_ratios(tmp_path):
    common = {
        "reflection_dir": "",
        "breakdown_dir": "",
        "local_save_dir": str(tmp_path),
        "train_size": None,
        "val_size": None,
        "mix_ratio": 0.5,
        "seed": 7,
        "val_ratio": 0.2,
    }
    with pytest.raises(SystemExit):
        builder.main_cli(**common)

    reflection_dir = _write_snapshot(tmp_path, "reflection", [_reflection_row("r")])
    common["reflection_dir"] = str(reflection_dir)
    common["mix_ratio"] = 1.1
    with pytest.raises(SystemExit):
        builder.main_cli(**common)


def test_mismatched_transition_uris_are_rejected(tmp_path):
    bad = _reflection_row("bad", states=2)
    bad["output_image"][0] = "./images/not_the_next_input.png"
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "bad"
    assert report["rejections"][0]["reason"] == "transition_hash_mismatch"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_empty_reflection_image_uris_are_rejected(tmp_path, blank):
    bad = _reflection_row("blank", states=2)
    bad["input_image"][0] = blank
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"good{i}", 1) for i in range(20)] + [bad],
        breakdown_rows=[],
    )
    report = json.loads((output / "build_report.json").read_text())
    assert report["rejection_count"] == 1
    assert report["rejections"][0]["data_id"] == "blank"
    assert report["rejections"][0]["reason"] == "missing_image"


def test_hub_refs_main_beats_lexicographically_later_snapshot(tmp_path):
    root = tmp_path / "reflection"
    stale_sha = "ffffffffffffffffffffffffffffffffffffffff"
    main_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    stale = root / "snapshots" / stale_sha
    current = root / "snapshots" / main_sha
    stale.mkdir(parents=True)
    current.mkdir(parents=True)
    (stale / "metadata.json").write_text(json.dumps([_reflection_row("stale", 1)]))
    (current / "metadata.json").write_text(json.dumps([_reflection_row(f"keep{i}", 1) for i in range(20)]))
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(f"{main_sha}\n")

    output = tmp_path / "output"
    builder.main_cli(
        reflection_dir=str(root),
        breakdown_dir="",
        local_save_dir=str(output),
        train_size=None,
        val_size=None,
        mix_ratio=0.5,
        seed=7,
        val_ratio=0.2,
    )
    ids = {extra["data_id"] for extra in pd.concat([_read(output, "train"), _read(output, "val")])["extra_info"]}
    assert "stale" not in ids
    assert "keep0" in ids


def test_capped_size_fails_closed_when_mix_cannot_be_met(tmp_path):
    with pytest.raises(SystemExit, match="cannot meet mix_ratio"):
        _build(
            tmp_path,
            reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
            breakdown_rows=[],
            train_size=20,
            val_size=None,
            mix_ratio=0.5,
        )


def test_capped_size_fails_closed_when_pool_is_smaller_than_requested(tmp_path):
    with pytest.raises(SystemExit, match="requested 100"):
        _build(
            tmp_path,
            reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
            breakdown_rows=[],
            train_size=100,
            val_size=None,
            mix_ratio=1.0,
        )


def test_build_report_records_requested_and_actual_size(tmp_path):
    output = _build(
        tmp_path,
        reflection_rows=[_reflection_row(f"r{i}", 1) for i in range(40)],
        breakdown_rows=[_breakdown_row(f"b{i}") for i in range(40)],
        train_size=20,
        val_size=8,
        mix_ratio=0.25,
    )
    report = json.loads((output / "build_report.json").read_text())
    train = report["splits"]["train"]
    assert train["requested_size"] == 20
    assert train["actual_size"] == 20
    assert train["total"] == 20
    assert train["shortfall"] is None
