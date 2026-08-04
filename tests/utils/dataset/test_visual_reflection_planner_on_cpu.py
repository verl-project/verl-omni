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
"""CPU tests for atomic planning, synthesis results, manifests, and ledgers."""

import copy
import json

import pytest

from tests.utils.dataset._visual_reflection_fixtures import (
    make_visual_reflection_trajectory,
    write_rgb_png,
)
from verl_omni.utils.dataset.visual_reflection import (
    LocalImageResolver,
    RejectionLedger,
    RejectionReason,
    VisualReflectionDataError,
    assign_source_splits,
    build_data_manifest,
    derive_manifest_id,
    format_reflection_response,
    make_split_provenance,
    plan_atomic_examples,
    validate_atomic_example,
)
from verl_omni.utils.dataset.visual_reflection.contracts import (
    derive_pair_source_dedup_key,
    derive_trajectory_id,
)
from verl_omni.utils.dataset.visual_reflection.partition import derive_source_seed, derive_split_for_dedup_key
from verl_omni.utils.dataset.visual_reflection.sources import (
    adapt_echo_pair_record,
    adapt_midjourney_prompt_record,
    build_pair_synthesis_request,
    build_prompt_synthesis_request,
    make_draft_generation_request,
    make_edit_synthesis_request,
    make_edit_verification_request,
    make_image_synthesis_result,
    make_pair_verification_request,
    make_reflection_synthesis_request,
    make_reflection_synthesis_result,
    make_synthesis_result,
    make_terminal_verification_request,
    make_verification_result,
    synthesis_request_source,
)


def _recipe():
    return {
        "sources": [
            {
                "dataset_id": "example/visual-reflection",
                "config": "default",
                "split": "train",
                "revision": "a" * 40,
                "license": "apache-2.0",
                "pipeline_variant": "direct_parse",
            }
        ],
        "code_revision": "18a9934d34f7b77b7f6025bfed8a5c691377ca11",
        "models": {"generator": None, "reflector": None, "verifier": None, "editor": None},
        "prompt_template_hashes": {},
        "seeds": {},
        "converter_config": {"adapter": "example_direct_parser", "adapter_version": "v1"},
        "partition": make_split_provenance(seed="17", ratios={"train": 0.8, "validation": 0.1, "test": 0.1}),
    }


def _source_partition(trajectory, split):
    ratios = {"train": 0.0, "validation": 0.0, "test": 0.0}
    ratios[split] = 1.0
    partition = make_split_provenance(ratios=ratios, seed="test")
    assignment = assign_source_splits([trajectory], ratios=ratios, seed="test")[
        (trajectory["source_dataset"], trajectory["source_record_id"])
    ]
    return assignment, partition


def test_manifest_accepts_arbitrary_pinned_direct_parse_source():
    manifest_id = derive_manifest_id(**_recipe())

    assert manifest_id.startswith("manifest_")


@pytest.mark.parametrize("edit_turns", [0, 1, 2])
def test_atomic_planner_emits_exact_counts_and_terminal_t2i_target(tmp_path, edit_turns):
    trajectory = make_visual_reflection_trajectory(
        tmp_path,
        edit_turns=edit_turns,
        source_record_id=f"row-{edit_turns}",
    )

    source_split, partition = _source_partition(trajectory, "train")
    atoms = plan_atomic_examples(trajectory, source_split=source_split, partition=partition)

    assert len(atoms) == 2 * (edit_turns + 1)
    assert [atom["task"] for atom in atoms] == ["t2i"] + ["reflect"] * (edit_turns + 1) + ["edit"] * edit_turns
    assert atoms[0]["target_image"] == trajectory["images"][-1]
    assert atoms[0]["target_state_index"] == edit_turns
    assert {atom["partition_id"] for atom in atoms} == {partition["partition_id"]}
    assert {atom["source_dedup_key"] for atom in atoms} == {source_split["dedup_key"]}
    assert [validate_atomic_example(atom, partition=partition, source_record=trajectory) for atom in atoms] == atoms
    reflect_atoms = [atom for atom in atoms if atom["task"] == "reflect"]
    edit_atoms = [atom for atom in atoms if atom["task"] == "edit"]
    for index, atom in enumerate(reflect_atoms):
        assert atom["current_image"] == trajectory["images"][index]
        assert atom["response"] == trajectory["steps"][index]
        assert format_reflection_response(atom["response"]).startswith("Reflection:")
    for index, atom in enumerate(edit_atoms):
        assert atom["current_image"] == trajectory["images"][index]
        assert atom["target_image"] == trajectory["images"][index + 1]
        assert atom["edit"] == trajectory["steps"][index]["edit"]


def test_atomic_planner_is_deterministic_and_does_not_mutate_trajectory(tmp_path):
    trajectory = make_visual_reflection_trajectory(tmp_path, edit_turns=2)
    before = copy.deepcopy(trajectory)

    source_split, partition = _source_partition(trajectory, "validation")
    first = plan_atomic_examples(trajectory, source_split=source_split, partition=partition)
    second = plan_atomic_examples(trajectory, source_split=source_split, partition=partition)

    assert first == second
    assert len({atom["atom_id"] for atom in first}) == len(first)
    assert trajectory == before


def test_planner_validates_entire_trajectory_before_emitting_atoms():
    invalid = {
        "trajectory_id": "traj",
        "manifest_id": "manifest",
        "source_dataset": "dataset",
        "source_record_id": "row",
        "pipeline_variant": "direct_parse",
        "prompt": "prompt",
        "images": [{"uri": "a.png", "sha256": "a" * 64}],
        "steps": [{"reflection": "not done", "action": "continue", "edit": "fix"}],
    }

    with pytest.raises(VisualReflectionDataError):
        plan_atomic_examples(invalid, source_split={}, partition={})


def test_planner_recomputes_split_before_atomic_expansion(tmp_path):
    trajectory = make_visual_reflection_trajectory(tmp_path, edit_turns=0)
    source_split, partition = _source_partition(trajectory, "train")
    tampered = {**source_split, "split": "test"}

    with pytest.raises(VisualReflectionDataError) as error:
        plan_atomic_examples(trajectory, source_split=tampered, partition=partition)

    assert error.value.reason is RejectionReason.INVALID_SPLIT


def test_pair_planner_turn_limit_error_is_source_neutral(tmp_path):
    trajectory = make_visual_reflection_trajectory(
        tmp_path,
        edit_turns=2,
        source_dataset="example/image-pairs",
        pipeline_variant="pair_0_1_turn",
    )
    source = {
        "source_dataset": trajectory["source_dataset"],
        "source_record_id": trajectory["source_record_id"],
        "pipeline_variant": trajectory["pipeline_variant"],
        "prompt": trajectory["prompt"],
        "reference_image": trajectory["images"][-1],
        "dedup_key": derive_pair_source_dedup_key(trajectory["prompt"], trajectory["images"][-1]),
    }
    source_split, partition = _source_partition(source, "train")

    with pytest.raises(VisualReflectionDataError) as error:
        plan_atomic_examples(
            trajectory,
            source_split=source_split,
            partition=partition,
            source_record=source,
        )

    assert str(error.value) == "turn_limit_exhausted: Pair-source trajectories may contain at most one edit"


def test_pair_planner_reference_error_is_source_neutral(tmp_path):
    trajectory = make_visual_reflection_trajectory(
        tmp_path,
        edit_turns=1,
        source_dataset="example/image-pairs",
        pipeline_variant="pair_0_1_turn",
    )
    source = {
        "source_dataset": trajectory["source_dataset"],
        "source_record_id": trajectory["source_record_id"],
        "pipeline_variant": trajectory["pipeline_variant"],
        "prompt": trajectory["prompt"],
        "reference_image": trajectory["images"][0],
        "dedup_key": derive_pair_source_dedup_key(trajectory["prompt"], trajectory["images"][0]),
    }
    source_split, partition = _source_partition(source, "train")

    with pytest.raises(VisualReflectionDataError) as error:
        plan_atomic_examples(
            trajectory,
            source_split=source_split,
            partition=partition,
            source_record=source,
        )

    assert str(error.value) == (
        "incompatible_edit_target: One-edit pair-source trajectory must terminate at the reference image"
    )


def test_persisted_atom_validator_rejects_content_or_id_tampering(tmp_path):
    trajectory = make_visual_reflection_trajectory(tmp_path, edit_turns=0)
    source_split, partition = _source_partition(trajectory, "train")
    atom = plan_atomic_examples(trajectory, source_split=source_split, partition=partition)[0]

    with pytest.raises(VisualReflectionDataError):
        validate_atomic_example(
            {**atom, "prompt": "tampered prompt"},
            partition=partition,
            source_record=trajectory,
        )
    with pytest.raises(VisualReflectionDataError):
        validate_atomic_example(
            {**atom, "atom_id": f"atom_{'0' * 64}"},
            partition=partition,
            source_record=trajectory,
        )


@pytest.mark.parametrize("edit_turns", [0, 1])
def test_synthesis_result_enforces_accepted_and_rejected_shapes(tmp_path, edit_turns):
    trajectory = make_visual_reflection_trajectory(tmp_path, edit_turns=edit_turns)

    source = adapt_midjourney_prompt_record({"id": "mj-1", "prompt": trajectory["prompt"]})
    source_split = assign_source_splits([source], ratios={"train": 1.0, "validation": 0.0, "test": 0.0}, seed=7)[
        (source["source_dataset"], source["source_record_id"])
    ]
    request = build_prompt_synthesis_request(
        source,
        manifest_id="manifest-synthesis",
        source_split=source_split,
        partition=make_split_provenance(
            ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
            seed=7,
        ),
        seed=7,
        max_attempts=3,
        max_edit_turns=2,
    )
    synthesized = {
        **trajectory,
        "manifest_id": request["manifest_id"],
        "source_dataset": request["source_dataset"],
        "source_record_id": request["source_record_id"],
        "pipeline_variant": request["pipeline_variant"],
        "prompt": request["prompt"],
    }
    synthesized["trajectory_id"] = derive_trajectory_id(
        manifest_id=synthesized["manifest_id"],
        source_dataset=synthesized["source_dataset"],
        source_record_id=synthesized["source_record_id"],
        pipeline_variant=synthesized["pipeline_variant"],
        prompt=synthesized["prompt"],
        images=synthesized["images"],
        steps=synthesized["steps"],
    )
    draft_request = make_draft_generation_request(parent_request=request, attempt=0)
    image_synthesis_results = [
        make_image_synthesis_result(
            parent_request=request,
            request=draft_request,
            output_image=synthesized["images"][0],
        )
    ]
    reflection_synthesis_results = []
    for turn_index, (image, step) in enumerate(zip(synthesized["images"], synthesized["steps"], strict=True)):
        reflection_request = make_reflection_synthesis_request(
            parent_request=request,
            current_image=image,
            turn_index=turn_index,
            attempt=0,
        )
        reflection_synthesis_results.append(
            make_reflection_synthesis_result(
                parent_request=request,
                request=reflection_request,
                response_text=format_reflection_response(step),
            )
        )
    verification_results = []
    for turn_index, step in enumerate(synthesized["steps"][:-1]):
        edit_synthesis_request = make_edit_synthesis_request(
            parent_request=request,
            current_image=synthesized["images"][turn_index],
            edit=step["edit"],
            turn_index=turn_index,
            attempt=0,
        )
        image_synthesis_results.append(
            make_image_synthesis_result(
                parent_request=request,
                request=edit_synthesis_request,
                output_image=synthesized["images"][turn_index + 1],
            )
        )
        edit_request = make_edit_verification_request(
            parent_request=request,
            current_image=synthesized["images"][turn_index],
            target_image=synthesized["images"][turn_index + 1],
            edit=step["edit"],
            turn_index=turn_index,
            attempt=0,
        )
        verification_results.append(
            make_verification_result(
                parent_request=request,
                request=edit_request,
                verdict="accept",
                reason="The edit is faithful and improves the image.",
            )
        )
    terminal_request = make_terminal_verification_request(
        parent_request=request,
        current_image=synthesized["images"][-1],
        proposed_step=synthesized["steps"][-1],
        turn_index=edit_turns,
        attempt=0,
    )
    verification_results.append(
        make_verification_result(
            parent_request=request,
            request=terminal_request,
            verdict="accept",
            reason="The terminal state satisfies the prompt.",
        )
    )

    accepted = make_synthesis_result(
        request=request,
        status="accepted",
        attempt=0,
        trajectory=synthesized,
        image_synthesis_results=image_synthesis_results,
        reflection_synthesis_results=reflection_synthesis_results,
        verification_results=verification_results,
    )
    rejected = make_synthesis_result(
        request=request,
        status="rejected",
        attempt=2,
        rejection_reason="turn_limit_exhausted",
    )

    assert accepted["trajectory"] == synthesized
    assert accepted["request"] == request
    assert accepted["source_split"]["split"] == "train"
    reordered = make_synthesis_result(
        request=request,
        status="accepted",
        attempt=0,
        trajectory=synthesized,
        image_synthesis_results=list(reversed(image_synthesis_results)),
        reflection_synthesis_results=list(reversed(reflection_synthesis_results)),
        verification_results=list(reversed(verification_results)),
    )
    assert reordered == accepted
    with pytest.raises(VisualReflectionDataError):
        plan_atomic_examples(
            accepted["trajectory"],
            source_split=accepted["source_split"],
            partition=accepted["request"]["partition"],
        )
    assert (
        plan_atomic_examples(
            accepted["trajectory"],
            source_split=accepted["source_split"],
            partition=accepted["request"]["partition"],
            source_record=synthesis_request_source(accepted["request"]),
        )[0]["task"]
        == "t2i"
    )
    forged_key = "0" * 64
    forged_split = {
        **accepted["source_split"],
        "dedup_key": forged_key,
        "split": derive_split_for_dedup_key(forged_key, partition=accepted["request"]["partition"]),
    }
    with pytest.raises(VisualReflectionDataError) as dedup_error:
        plan_atomic_examples(
            accepted["trajectory"],
            source_split=forged_split,
            partition=accepted["request"]["partition"],
            source_record=synthesis_request_source(accepted["request"]),
        )
    assert dedup_error.value.reason is RejectionReason.INVALID_SPLIT
    assert rejected["trajectory"] is None
    with pytest.raises(VisualReflectionDataError):
        make_synthesis_result(
            request=request,
            status="accepted",
            attempt=0,
            rejection_reason="bad",
        )
    with pytest.raises(VisualReflectionDataError) as verifier_error:
        make_synthesis_result(
            request=request,
            status="accepted",
            attempt=0,
            trajectory=synthesized,
            image_synthesis_results=image_synthesis_results,
            reflection_synthesis_results=reflection_synthesis_results,
        )
    assert verifier_error.value.reason is RejectionReason.VERIFIER_REJECTED
    with pytest.raises(VisualReflectionDataError) as retry_error:
        make_synthesis_result(
            request=request,
            status="retry",
            attempt=2,
            rejection_reason="try again",
        )
    assert retry_error.value.reason is RejectionReason.TURN_LIMIT_EXHAUSTED


@pytest.mark.parametrize("edit_turns", [0, 1])
def test_echo_synthesis_result_requires_exact_zero_or_one_turn_quality_gates(tmp_path, edit_turns):
    partition = make_split_provenance(
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        seed=5,
    )
    recipe = {
        "sources": [
            {
                "dataset_id": "Yejy53/Echo-4o-Image",
                "config": "Instruction-Following-Image",
                "split": "train",
                "revision": "c7341967e86c63beca0953bc5536fd99ee7bd303",
                "license": "mit",
                "pipeline_variant": "pair_0_1_turn",
            }
        ],
        "code_revision": "18a9934d34f7b77b7f6025bfed8a5c691377ca11",
        "models": {
            "generator": {"model_id": "model/generator", "revision": "a" * 40},
            "reflector": {"model_id": "model/reflector", "revision": "b" * 40},
            "verifier": {"model_id": "model/verifier", "revision": "c" * 40},
            "editor": None,
        },
        "prompt_template_hashes": {"reflector": "1" * 64, "verifier": "2" * 64},
        "seeds": {"root": 321},
        "converter_config": {"pair_max_attempts": 2},
        "partition": partition,
    }
    manifest_id = derive_manifest_id(**recipe)
    base = make_visual_reflection_trajectory(
        tmp_path,
        edit_turns=edit_turns,
        source_record_id=f"pair-image-{edit_turns}",
        manifest_id=manifest_id,
    )
    reference_uri = base["images"][-1]["uri"]
    if edit_turns == 0:
        reference_uri = "public_reference.png"
        write_rgb_png(tmp_path / reference_uri, (1, 2, 3))
    source = adapt_echo_pair_record(
        {
            "task_type": "t2i",
            "instruction": base["prompt"],
            "output_image": reference_uri,
        },
        image_resolver=LocalImageResolver(tmp_path),
    )
    source_split = assign_source_splits(
        [source],
        ratios=partition["ratios"],
        seed=partition["seed"],
    )[(source["source_dataset"], source["source_record_id"])]
    request = build_pair_synthesis_request(
        source,
        manifest_id=manifest_id,
        source_split=source_split,
        partition=partition,
        seed=derive_source_seed(recipe["seeds"]["root"], source),
        max_attempts=2,
    )
    trajectory = {
        **base,
        "source_dataset": source["source_dataset"],
        "source_record_id": source["source_record_id"],
        "pipeline_variant": source["pipeline_variant"],
        "prompt": source["prompt"],
    }
    trajectory["trajectory_id"] = derive_trajectory_id(
        manifest_id=trajectory["manifest_id"],
        source_dataset=trajectory["source_dataset"],
        source_record_id=trajectory["source_record_id"],
        pipeline_variant=trajectory["pipeline_variant"],
        prompt=trajectory["prompt"],
        images=trajectory["images"],
        steps=trajectory["steps"],
    )
    draft_request = make_draft_generation_request(parent_request=request, attempt=0)
    draft_result = make_image_synthesis_result(
        parent_request=request,
        request=draft_request,
        output_image=trajectory["images"][0],
    )
    reflection_results = []
    for turn_index, (image, step) in enumerate(zip(trajectory["images"], trajectory["steps"], strict=True)):
        reflection_request = make_reflection_synthesis_request(
            parent_request=request,
            current_image=image,
            turn_index=turn_index,
            attempt=0,
        )
        reflection_results.append(
            make_reflection_synthesis_result(
                parent_request=request,
                request=reflection_request,
                response_text=format_reflection_response(step),
            )
        )
    verification_results = []
    if edit_turns:
        pair_request = make_pair_verification_request(
            parent_request=request,
            draft_image=trajectory["images"][0],
            proposed_step=trajectory["steps"][0],
            attempt=0,
        )
        verification_results.append(
            make_verification_result(
                parent_request=request,
                request=pair_request,
                verdict="accept",
                reason="The reference is a compatible one-edit target.",
            )
        )
    terminal_request = make_terminal_verification_request(
        parent_request=request,
        current_image=trajectory["images"][-1],
        proposed_step=trajectory["steps"][-1],
        turn_index=edit_turns,
        attempt=0,
    )
    terminal_result = make_verification_result(
        parent_request=request,
        request=terminal_request,
        verdict="accept",
        reason="The terminal image satisfies the prompt.",
    )
    verification_results.append(terminal_result)

    result = make_synthesis_result(
        request=request,
        status="accepted",
        attempt=0,
        trajectory=trajectory,
        image_synthesis_results=[draft_result],
        reflection_synthesis_results=reflection_results,
        verification_results=verification_results,
    )

    assert result["status"] == "accepted"
    assert len(result["verification_results"]) == edit_turns + 1
    manifest = build_data_manifest(
        **recipe,
        accepted_trajectories=[],
        accepted_source_splits=[],
        accepted_synthesis_results=[result],
        image_asset_roots={source["source_dataset"]: tmp_path},
        rejection_ledger=RejectionLedger(manifest_id),
    )
    assert source["reference_image"]["sha256"] in manifest["image_hashes"]
    if edit_turns == 0:
        assert len(manifest["image_hashes"]) == 2
    with pytest.raises(VisualReflectionDataError) as missing_error:
        make_synthesis_result(
            request=request,
            status="accepted",
            attempt=0,
            trajectory=trajectory,
            image_synthesis_results=[draft_result],
            reflection_synthesis_results=reflection_results,
            verification_results=verification_results[:-1],
        )
    assert missing_error.value.reason is RejectionReason.VERIFIER_REJECTED

    retry_terminal = make_verification_result(
        parent_request=request,
        request=terminal_request,
        verdict="retry",
        reason="The terminal state still needs review.",
    )
    with pytest.raises(VisualReflectionDataError) as verdict_error:
        make_synthesis_result(
            request=request,
            status="accepted",
            attempt=0,
            trajectory=trajectory,
            image_synthesis_results=[draft_result],
            reflection_synthesis_results=reflection_results,
            verification_results=[*verification_results[:-1], retry_terminal],
        )
    assert verdict_error.value.reason is RejectionReason.VERIFIER_REJECTED


def test_manifest_audits_synthesis_request_seed_bounds_partition_and_receipts(tmp_path):
    recipe = {
        "sources": [
            {
                "dataset_id": "brivangl/midjourney-v6-llava",
                "config": "default",
                "split": "train",
                "revision": "fd982f08008591cf739e8a96af2731021a45b4e6",
                "license": "mit",
                "pipeline_variant": "prompt_k_turn",
            }
        ],
        "code_revision": "18a9934d34f7b77b7f6025bfed8a5c691377ca11",
        "models": {
            "generator": {"model_id": "model/generator", "revision": "a" * 40},
            "reflector": {"model_id": "model/reflector", "revision": "b" * 40},
            "verifier": {"model_id": "model/verifier", "revision": "c" * 40},
            "editor": {"model_id": "model/editor", "revision": "d" * 40},
        },
        "prompt_template_hashes": {
            "reflector": "1" * 64,
            "verifier": "2" * 64,
            "editor": "3" * 64,
        },
        "seeds": {"root": 123},
        "converter_config": {"prompt_max_attempts": 2, "prompt_max_edit_turns": 1},
        "partition": make_split_provenance(
            seed="19",
            ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        ),
    }
    manifest_id = derive_manifest_id(**recipe)
    source = adapt_midjourney_prompt_record({"id": "mj-release", "prompt": "A precise visual prompt."})
    source_split = assign_source_splits(
        [source],
        ratios=recipe["partition"]["ratios"],
        seed=recipe["partition"]["seed"],
    )[(source["source_dataset"], source["source_record_id"])]
    base = make_visual_reflection_trajectory(
        tmp_path,
        edit_turns=0,
        source_record_id="image-source",
        manifest_id=manifest_id,
    )
    trajectory = {
        **base,
        "source_dataset": source["source_dataset"],
        "source_record_id": source["source_record_id"],
        "pipeline_variant": source["pipeline_variant"],
        "prompt": source["prompt"],
    }
    trajectory["trajectory_id"] = derive_trajectory_id(
        manifest_id=trajectory["manifest_id"],
        source_dataset=trajectory["source_dataset"],
        source_record_id=trajectory["source_record_id"],
        pipeline_variant=trajectory["pipeline_variant"],
        prompt=trajectory["prompt"],
        images=trajectory["images"],
        steps=trajectory["steps"],
    )

    def make_result(seed):
        request = build_prompt_synthesis_request(
            source,
            manifest_id=manifest_id,
            source_split=source_split,
            partition=recipe["partition"],
            seed=seed,
            max_attempts=2,
            max_edit_turns=1,
        )
        draft_request = make_draft_generation_request(parent_request=request, attempt=0)
        draft_result = make_image_synthesis_result(
            parent_request=request,
            request=draft_request,
            output_image=trajectory["images"][0],
        )
        reflection_request = make_reflection_synthesis_request(
            parent_request=request,
            current_image=trajectory["images"][0],
            turn_index=0,
            attempt=0,
        )
        reflection_result = make_reflection_synthesis_result(
            parent_request=request,
            request=reflection_request,
            response_text=format_reflection_response(trajectory["steps"][0]),
        )
        terminal_request = make_terminal_verification_request(
            parent_request=request,
            current_image=trajectory["images"][0],
            proposed_step=trajectory["steps"][0],
            turn_index=0,
            attempt=0,
        )
        terminal_result = make_verification_result(
            parent_request=request,
            request=terminal_request,
            verdict="accept",
            reason="The prompt is satisfied.",
        )
        return make_synthesis_result(
            request=request,
            status="accepted",
            attempt=0,
            trajectory=trajectory,
            image_synthesis_results=[draft_result],
            reflection_synthesis_results=[reflection_result],
            verification_results=[terminal_result],
        )

    accepted = make_result(derive_source_seed(recipe["seeds"]["root"], source))
    ledger = RejectionLedger(manifest_id)
    manifest = build_data_manifest(
        **recipe,
        accepted_trajectories=[],
        accepted_source_splits=[],
        accepted_synthesis_results=[accepted],
        image_asset_roots={source["source_dataset"]: tmp_path},
        rejection_ledger=ledger,
    )

    assert manifest["counts"]["accepted"] == 1
    assert manifest["counts"]["accepted_by_split"] == {"train": 1, "validation": 0, "test": 0}

    wrong_seed_result = make_result(derive_source_seed(recipe["seeds"]["root"], source) + 1)
    with pytest.raises(VisualReflectionDataError) as error:
        build_data_manifest(
            **recipe,
            accepted_trajectories=[],
            accepted_source_splits=[],
            accepted_synthesis_results=[wrong_seed_result],
            image_asset_roots={source["source_dataset"]: tmp_path},
            rejection_ledger=ledger,
        )
    assert error.value.reason is RejectionReason.INVALID_PROVENANCE

    rejected = make_synthesis_result(
        request=accepted["request"],
        status="rejected",
        attempt=1,
        rejection_reason="turn_limit_exhausted",
    )
    stale_draft_request = make_draft_generation_request(parent_request=accepted["request"], attempt=1)
    stale_draft_result = make_image_synthesis_result(
        parent_request=accepted["request"],
        request=stale_draft_request,
        output_image=trajectory["images"][0],
    )
    with pytest.raises(VisualReflectionDataError):
        make_synthesis_result(
            request=accepted["request"],
            status="rejected",
            attempt=0,
            rejection_reason="verifier_rejected",
            image_synthesis_results=[stale_draft_result],
        )
    rejection_ledger = RejectionLedger(manifest_id)
    rejection_ledger.append_synthesis_result(
        rejected,
        stage="prompt_k_turn",
        message="bounded attempts were exhausted",
    )
    rejected_manifest = build_data_manifest(
        **recipe,
        accepted_trajectories=[],
        accepted_source_splits=[],
        accepted_synthesis_results=[],
        image_asset_roots={},
        rejection_ledger=rejection_ledger,
    )

    assert rejected_manifest["counts"]["rejected_by_reason"] == {"turn_limit_exhausted": 1}
    with pytest.raises(VisualReflectionDataError):
        rejection_ledger.append(
            source_dataset=source["source_dataset"],
            source_record_id="forged-source",
            pipeline_variant="prompt_k_turn",
            stage="prompt_k_turn",
            reason=RejectionReason.TURN_LIMIT_EXHAUSTED,
            message="forged",
            attempt=999,
        )

    source_ledger = RejectionLedger(manifest_id)
    source_ledger.append_source_rejection(
        source_dataset=source["source_dataset"],
        source_record_id="bad-adapter-row",
        pipeline_variant="prompt_k_turn",
        stage="source_adapter",
        reason=RejectionReason.EMPTY_PROMPT,
        message="the public prompt was empty",
    )
    source_rejected_manifest = build_data_manifest(
        **recipe,
        accepted_trajectories=[],
        accepted_source_splits=[],
        accepted_synthesis_results=[],
        image_asset_roots={},
        rejection_ledger=source_ledger,
    )

    assert source_rejected_manifest["counts"]["rejected_by_reason"] == {"empty_prompt": 1}


def test_manifest_id_excludes_result_counts_and_ledger_is_consistent(tmp_path):
    recipe = _recipe()
    manifest_id = derive_manifest_id(**recipe)
    asset_root = tmp_path / "assets"
    first = make_visual_reflection_trajectory(
        asset_root,
        edit_turns=0,
        source_record_id="good-row-1",
        manifest_id=manifest_id,
    )
    second = make_visual_reflection_trajectory(
        asset_root,
        edit_turns=1,
        source_record_id="good-row-2",
        manifest_id=manifest_id,
    )
    assignments = assign_source_splits(
        [first, second],
        ratios=recipe["partition"]["ratios"],
        seed=recipe["partition"]["seed"],
    )
    first_assignment = assignments[(first["source_dataset"], first["source_record_id"])]
    second_assignment = assignments[(second["source_dataset"], second["source_record_id"])]
    ledger = RejectionLedger(manifest_id)
    ledger.append(
        source_dataset="example/visual-reflection",
        source_record_id="bad-row",
        pipeline_variant="direct_parse",
        stage="direct_parse",
        reason=RejectionReason.EMPTY_REFLECTION,
        message="both reflection fields were empty",
        details={"field": "eval_summary[0]"},
    )
    manifest_one = build_data_manifest(
        **recipe,
        accepted_trajectories=[first],
        accepted_source_splits=[first_assignment],
        accepted_synthesis_results=[],
        image_asset_roots={first["source_dataset"]: asset_root},
        rejection_ledger=ledger,
    )
    manifest_many = build_data_manifest(
        **recipe,
        accepted_trajectories=[first, second],
        accepted_source_splits=[first_assignment, second_assignment],
        accepted_synthesis_results=[],
        image_asset_roots={first["source_dataset"]: asset_root},
        rejection_ledger=ledger,
    )

    assert manifest_one["manifest_id"] == manifest_many["manifest_id"] == manifest_id
    expected_splits = {"train": 0, "validation": 0, "test": 0}
    expected_splits[first_assignment["split"]] += 1
    assert manifest_one["counts"] == {
        "accepted": 1,
        "accepted_by_split": expected_splits,
        "rejected": 1,
        "rejected_by_reason": {"empty_reflection": 1},
    }
    assert manifest_many["counts"]["accepted"] == 2
    assert sum(manifest_many["counts"]["accepted_by_split"].values()) == 2
    assert manifest_one["image_hashes"] == [first["images"][0]["sha256"]]
    json.loads(json.dumps(manifest_one))
    ledger_path = tmp_path / "rejections.jsonl"
    ledger.write_jsonl(ledger_path)
    assert json.loads(ledger_path.read_text().strip())["reason"] == "empty_reflection"


@pytest.mark.parametrize("revision", ["main", "dev", "refs/heads/release", "v1.0"])
def test_manifest_rejects_non_commit_source_revision(revision):
    recipe = _recipe()
    recipe["sources"][0]["revision"] = revision

    with pytest.raises(VisualReflectionDataError) as error:
        derive_manifest_id(**recipe)

    assert error.value.reason is RejectionReason.INVALID_PROVENANCE


def test_manifest_rejects_tampered_partition_id():
    recipe = _recipe()
    recipe["partition"]["partition_id"] = f"partition_{'0' * 64}"

    with pytest.raises(VisualReflectionDataError) as error:
        derive_manifest_id(**recipe)

    assert error.value.reason is RejectionReason.INVALID_PROVENANCE


def test_manifest_rejects_empty_source_split():
    recipe = _recipe()
    recipe["sources"][0]["split"] = " "

    with pytest.raises(VisualReflectionDataError) as error:
        derive_manifest_id(**recipe)

    assert error.value.reason is RejectionReason.INVALID_FIELD_TYPE


def test_manifest_requires_matching_source_split_and_rehashes_assets(tmp_path):
    recipe = _recipe()
    manifest_id = derive_manifest_id(**recipe)
    asset_root = tmp_path / "assets"
    trajectory = make_visual_reflection_trajectory(
        asset_root,
        edit_turns=0,
        source_record_id="good-row",
        manifest_id=manifest_id,
    )
    assignment = assign_source_splits(
        [trajectory],
        ratios=recipe["partition"]["ratios"],
        seed=recipe["partition"]["seed"],
    )[(trajectory["source_dataset"], trajectory["source_record_id"])]
    ledger = RejectionLedger(manifest_id)

    with pytest.raises(VisualReflectionDataError) as missing_split_error:
        build_data_manifest(
            **recipe,
            accepted_trajectories=[trajectory],
            accepted_source_splits=[],
            accepted_synthesis_results=[],
            image_asset_roots={trajectory["source_dataset"]: asset_root},
            rejection_ledger=ledger,
        )
    assert missing_split_error.value.reason is RejectionReason.INVALID_PROVENANCE

    wrong_partition = assign_source_splits(
        [trajectory],
        ratios=recipe["partition"]["ratios"],
        seed="different-seed",
    )[(trajectory["source_dataset"], trajectory["source_record_id"])]
    with pytest.raises(VisualReflectionDataError) as partition_error:
        build_data_manifest(
            **recipe,
            accepted_trajectories=[trajectory],
            accepted_source_splits=[wrong_partition],
            accepted_synthesis_results=[],
            image_asset_roots={trajectory["source_dataset"]: asset_root},
            rejection_ledger=ledger,
        )
    assert partition_error.value.reason is RejectionReason.INVALID_SPLIT

    wrong_split = {
        **assignment,
        "split": "test" if assignment["split"] != "test" else "train",
    }
    with pytest.raises(VisualReflectionDataError) as split_error:
        build_data_manifest(
            **recipe,
            accepted_trajectories=[trajectory],
            accepted_source_splits=[wrong_split],
            accepted_synthesis_results=[],
            image_asset_roots={trajectory["source_dataset"]: asset_root},
            rejection_ledger=ledger,
        )
    assert split_error.value.reason is RejectionReason.INVALID_SPLIT

    (asset_root / trajectory["images"][0]["uri"]).unlink()
    with pytest.raises(VisualReflectionDataError) as image_error:
        build_data_manifest(
            **recipe,
            accepted_trajectories=[trajectory],
            accepted_source_splits=[assignment],
            accepted_synthesis_results=[],
            image_asset_roots={trajectory["source_dataset"]: asset_root},
            rejection_ledger=ledger,
        )
    assert image_error.value.reason is RejectionReason.MISSING_IMAGE


def test_manifest_id_binds_the_exact_converter_config():
    default_recipe = _recipe()
    changed_recipe = copy.deepcopy(default_recipe)
    changed_recipe["converter_config"]["adapter_version"] = "v2"

    assert derive_manifest_id(**default_recipe) != derive_manifest_id(**changed_recipe)


def test_rejection_ledger_defensive_copies_nested_details():
    manifest_id = derive_manifest_id(**_recipe())
    ledger = RejectionLedger(manifest_id)
    details = {"nested": {"values": [1]}}
    ledger.append(
        source_dataset="example/visual-reflection",
        source_record_id="bad-row",
        pipeline_variant="direct_parse",
        stage="direct_parse",
        reason=RejectionReason.EMPTY_REFLECTION,
        message="empty",
        details=details,
    )

    details["nested"]["values"].append(2)
    returned = ledger.entries
    returned[0]["details"]["nested"]["values"].append(3)

    assert ledger.entries[0]["details"] == {"nested": {"values": [1]}}


def test_rejection_ledger_append_error_has_direct_parse_lineage():
    ledger = RejectionLedger("manifest-test")
    error = VisualReflectionDataError(
        RejectionReason.EMPTY_REFLECTION,
        "reflection is empty",
        field="eval_summary[0]",
        details={"index": 0},
    )

    entry = ledger.append_error(
        error,
        source_dataset="example/visual-reflection",
        source_record_id="bad-row",
    )

    assert entry["pipeline_variant"] == "direct_parse"
    assert entry["lineage_kind"] == "direct_parse"
    assert entry["stage"] == "direct_parse"
    assert entry["attempt"] == 0
    assert entry["details"] == {"field": "eval_summary[0]", "index": 0}


def test_rejection_ledger_accepts_only_one_final_row_per_source():
    manifest_id = derive_manifest_id(**_recipe())
    ledger = RejectionLedger(manifest_id)
    kwargs = {
        "source_dataset": "example/visual-reflection",
        "source_record_id": "bad-row",
        "pipeline_variant": "direct_parse",
        "stage": "direct_parse",
        "reason": RejectionReason.EMPTY_REFLECTION,
        "message": "empty",
    }
    ledger.append(**kwargs)

    with pytest.raises(VisualReflectionDataError) as error:
        ledger.append(**kwargs)

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


def test_rejection_ledger_serialization_is_independent_of_append_order():
    manifest_id = derive_manifest_id(**_recipe())
    first = RejectionLedger(manifest_id)
    second = RejectionLedger(manifest_id)
    rows = [
        {
            "source_dataset": "example/visual-reflection",
            "source_record_id": record_id,
            "pipeline_variant": "direct_parse",
            "stage": "direct_parse",
            "reason": RejectionReason.EMPTY_REFLECTION,
            "message": "empty",
        }
        for record_id in ("b-row", "a-row")
    ]
    for row in rows:
        first.append(**row)
    for row in reversed(rows):
        second.append(**row)

    assert first.to_jsonl() == second.to_jsonl()


def test_manifest_rejects_synthesis_source_without_models_templates_and_seed():
    recipe = _recipe()
    recipe["sources"] = [
        {
            "dataset_id": "Yejy53/Echo-4o-Image",
            "config": "Instruction-Following-Image",
            "split": "train",
            "revision": "c7341967e86c63beca0953bc5536fd99ee7bd303",
            "license": "mit",
            "pipeline_variant": "pair_0_1_turn",
        }
    ]

    with pytest.raises(VisualReflectionDataError) as error:
        derive_manifest_id(**recipe)

    assert error.value.reason is RejectionReason.INVALID_PROVENANCE
