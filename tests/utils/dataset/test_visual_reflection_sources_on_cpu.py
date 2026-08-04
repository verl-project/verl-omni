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
"""CPU tests for public source adapters, synthesis I/O, dedup, and splitting."""

import pytest

from tests.utils.dataset._visual_reflection_fixtures import write_rgb_png
from verl_omni.utils.dataset.visual_reflection import (
    LocalImageResolver,
    RejectionReason,
    VisualReflectionDataError,
    assign_source_splits,
    deduplicate_source_records,
    make_split_provenance,
)
from verl_omni.utils.dataset.visual_reflection.partition import derive_source_seed
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
    make_terminal_verification_request,
    make_verification_result,
)


class PoisonedExcludedFields(dict):
    """Mapping that raises if a prompt-only adapter touches excluded columns."""

    def __getitem__(self, key):
        if key in {"image", "llava", "llava_status"}:
            raise AssertionError(f"excluded field was accessed: {key}")
        return super().__getitem__(key)


def _echo_record(image_path="reference.png"):
    return {
        "task_type": "t2i",
        "instruction": "A blue ceramic cup.",
        "output_image": image_path,
        "type": "easy",
    }


def _pair_parent_request(tmp_path, *, max_attempts=3):
    write_rgb_png(tmp_path / "reference.png", (0, 255, 0))
    source = adapt_echo_pair_record(
        _echo_record(),
        image_resolver=LocalImageResolver(tmp_path),
    )
    source_split = assign_source_splits(
        [source],
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        seed=1,
    )[(source["source_dataset"], source["source_record_id"])]
    partition = make_split_provenance(
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        seed=1,
    )
    return build_pair_synthesis_request(
        source,
        manifest_id="manifest-1",
        source_split=source_split,
        partition=partition,
        seed=17,
        max_attempts=max_attempts,
    )


def test_echo_adapter_normalizes_real_public_field_names(tmp_path):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))

    source = adapt_echo_pair_record(
        _echo_record(),
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert source["source_record_id"] == "reference.png"
    assert source["prompt"] == "A blue ceramic cup."
    assert source["pipeline_variant"] == "pair_0_1_turn"
    assert source["reference_image"]["uri"] == "reference.png"
    assert len(source["reference_image"]["sha256"]) == 64


def test_echo_adapter_remaps_pinned_logical_image_root(tmp_path):
    write_rgb_png(tmp_path / "images/00000.jpg", (0, 0, 255))
    record = _echo_record("/Echo-4o-Image/Instruction-Following-Image/images/00000.jpg")

    source = adapt_echo_pair_record(record, image_resolver=LocalImageResolver(tmp_path))

    assert source["source_record_id"] == record["output_image"]
    assert source["reference_image"]["uri"] == "images/00000.jpg"


def test_echo_source_override_cannot_relabel_public_image_path(tmp_path):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))

    with pytest.raises(VisualReflectionDataError) as error:
        adapt_echo_pair_record(
            _echo_record(),
            source_record_id="different-id",
            image_resolver=LocalImageResolver(tmp_path),
        )

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


@pytest.mark.parametrize(
    "record",
    [
        {"task_type": "edit", "instruction": "prompt", "output_image": "reference.png"},
        {"task_type": "t2i", "instruction": " ", "output_image": "reference.png"},
        {"task_type": "t2i", "instruction": "prompt"},
    ],
)
def test_echo_adapter_rejects_wrong_route_or_missing_pair_fields(tmp_path, record):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))

    with pytest.raises(VisualReflectionDataError):
        adapt_echo_pair_record(record, image_resolver=LocalImageResolver(tmp_path))


def test_local_resolver_rehashes_canonical_refs_and_blocks_path_escape(tmp_path):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))
    resolver = LocalImageResolver(tmp_path)

    with pytest.raises(VisualReflectionDataError) as hash_error:
        resolver(
            {"uri": "reference.png", "sha256": "0" * 64},
            field="image",
            index=0,
            source_record_id="row",
        )
    with pytest.raises(VisualReflectionDataError) as path_error:
        resolver("../outside.png", field="image", index=0, source_record_id="row")

    assert hash_error.value.reason is RejectionReason.INVALID_IMAGE_HASH
    assert path_error.value.reason is RejectionReason.MISSING_IMAGE


def test_local_resolver_canonicalizes_alias_and_absolute_uris(tmp_path):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))
    resolver = LocalImageResolver(tmp_path)
    canonical = resolver("reference.png", field="image", index=0, source_record_id="row")

    alias = resolver(
        {"uri": "sub/../reference.png", "sha256": canonical["sha256"]},
        field="image",
        index=0,
        source_record_id="row",
    )
    absolute = resolver(
        {"uri": str(tmp_path / "reference.png"), "sha256": canonical["sha256"]},
        field="image",
        index=0,
        source_record_id="row",
    )

    assert alias == absolute == canonical


def test_local_resolver_requires_hf_image_bytes_to_be_materialized(tmp_path):
    resolver = LocalImageResolver(tmp_path)

    with pytest.raises(VisualReflectionDataError) as missing_error:
        resolver(
            {"path": "virtual/image.png", "bytes": b"image-bytes"},
            field="image",
            index=0,
            source_record_id="row",
        )
    (tmp_path / "virtual").mkdir()
    (tmp_path / "virtual/image.png").write_bytes(b"different-bytes")
    with pytest.raises(VisualReflectionDataError) as mismatch_error:
        resolver(
            {"path": "virtual/image.png", "bytes": b"image-bytes"},
            field="image",
            index=0,
            source_record_id="row",
        )

    assert missing_error.value.reason is RejectionReason.MISSING_IMAGE
    assert mismatch_error.value.reason is RejectionReason.INVALID_IMAGE_HASH


def test_midjourney_adapter_reads_only_prompt_and_id():
    record = PoisonedExcludedFields(
        {
            "id": "mj-1",
            "prompt": "A luminous city in rain.",
            "image": object(),
            "llava": object(),
            "llava_status": object(),
        }
    )

    source = adapt_midjourney_prompt_record(record)

    assert source == {
        "source_dataset": "brivangl/midjourney-v6-llava",
        "source_record_id": "mj-1",
        "pipeline_variant": "prompt_k_turn",
        "prompt": "A luminous city in rain.",
        "dedup_key": source["dedup_key"],
    }
    assert "image" not in source
    assert "llava" not in source


def test_midjourney_source_override_cannot_relabel_public_id():
    with pytest.raises(VisualReflectionDataError) as error:
        adapt_midjourney_prompt_record(
            {"id": "public-id", "prompt": "A luminous city in rain."},
            source_record_id="different-id",
        )

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


def test_synthesis_requests_are_stable_and_split_assigned(tmp_path):
    write_rgb_png(tmp_path / "reference.png", (0, 0, 255))
    pair_source = adapt_echo_pair_record(
        _echo_record(),
        image_resolver=LocalImageResolver(tmp_path),
    )
    prompt_source = adapt_midjourney_prompt_record({"id": "mj-1", "prompt": "A city."})
    pair_split = assign_source_splits([pair_source], ratios={"train": 1.0, "validation": 0.0, "test": 0.0}, seed=1)[
        (pair_source["source_dataset"], pair_source["source_record_id"])
    ]
    prompt_split = assign_source_splits([prompt_source], ratios={"train": 0.0, "validation": 1.0, "test": 0.0}, seed=1)[
        (prompt_source["source_dataset"], prompt_source["source_record_id"])
    ]
    pair_partition = make_split_provenance(
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        seed=1,
    )
    prompt_partition = make_split_provenance(
        ratios={"train": 0.0, "validation": 1.0, "test": 0.0},
        seed=1,
    )

    pair_request = build_pair_synthesis_request(
        pair_source,
        manifest_id="manifest-1",
        source_split=pair_split,
        partition=pair_partition,
        seed=17,
        max_attempts=3,
    )
    prompt_request = build_prompt_synthesis_request(
        prompt_source,
        manifest_id="manifest-1",
        source_split=prompt_split,
        partition=prompt_partition,
        seed=19,
        max_attempts=2,
        max_edit_turns=3,
    )

    assert pair_request == build_pair_synthesis_request(
        pair_source,
        manifest_id="manifest-1",
        source_split=pair_split,
        partition=pair_partition,
        seed=17,
        max_attempts=3,
    )
    assert pair_request["split"] == "train"
    assert prompt_request["split"] == "validation"
    assert prompt_request["max_edit_turns"] == 3


def test_reflector_contract_cannot_expose_pair_reference(tmp_path):
    write_rgb_png(tmp_path / "draft.png", (255, 0, 0))
    parent = _pair_parent_request(tmp_path)
    resolver = LocalImageResolver(tmp_path)
    draft = resolver("draft.png", field="draft", index=0, source_record_id="row")
    reference = resolver("reference.png", field="reference", index=0, source_record_id="row")

    reflection_request = make_reflection_synthesis_request(
        parent_request=parent,
        current_image=draft,
        turn_index=0,
        attempt=0,
    )
    reflection_result = make_reflection_synthesis_result(
        parent_request=parent,
        request=reflection_request,
        response_text="Reflection: The draft is acceptable.\nAction: stop\nEdit:",
    )
    verification_request = make_pair_verification_request(
        parent_request=parent,
        draft_image=draft,
        proposed_step={"reflection": "There are two apples.", "action": "continue", "edit": "Remove one."},
        attempt=0,
    )
    terminal_request = make_terminal_verification_request(
        parent_request=parent,
        current_image=draft,
        proposed_step={"reflection": "There is one apple.", "action": "stop", "edit": ""},
        turn_index=0,
        attempt=0,
    )

    assert "reference_image" not in reflection_request
    assert "reference_image" not in terminal_request
    assert reflection_request["parent_request_id"] == parent["request_id"]
    assert reflection_result["response"]["action"] == "stop"
    assert verification_request["reference_image"] == reference
    assert verification_request["proposed_step"]["action"] == "continue"


def test_stage_request_ids_bind_the_exact_proposal(tmp_path):
    write_rgb_png(tmp_path / "draft.png", (255, 0, 0))
    parent = _pair_parent_request(tmp_path)
    image = LocalImageResolver(tmp_path)("draft.png", field="draft", index=0, source_record_id="row")

    first = make_terminal_verification_request(
        parent_request=parent,
        current_image=image,
        proposed_step={"reflection": "Looks good.", "action": "stop", "edit": ""},
        turn_index=0,
        attempt=0,
    )
    second = make_terminal_verification_request(
        parent_request=parent,
        current_image=image,
        proposed_step={"reflection": "Still blurry.", "action": "stop", "edit": ""},
        turn_index=0,
        attempt=0,
    )

    assert first["request_id"] != second["request_id"]


def test_stage_requests_enforce_parent_route_turns_attempts_and_receipts(tmp_path):
    write_rgb_png(tmp_path / "draft.png", (255, 0, 0))
    write_rgb_png(tmp_path / "target.png", (0, 0, 255))
    resolver = LocalImageResolver(tmp_path)
    draft = resolver("draft.png", field="draft", index=0, source_record_id="row")
    target = resolver("target.png", field="target", index=0, source_record_id="row")
    pair_parent = _pair_parent_request(tmp_path, max_attempts=2)

    with pytest.raises(VisualReflectionDataError) as route_error:
        make_edit_synthesis_request(
            parent_request=pair_parent,
            current_image=draft,
            edit="Improve the lighting.",
            turn_index=0,
            attempt=0,
        )
    assert route_error.value.reason is RejectionReason.INVALID_FIELD_TYPE
    with pytest.raises(VisualReflectionDataError) as pair_turn_error:
        make_reflection_synthesis_request(
            parent_request=pair_parent,
            current_image=draft,
            turn_index=1,
            attempt=0,
        )
    assert pair_turn_error.value.reason is RejectionReason.INCOMPATIBLE_EDIT_TARGET

    prompt_source = adapt_midjourney_prompt_record({"id": "mj-stage", "prompt": "A blue ceramic cup."})
    prompt_split = assign_source_splits(
        [prompt_source],
        ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
        seed=1,
    )[(prompt_source["source_dataset"], prompt_source["source_record_id"])]
    prompt_parent = build_prompt_synthesis_request(
        prompt_source,
        manifest_id="manifest-1",
        source_split=prompt_split,
        partition=make_split_provenance(
            ratios={"train": 1.0, "validation": 0.0, "test": 0.0},
            seed=1,
        ),
        seed=19,
        max_attempts=2,
        max_edit_turns=1,
    )
    draft_request = make_draft_generation_request(parent_request=prompt_parent, attempt=0)
    draft_result = make_image_synthesis_result(
        parent_request=prompt_parent,
        request=draft_request,
        output_image=draft,
    )
    edit_request = make_edit_synthesis_request(
        parent_request=prompt_parent,
        current_image=draft,
        edit="Improve the lighting.",
        turn_index=0,
        attempt=0,
    )
    verification_request = make_edit_verification_request(
        parent_request=prompt_parent,
        current_image=draft,
        target_image=target,
        edit=edit_request["edit"],
        turn_index=0,
        attempt=0,
    )
    result = make_verification_result(
        parent_request=prompt_parent,
        request=verification_request,
        verdict="accept",
        reason="The edit is faithful and improves the image.",
    )

    assert result["request_id"] == verification_request["request_id"]
    assert draft_request["seed"] == prompt_parent["seed"]
    assert draft_result["output_image"] == draft
    assert (
        edit_request["seed"]
        == make_edit_synthesis_request(
            parent_request=prompt_parent,
            current_image=draft,
            edit="Improve the lighting.",
            turn_index=0,
            attempt=0,
        )["seed"]
    )
    assert verification_request["parent_request_id"] == prompt_parent["request_id"]

    tampered = {**verification_request, "edit": "Replace the entire scene."}
    with pytest.raises(VisualReflectionDataError):
        make_verification_result(
            parent_request=prompt_parent,
            request=tampered,
            verdict="accept",
            reason="stale result",
        )
    with pytest.raises(VisualReflectionDataError) as turn_error:
        make_edit_synthesis_request(
            parent_request=prompt_parent,
            current_image=draft,
            edit="Another edit.",
            turn_index=1,
            attempt=0,
        )
    assert turn_error.value.reason is RejectionReason.TURN_LIMIT_EXHAUSTED
    with pytest.raises(VisualReflectionDataError) as attempt_error:
        make_terminal_verification_request(
            parent_request=prompt_parent,
            current_image=target,
            proposed_step={"reflection": "Looks good.", "action": "stop", "edit": ""},
            turn_index=1,
            attempt=2,
        )
    assert attempt_error.value.reason is RejectionReason.TURN_LIMIT_EXHAUSTED


def test_exact_dedup_survivor_and_splits_do_not_depend_on_input_order(tmp_path):
    base = adapt_midjourney_prompt_record({"id": "z-row", "prompt": "A blue ceramic cup."})
    duplicate = adapt_midjourney_prompt_record({"id": "a-row", "prompt": "A blue ceramic cup."})

    retained_a, duplicates_a = deduplicate_source_records([base, duplicate])
    retained_b, duplicates_b = deduplicate_source_records([duplicate, base])
    splits = assign_source_splits([base, duplicate], ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7)

    assert retained_a == retained_b == [duplicate]
    assert duplicates_a == duplicates_b
    assert duplicates_a[0]["source_record_id"] == "z-row"
    assert splits[(base["source_dataset"], "z-row")]["split"] == splits[(base["source_dataset"], "a-row")]["split"]


def test_same_prompt_with_different_reference_is_not_deduplicated(tmp_path):
    write_rgb_png(tmp_path / "a.png", (1, 2, 3))
    write_rgb_png(tmp_path / "b.png", (4, 5, 6))
    resolver = LocalImageResolver(tmp_path)
    source_a = adapt_echo_pair_record(_echo_record("a.png"), image_resolver=resolver)
    source_b = adapt_echo_pair_record(_echo_record("b.png"), image_resolver=resolver)

    retained, duplicates = deduplicate_source_records([source_a, source_b])

    assert len(retained) == 2
    assert duplicates == []


def test_conflicting_duplicate_source_identity_fails_closed():
    first = adapt_midjourney_prompt_record({"id": "same", "prompt": "A cat."})
    second = adapt_midjourney_prompt_record({"id": "same", "prompt": "A dog."})

    with pytest.raises(VisualReflectionDataError) as error:
        deduplicate_source_records([first, second])

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


def test_tampered_dedup_key_cannot_force_unrelated_sources_to_collapse():
    cat = adapt_midjourney_prompt_record({"id": "cat", "prompt": "A cat."})
    dog = adapt_midjourney_prompt_record({"id": "dog", "prompt": "A dog."})
    dog["dedup_key"] = cat["dedup_key"]

    with pytest.raises(VisualReflectionDataError) as error:
        deduplicate_source_records([cat, dog])

    assert error.value.reason is RejectionReason.DUPLICATE_CONTENT


def test_split_ratios_and_source_seed_are_validated():
    source = adapt_midjourney_prompt_record({"id": "mj-1", "prompt": "A city."})

    assert derive_source_seed(123, source) == derive_source_seed(123, source)
    with pytest.raises(VisualReflectionDataError) as error:
        assign_source_splits([source], ratios={"train": 0.9, "validation": 0.1, "test": 0.1}, seed=1)

    assert error.value.reason is RejectionReason.INVALID_SPLIT
