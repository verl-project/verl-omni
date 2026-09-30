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
"""CPU tests for Boogu-Image LoRA rollout name mapping.

Regression cover for https://github.com/verl-project/verl-omni/issues/658. The
actor's diffusers module tree and the vllm-omni Boogu transformer disagree in
two places, and either one silently drops deltas on the rollout:

- the attention output projection (``attn.to_out.0`` vs ``attn.to_out``), and
- the joint attention's per-stream projections, which the actor owns under
  ``img_instruct_attn.processor.*`` and the transformer exposes directly.

vllm-omni only warns when *nothing* binds, so a partial miss is invisible:
``TestBindingCompleteness`` is what catches it.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    BOOGU_LORA_TARGETS,
    lora_engine_module_names,
    lora_module_name,
    rename_boogu_lora_name,
    unbindable_boogu_lora_module_names,
    unwrappable_boogu_lora_module_names,
    validate_boogu_lora_targets,
)

RANK = 4
HIDDEN = 8

# Both shipped Boogu recipes:
#   examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh
#   examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh
RECIPE_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "img_to_q",
    "img_to_k",
    "img_to_v",
    "img_out",
    "instruct_to_q",
    "instruct_to_k",
    "instruct_to_v",
    "instruct_out",
    "feed_forward.linear_1",
    "feed_forward.linear_2",
    "feed_forward.linear_3",
    "img_feed_forward.linear_1",
    "img_feed_forward.linear_2",
    "img_feed_forward.linear_3",
]

#: Projections ``BooguImageJointAttention`` exposes as direct children, mirroring
#: ``boogu_image_transformer.py``. The trainer keeps the same eight on a custom
#: attention processor, which is the mismatch the rename table has to bridge.
JOINT_ATTENTION_ENGINE_LEAVES = (
    "img_to_q",
    "img_to_k",
    "img_to_v",
    "instruct_to_q",
    "instruct_to_k",
    "instruct_to_v",
    "img_out",
    "instruct_out",
)
_SELF_ATTENTION_ENGINE_LEAVES = ("to_q", "to_k", "to_v", "to_out")
_FEED_FORWARD_ENGINE_LEAVES = ("linear_1", "linear_2", "linear_3")


def _engine_leaf_modules(leaves) -> nn.Module:
    module = nn.Module()
    for leaf in leaves:
        setattr(module, leaf, nn.Linear(HIDDEN, HIDDEN, bias=False))
    return module


def _engine_base_block() -> nn.Module:
    block = nn.Module()
    block.attn = _engine_leaf_modules(_SELF_ATTENTION_ENGINE_LEAVES)
    block.feed_forward = _engine_leaf_modules(_FEED_FORWARD_ENGINE_LEAVES)
    return block


def _engine_double_stream_block() -> nn.Module:
    block = nn.Module()
    block.img_self_attn = _engine_leaf_modules(_SELF_ATTENTION_ENGINE_LEAVES)
    block.img_instruct_attn = _engine_leaf_modules((*JOINT_ATTENTION_ENGINE_LEAVES, "to_out"))
    block.img_feed_forward = _engine_leaf_modules(_FEED_FORWARD_ENGINE_LEAVES)
    return block


def _engine_transformer(blocks_per_group: int = 2) -> nn.Module:
    """A stand-in whose module *names* mirror the vllm-omni Boogu transformer."""
    transformer = nn.Module()
    for group in (
        "double_stream_layers",
        "single_stream_layers",
        "context_refiner",
        "noise_refiner",
        "ref_image_refiner",
    ):
        builder = _engine_double_stream_block if group == "double_stream_layers" else _engine_base_block
        setattr(transformer, group, nn.ModuleList([builder() for _ in range(blocks_per_group)]))
    return transformer


class _StubPipeline:
    """Carries the component attribute ``lora_engine_module_names`` scans for."""

    def __init__(self, transformer: nn.Module):
        self.transformer = transformer


def _mapper(pipeline_cls=None, transformer: nn.Module | None = None):
    """Bind a real Boogu mapper onto a stand-in pipeline holding an engine tree.

    ``map_lora_update_to_engine`` reads instance state (the engine tree it validates
    the pushed keys against), so the tests must supply one: a bare ``__new__``
    instance would silently skip the binding check and let a regression through.
    """
    pytest.importorskip("vllm_omni")
    if pipeline_cls is None:
        from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import (
            BooguImagePipelineWithLogProb as pipeline_cls,
        )

    stub = _StubPipeline(transformer if transformer is not None else _engine_transformer())
    stub.map_lora_update_to_engine = pipeline_cls.map_lora_update_to_engine.__get__(stub)
    return stub


# Block-local module names as the actor holds them, taken from the FSDP shard keys
# of this recipe's own checkpoint (e.g.
# ``double_stream_layers.0.img_instruct_attn.to_out.0.lora_A.default.weight``).
#
# The joint attention is the one block whose projections are *not* direct
# children: ``BooguImageTransformerBlock`` hands them to a custom processor and
# deletes ``Attention``'s own ``to_q``/``to_k``/``to_v``, so the actor names them
# ``img_instruct_attn.processor.*`` while the vllm-omni
# ``BooguImageJointAttention`` holds all nine directly. Spelling these names
# without the ``.processor.`` step is what let the original o-proj fix ship
# "complete" while eight of the nine projections still bound nothing.
JOINT_ATTENTION_PROCESSOR_MODULES = (
    "img_instruct_attn.processor.img_to_q",
    "img_instruct_attn.processor.img_to_k",
    "img_instruct_attn.processor.img_to_v",
    "img_instruct_attn.processor.instruct_to_q",
    "img_instruct_attn.processor.instruct_to_k",
    "img_instruct_attn.processor.instruct_to_v",
    "img_instruct_attn.processor.img_out",
    "img_instruct_attn.processor.instruct_out",
)
DOUBLE_STREAM_MODULES = (
    "img_instruct_attn.to_out.0",
    *JOINT_ATTENTION_PROCESSOR_MODULES,
    "img_self_attn.to_q",
    "img_feed_forward.linear_3",
)
BASE_BLOCK_MODULES = ("attn.to_q", "attn.to_out.0", "feed_forward.linear_1")


def _trainer_lora_tensors(
    block: str = "double_stream_layers.0", modules=DOUBLE_STREAM_MODULES
) -> dict[str, torch.Tensor]:
    """Build a PEFT-style Boogu LoRA state dict as the actor exports it.

    Keys carry the ``transformer.`` component prefix, because the actor's weight
    sync pushes ``f"transformer.{name}"`` and the engine keys its wrappable layers
    the same way.
    """
    tensors = {}
    for module in modules:
        tensors[f"transformer.{block}.{module}.lora_A.weight"] = torch.randn(RANK, HIDDEN)
        tensors[f"transformer.{block}.{module}.lora_B.weight"] = torch.randn(HIDDEN, RANK)
    return tensors


def _peft_config() -> dict:
    return {"r": RANK, "lora_alpha": 8, "target_modules": list(RECIPE_TARGETS)}


class TestNameTranslation:
    @pytest.mark.parametrize(
        ("diffusers_name", "vllm_name"),
        [
            ("attn.to_out.0", "attn.to_out"),
            ("img_instruct_attn.to_out.0", "img_instruct_attn.to_out"),
        ],
    )
    def test_output_projection_is_unwrapped(self, diffusers_name, vllm_name):
        assert rename_boogu_lora_name(diffusers_name) == vllm_name

    @pytest.mark.parametrize("module", JOINT_ATTENTION_PROCESSOR_MODULES)
    def test_joint_attention_processor_prefix_is_dropped(self, module):
        """``.processor.`` is the second mismatch; without it eight deltas vanish.

        The joint attention's projections are owned by a custom processor on the
        trainer side and are direct children on the vllm-omni side, so the infix
        has to come out of both the tensor keys and nothing else.
        """
        trainer_name = f"double_stream_layers.0.{module}"
        vllm_name = f"double_stream_layers.0.{module.replace('.processor.', '.')}"
        assert rename_boogu_lora_name(trainer_name) == vllm_name

    def test_processor_rename_keeps_the_lora_weight_suffix(self):
        trainer_name = "double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.default.weight"
        assert rename_boogu_lora_name(trainer_name) == (
            "double_stream_layers.0.img_instruct_attn.img_to_q.lora_A.default.weight"
        )

    def test_verbatim_targets_are_untouched(self):
        # Everything except the o-proj already matches, so the mapper must not
        # disturb it -- a broad rewrite is how the q/k/v half got broken before.
        for target in RECIPE_TARGETS:
            if target != "to_out.0":
                assert rename_boogu_lora_name(target) == target

    def test_lora_tensor_keys_are_renamed(self):
        mapped, _ = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        modules = {lora_module_name(name) for name in mapped}
        assert "transformer.double_stream_layers.0.img_instruct_attn.to_out" in modules
        assert "transformer.double_stream_layers.0.img_instruct_attn.img_to_q" in modules
        assert "transformer.double_stream_layers.0.img_self_attn.to_q" in modules
        assert not any(".to_out.0" in module for module in modules)
        assert not any(".processor." in module for module in modules)

    def test_base_block_output_projection_is_renamed_too(self):
        mapped, _ = _mapper().map_lora_update_to_engine(
            _trainer_lora_tensors("context_refiner.1", BASE_BLOCK_MODULES), _peft_config()
        )
        modules = {name.rsplit(".lora_", 1)[0] for name in mapped}
        assert "transformer.context_refiner.1.attn.to_out" in modules
        assert "transformer.context_refiner.1.attn.to_q" in modules

    def test_component_prefix_is_preserved(self):
        # Pushed keys may or may not carry a component prefix; the rename is a leaf
        # change and must not depend on it.
        tensors = {"transformer.context_refiner.0.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN)}
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["transformer.context_refiner.0.attn.to_out.lora_A.weight"]

    def test_peft_wrapper_prefix_is_stripped(self):
        # The actor exports PEFT keys, whose names carry the PEFT wrapper
        # (``fsdp_utils.py`` builds them as ``base_model.model.<module>``). Upstream
        # #661 strips it in its per-pipeline mappers; left in place every delta would
        # fail to resolve against the engine tree and trip the binding guard.
        tensors = {
            "transformer.base_model.model.context_refiner.0.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN)
        }
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["transformer.context_refiner.0.attn.to_out.lora_A.weight"]

    def test_non_lora_names_pass_through(self):
        tensors = {"some_unrelated.weight": torch.randn(2, 2)}
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert list(mapped) == ["some_unrelated.weight"]

    def test_colliding_renames_are_rejected(self):
        # A half-renamed state dict would make one delta overwrite the other.
        tensors = {
            "context_refiner.0.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN),
            "context_refiner.0.attn.to_out.lora_A.weight": torch.randn(RANK, HIDDEN),
        }
        with pytest.raises(ValueError, match="collapsed distinct tensors"):
            _mapper().map_lora_update_to_engine(tensors, _peft_config())


class TestTargetValidation:
    def test_recipe_targets_are_accepted_and_translated(self):
        translated = validate_boogu_lora_targets(RECIPE_TARGETS)
        assert len(translated) == len(RECIPE_TARGETS)
        assert "to_out" in translated
        assert "to_out.0" not in translated

    def test_mapper_rewrites_target_modules_in_config(self):
        _, config = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        assert "to_out" in config["target_modules"]
        assert "to_out.0" not in config["target_modules"]
        assert config["r"] == RANK  # other fields untouched

    def test_input_config_is_not_mutated(self):
        config = _peft_config()
        _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), config)
        assert "to_out.0" in config["target_modules"]

    @pytest.mark.parametrize("target_modules", ["all-linear", ["to_q", "adaln_proj.linear"], ["to_q", "norm_q"]])
    def test_unbindable_targets_raise_instead_of_being_dropped(self, target_modules):
        config = {**_peft_config(), "target_modules": target_modules}
        with pytest.raises(ValueError, match="unsupported targets"):
            _mapper().map_lora_update_to_engine({}, config)

    def test_empty_targets_are_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_boogu_lora_targets([])

    def test_missing_target_modules_are_rejected(self):
        with pytest.raises(ValueError, match="explicit target_modules"):
            validate_boogu_lora_targets(None)


class TestVllmManagerInterplay:
    """Guard against the original failure: the o-proj target matched no vllm module."""

    # Leaf modules as they appear on the vllm-omni Boogu transformer
    # (boogu_image_transformer.py: BooguImageSelfAttention, BooguImageJointAttention,
    # LuminaFeedForward).
    VLLM_MODULES = [
        "noise_refiner.0.attn.to_q",
        "noise_refiner.0.attn.to_k",
        "noise_refiner.0.attn.to_v",
        "noise_refiner.0.attn.to_out",
        "noise_refiner.0.feed_forward.linear_1",
        "noise_refiner.0.feed_forward.linear_2",
        "noise_refiner.0.feed_forward.linear_3",
        "double_stream_layers.0.img_instruct_attn.img_to_q",
        "double_stream_layers.0.img_instruct_attn.img_to_k",
        "double_stream_layers.0.img_instruct_attn.img_to_v",
        "double_stream_layers.0.img_instruct_attn.instruct_to_q",
        "double_stream_layers.0.img_instruct_attn.instruct_to_k",
        "double_stream_layers.0.img_instruct_attn.instruct_to_v",
        "double_stream_layers.0.img_instruct_attn.instruct_out",
        "double_stream_layers.0.img_instruct_attn.img_out",
        "double_stream_layers.0.img_instruct_attn.to_out",
        "double_stream_layers.0.img_self_attn.to_out",
        "double_stream_layers.0.img_feed_forward.linear_1",
        "double_stream_layers.0.img_feed_forward.linear_2",
        "double_stream_layers.0.img_feed_forward.linear_3",
        "single_stream_layers.0.attn.to_q",
        "single_stream_layers.0.attn.to_out",
        "single_stream_layers.0.feed_forward.linear_1",
    ]

    @pytest.fixture()
    def match(self):
        pytest.importorskip("vllm_omni")
        from vllm_omni.diffusion.lora.utils import _match_target_modules

        return _match_target_modules

    def test_untranslated_target_matches_nothing_on_vllm(self, match):
        """This is the bug: ``to_out.0`` binds no layer on the Boogu transformer."""
        assert not any(match(module, ["to_out.0"]) for module in self.VLLM_MODULES)

    def test_translated_target_matches_the_o_proj_modules(self, match):
        matched = {module for module in self.VLLM_MODULES if match(module, ["to_out"])}
        assert matched == {
            "noise_refiner.0.attn.to_out",
            "double_stream_layers.0.img_instruct_attn.to_out",
            "double_stream_layers.0.img_self_attn.to_out",
            "single_stream_layers.0.attn.to_out",
        }

    def test_every_recipe_target_binds_at_least_one_module(self, match):
        """No target in the shipped recipes may be silently unbindable."""
        for target in validate_boogu_lora_targets(RECIPE_TARGETS):
            assert any(match(module, [target]) for module in self.VLLM_MODULES), (
                f"target {target!r} matches no Boogu transformer module; "
                "it would be shipped and silently dropped by the rollout"
            )

    def test_pushed_key_matches_a_binding_candidate(self):
        """Matching alone is not enough: the manager must also *bind* the tensor.

        Mirrors ``DiffusionLoRAManager._get_lora_weights``, which tries the full name,
        the component-relative name and the bare suffix. With the original
        ``...to_out.0`` key none of those candidates matched, so even a wrapped layer
        would have stayed unbound.
        """
        mapped, _ = _mapper().map_lora_update_to_engine(
            {"context_refiner.1.attn.to_out.0.lora_A.weight": torch.randn(RANK, HIDDEN)}, _peft_config()
        )
        (key,) = mapped
        module = "context_refiner.1.attn.to_out"
        candidates = {f"transformer.{module}", module, module.split(".")[-1]}
        assert key.removesuffix(".lora_A.weight") in candidates

    def test_whitelist_covers_what_the_recipes_request(self):
        assert {rename_boogu_lora_name(target) for target in RECIPE_TARGETS} <= BOOGU_LORA_TARGETS


class TestBindingCompleteness:
    """Every delta the actor exports must be bindable, or it vanishes silently.

    ``TestVllmManagerInterplay`` checks *targets*; this checks the *keys* the
    mapper actually pushes. The two are not equivalent, which is exactly how the
    o-proj fix shipped "complete" while eight joint-attention projections still
    bound nothing: ``img_to_q`` is a bindable target, but the actor exports it as
    ``...img_instruct_attn.processor.img_to_q``, which binds no engine module.

    The engine names come from ``lora_engine_module_names`` on a stand-in tree, so
    these tests pin the same component-qualified key space the manager uses --
    building them from ``pipeline.transformer.named_modules()`` would drop the
    ``transformer.`` prefix and report every correct delta as unbindable.
    """

    @staticmethod
    def _unbindable(mapped: dict[str, torch.Tensor]) -> list[str]:
        return unbindable_boogu_lora_module_names(
            (lora_module_name(name) for name in mapped),
            lora_engine_module_names(_StubPipeline(_engine_transformer())),
        )

    @staticmethod
    def _unwrappable(mapped: dict[str, torch.Tensor], config: dict) -> list[str]:
        """Uses the mapper's own translated `target_modules`, as the manager would."""
        return unwrappable_boogu_lora_module_names(
            (lora_module_name(name) for name in mapped), config["target_modules"]
        )

    def test_every_pushed_delta_is_bindable(self):
        mapped, _ = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        assert self._unbindable(mapped) == []

    def test_every_pushed_delta_is_wrappable(self):
        """Bindable is not enough: the manager must also have wrapped the module.

        `_replace_layers_with_lora` registers LoRA layers only for modules matching
        `target_modules`, so a correctly-renamed key outside that set is dropped
        just as silently as a mis-named one. The `to_out.0` -> `to_out` translation
        matters here too: an untranslated target list matches no engine module and
        would leave every o-proj wrapper unbuilt.
        """
        for block, modules in (
            ("double_stream_layers.0", DOUBLE_STREAM_MODULES),
            ("context_refiner.1", BASE_BLOCK_MODULES),
        ):
            mapped, config = _mapper().map_lora_update_to_engine(_trainer_lora_tensors(block, modules), _peft_config())
            assert self._unwrappable(mapped, config) == []

    def test_base_block_deltas_are_bindable(self):
        mapped, _ = _mapper().map_lora_update_to_engine(
            _trainer_lora_tensors("context_refiner.1", BASE_BLOCK_MODULES), _peft_config()
        )
        assert self._unbindable(mapped) == []

    def test_untranslated_processor_keys_are_reported(self):
        """The pre-fix spelling must be caught, not merely tolerated."""
        stale = [f"transformer.double_stream_layers.0.{module}" for module in JOINT_ATTENTION_PROCESSOR_MODULES]
        assert unbindable_boogu_lora_module_names(
            stale, lora_engine_module_names(_StubPipeline(_engine_transformer()))
        ) == sorted(stale)

    def test_lora_weight_suffix_is_stripped(self):
        assert lora_module_name("a.b.to_q.lora_A.default.weight") == "a.b.to_q"
        assert lora_module_name("a.b.to_q.lora_B.weight") == "a.b.to_q"
        assert lora_module_name("a.b.to_q.lora_A.weight") == "a.b.to_q"
        assert lora_module_name("not_a_lora.weight") == "not_a_lora.weight"


class TestEngineKeySpace:
    """The guard is only as good as the key space it validates against."""

    def test_engine_names_are_component_qualified(self):
        names = lora_engine_module_names(_StubPipeline(_engine_transformer()))
        assert "transformer.double_stream_layers.0.img_instruct_attn.img_to_q" in names
        assert "transformer.context_refiner.1.attn.to_out" in names

    def test_prefix_less_engine_names_would_report_correct_deltas(self):
        """Regression: an unprefixed basis reports every correct delta as unbindable.

        ``pipeline.transformer.named_modules()`` yields ``double_stream_layers.0...``
        while the pushed keys carry ``transformer.``, so using it as the guard's
        basis aborts the LoRA sync on a perfectly translated state dict.
        """
        transformer = _engine_transformer()
        mapped, _ = _mapper(transformer=transformer).map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        pushed = [lora_module_name(name) for name in mapped]
        prefix_less = [name for name, _ in transformer.named_modules() if name]
        assert prefix_less, "the stand-in tree must be non-empty for this regression to mean anything"
        assert unbindable_boogu_lora_module_names(pushed, prefix_less) != []
        assert unbindable_boogu_lora_module_names(pushed, lora_engine_module_names(_StubPipeline(transformer))) == []

    def test_guard_rejects_deltas_the_rename_table_cannot_place(self):
        tensors = {
            "transformer.double_stream_layers.0.img_instruct_attn.bogus_proj.lora_A.weight": torch.randn(RANK, HIDDEN)
        }
        with pytest.raises(ValueError, match="bind no vllm-omni module"):
            _mapper().map_lora_update_to_engine(tensors, _peft_config())

    def test_guard_catches_the_processor_rule_being_reverted(self, monkeypatch):
        """The original defect, replayed through the mapper rather than the helper."""
        from verl_omni.pipelines.boogu_image_flow_grpo import common

        monkeypatch.setattr(common, "_BOOGU_LORA_NAME_RENAMES", (("to_out.0", "to_out"),))
        with pytest.raises(ValueError, match="bind no vllm-omni module"):
            _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())

    def test_guard_ignores_non_lora_keys(self):
        tensors = {"some_unrelated.weight": torch.randn(2, 2), **_trainer_lora_tensors()}
        mapped, _ = _mapper().map_lora_update_to_engine(tensors, _peft_config())
        assert "some_unrelated.weight" in mapped

    def test_guard_refuses_a_pipeline_with_no_engine_tree(self):
        """A pipeline the guard cannot inspect must fail loudly, not pass vacuously."""
        with pytest.raises(ValueError, match="bind no vllm-omni module"):
            _mapper(transformer=nn.Module()).map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())

    def test_guard_rejects_keys_outside_the_target_list(self):
        """A correctly-named delta outside `target_modules` is dropped just as quietly.

        The manager registers LoRA layers only for modules matching
        `target_modules`, and `_get_lora_weights` searches nothing else, so a key
        that names a real engine module still binds nothing if the recipe never
        asked for it. `img_self_attn.to_q` is that case here.
        """
        config = _peft_config()
        config["target_modules"] = ["img_to_q"]
        with pytest.raises(ValueError, match="would not wrap"):
            _mapper().map_lora_update_to_engine(_trainer_lora_tensors(), config)

    def test_guard_accepts_two_component_targets(self):
        """`feed_forward.linear_1` is a bindable target, so it must not be refused."""
        config = _peft_config()
        config["target_modules"] = ["feed_forward.linear_1", "img_feed_forward.linear_1"]
        mapped, _ = _mapper().map_lora_update_to_engine(
            _trainer_lora_tensors("context_refiner.1", ("feed_forward.linear_1",)), config
        )
        assert "transformer.context_refiner.1.feed_forward.linear_1.lora_A.weight" in mapped


class TestRegisteredPipelinesExposeTheMapper:
    """The hijack only calls the mapper when the *registered* pipeline class defines it.

    ``VLLMOmniHijack`` does ``getattr(self.pipeline, "map_lora_update_to_engine", None)``
    and skips translation when it is absent, so a mapper that never reaches the
    registered class is indistinguishable from having no mapper at all. Boogu
    registers one rollout class per algorithm, and the DiffusionNFT class derives
    from the FlowGRPO one, so both must expose it or one recipe silently regresses.
    """

    ALGORITHMS = ("flow_grpo", "diffusion_nft")

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_registered_pipeline_defines_the_mapper(self, algorithm):
        pytest.importorskip("vllm_omni")
        import verl_omni.pipelines  # noqa: F401  (populates the registry)
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        pipeline_cls = VllmOmniPipelineBase.get_class("BooguImagePipeline", algorithm)
        assert pipeline_cls is not None, f"no Boogu rollout pipeline registered for {algorithm!r}"
        assert callable(getattr(pipeline_cls, "map_lora_update_to_engine", None)), (
            f"{pipeline_cls.__name__} does not expose map_lora_update_to_engine, so the "
            "rollout would silently drop every to_out.0 delta"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_registered_pipeline_mapper_translates_the_recipe(self, algorithm):
        pytest.importorskip("vllm_omni")
        import verl_omni.pipelines  # noqa: F401
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        pipeline_cls = VllmOmniPipelineBase.get_class("BooguImagePipeline", algorithm)
        # The mapper validates against the engine tree it is bound to, so bind the
        # registered class's own method onto a stand-in carrying a real one.
        mapped, config = _mapper(pipeline_cls).map_lora_update_to_engine(_trainer_lora_tensors(), _peft_config())
        assert "transformer.double_stream_layers.0.img_instruct_attn.to_out.lora_A.weight" in mapped
        assert "to_out" in config["target_modules"]
        assert "to_out.0" not in config["target_modules"]


@pytest.fixture
def boogu_manager(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer as boogu

    from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    # Match CPU-only CI even when the local host supports pinned-memory copies.
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(boogu, "Attention", lambda **_: torch.nn.Identity())
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        transformer = boogu.BooguImageTransformer2DModel(
            OmniDiffusionConfig(
                tf_model_config=TransformerConfig.from_dict(
                    {
                        "hidden_size": 32,
                        "num_layers": 2,
                        "num_double_stream_layers": 1,
                        "num_refiner_layers": 1,
                        "num_attention_heads": 1,
                        "num_kv_heads": 1,
                        "multiple_of": 32,
                        "axes_dim_rope": (8, 12, 12),
                        "axes_lens": (16, 16, 16),
                        "instruction_feature_configs": {
                            "instruction_feat_dim": 32,
                            "reduce_type": "mean",
                            "num_instruction_feature_layers": 1,
                        },
                    }
                )
            )
        )
        assert isinstance(transformer.single_stream_layers[0].attn.to_out, boogu.RowParallelLinear)
        assert isinstance(transformer.double_stream_layers[0].img_instruct_attn.to_out, boogu.ReplicatedLinear)
        pipeline = object.__new__(BooguImagePipelineWithLogProb)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        params = {}
        matched_targets = set()
        # Use real runtime sizes with actor names documented by Boogu's load_weights.
        # This models exported tensors, not a real Boogu actor/checkpoint export.
        for name, module in transformer.named_modules():
            actor_name = name
            if name.rsplit(".", 1)[-1] in JOINT_ATTENTION_ENGINE_LEAVES:
                actor_name = name.replace(".img_instruct_attn.", ".img_instruct_attn.processor.")
            if actor_name.endswith(".to_out"):
                actor_name += ".0"
            matches = {target for target in RECIPE_TARGETS if actor_name.endswith(f".{target}")}
            if not matches:
                continue
            matched_targets.update(matches)
            out_features, in_features = module.weight.shape
            prefix = f"transformer.{actor_name}"
            params[f"{prefix}.lora_A.weight"] = torch.full((4, in_features), 0.125)
            params[f"{prefix}.lora_B.weight"] = torch.full((out_features, 4), 0.25)
        assert matched_targets == set(RECIPE_TARGETS)
        assert len(params) == 88  # 44 module paths across all blocks and refiners.
        assert sum(".processor." in name for name in params) == 16
        config = {"r": 4, "lora_alpha": 8, "target_modules": RECIPE_TARGETS}
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        yield manager, params, config


@pytest.mark.parametrize("case", ["unmapped", "output_rename_only", "full", "zero_init", "output_only"])
def test_boogu_load_bind_activate_contract(boogu_manager, monkeypatch, case):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = boogu_manager
    if case == "unmapped":
        monkeypatch.setattr(manager.pipeline, "map_lora_update_to_engine", lambda tensors, config: (tensors, config))
    elif case == "output_rename_only":
        # Reproduce the old mapper: output projections bind, processor projections do not.
        monkeypatch.setattr(
            manager.pipeline,
            "map_lora_update_to_engine",
            lambda tensors, config: (
                {name.replace(".to_out.0.", ".to_out."): tensor for name, tensor in tensors.items()},
                {**config, "target_modules": ["to_out" if t == "to_out.0" else t for t in config["target_modules"]]},
            ),
        )
    elif case == "zero_init":
        params = {name: torch.zeros_like(tensor) if ".lora_B." in name else tensor for name, tensor in params.items()}
    elif case == "output_only":
        params = {name: tensor for name, tensor in params.items() if ".to_out.0." in name}
        config = {**config, "target_modules": ["to_out.0"]}
    # The loader may scale B in place when CPU tensors share storage.
    mapped, _ = manager.pipeline.map_lora_update_to_engine(
        {name: tensor.clone() for name, tensor in params.items()}, config
    )
    manager.set_active_adapter(
        OmniTensorLoRARequest(
            lora_name="boogu", lora_int_id=1, lora_path="in-memory", lora_tensors=params, peft_config=config
        )
    )
    assert manager._active_adapter_id == 1
    loaded = manager._registered_adapters[1]
    bound = {name for name in manager._lora_modules if manager._get_lora_weights(loaded, name) is not None}
    if case == "unmapped":
        assert len(bound) == 30  # Six output and eight processor projections stay unbound.
        assert any(name.endswith(".to_out.0") for name in loaded.loras)
        return
    if case == "output_rename_only":
        assert len(manager._lora_modules) == 44 and len(bound) == 36
        missing = set(manager._lora_modules) - bound
        assert missing == {
            f"transformer.double_stream_layers.0.img_instruct_attn.{t}" for t in JOINT_ATTENTION_ENGINE_LEAVES
        }
        for name in missing:
            layer = manager._lora_modules[name]
            assert all(torch.count_nonzero(t) == 0 for t in (*layer.lora_a_stacked, *layer.lora_b_stacked))
        return

    assert len(bound) == len(manager._lora_modules) == len(params) // 2
    assert any(name.endswith(".to_out") for name in bound)
    for name, layer in manager._lora_modules.items():
        expected_a = mapped[f"{name}.lora_A.weight"]
        expected_b = mapped[f"{name}.lora_B.weight"] * 2
        torch.testing.assert_close(layer.lora_a_stacked[0][0, 0, :4], expected_a, rtol=0, atol=0)
        torch.testing.assert_close(layer.lora_b_stacked[0][0, 0, :, :4], expected_b, rtol=0, atol=0)
