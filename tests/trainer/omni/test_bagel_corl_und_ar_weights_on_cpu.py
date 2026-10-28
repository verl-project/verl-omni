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

"""CPU regression: the UND AR replica must load real weights, not decode from ``dummy``.

The UND AR (Thinker) replica is its own Ray actor over a slice of the actor pool
(``init_colocated`` -> ``RolloutMode.COLOCATED``), and the inherited publish never hands it
weights: the V1 trainer forces ``checkpoint_engine.backend="naive"`` and
``CheckpointEngineManager.update_weights`` early-returns for that backend after publishing
through the *actor* to its own in-process GEN server (``verl/checkpoint_engine/base.py``).
``_publish_und_ar_weights`` pushes the policy into the Thinker stage only, so the DiT stage
decodes with whatever it loads at launch for the rest of the run.

``verl`` 0.9.0.dev -- the sibling checkout the 2026-10-13/14 runs reached through
``PYTHONPATH`` -- promoted precisely this case in ``vllm_async_server.__init__``::

    if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
        logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
        self.config.load_format = "auto"

The installed ``verl`` 0.10.0.dev dropped that promotion and forwards ``load_format``
untouched, so leaving the key out of the AR replica's config resolves to
``RolloutConfig.load_format``'s default, ``"dummy"``, and the Thinker is launched with
``--load_format dummy``.

Measured on 2026-10-15 (``bagel_corl_prfix_t07_20261015_170336_a1``, devices 4-7), which is
what these tests pin: ``actor/entropy`` 11.17 ~= ln(vocab) (uniform), ``episode/K=0.0``,
``episode/K_zero=1.0``, ``und/no_image_credit=1.0``, ``gen/skipped_no_groups=1.0``,
``rollout_corr/rollout_ppl`` 46.5 against ``training_ppl`` 247736, every UND turn of all 36
val samples looping ``\\tbuf_widget`` / ``\\U0001f4a8third`` with no ``<tool_call>``, and no
checkpoint-loading line anywhere in the Thinker stage's engine log -- while the reward
server and both DiT stages, which do load, each logged their shard progress.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync

# Two ``bagel_think`` stages over the replica's two logical devices, the shape the AR deploy
# yaml ships. ``max_model_len`` is what the builder reads back for the replica's context.
DEPLOY_YAML = """
pipeline: bagel_think
stages:
  - stage_id: 0
    max_model_len: 16384
    max_num_batched_tokens: 16384
    devices: "0"
  - stage_id: 1
    max_num_batched_tokens: 2048
    devices: "1"
    input_connectors:
      from_stage_0: shared_memory_connector
"""


def _entry_config():
    """The actor-entry config the builder clones, in the shape the recipe builds it."""
    return OmegaConf.create(
        {
            "data": {"max_prompt_length": 2048, "max_response_length": 2048},
            "actor_rollout_ref": {
                "model": {
                    "path": "/nonexistent/bagel-snapshot",
                    "tokenizer_path": "/nonexistent/bagel-snapshot",
                    "architecture": "OmniBagelForConditionalGeneration",
                    "trust_remote_code": True,
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    # The recipe merges the LoRA into the base weights, which is why the AR
                    # replica is a full-weight, adapter-free engine.
                    "lora": {"merge": True},
                },
                "rollout": {
                    "name": "vllm_omni",
                    # The entry value is the GEN hybrid's: the actor pushes its weights, so
                    # ``dummy`` is correct there and must survive this builder untouched.
                    "load_format": "dummy",
                    "tensor_model_parallel_size": 2,
                    # The V1 trainer forces this backend, so its publish skips the AR replica.
                    "checkpoint_engine": {"backend": "naive"},
                    "agent": {
                        "und_ar_serving_ready": 1,
                        "und_deploy_config": "<tmp>",
                        "und_hf_config_path": "<tmp>",
                        "und_n_gpus": 2,
                        "und_gpu_memory_utilization": 0.40,
                    },
                },
            },
        }
    )


def _trainer(tmp_path: Path):
    deploy = tmp_path / "bagel_corl_deploy_ar.yaml"
    deploy.write_text(DEPLOY_YAML)
    # ``__new__`` rather than ``__init__``: the builder reads only the config and a few
    # static helpers, while constructing the real trainer needs a full Hydra config.
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.config = _entry_config()
    trainer.config.actor_rollout_ref.rollout.agent.und_deploy_config = str(deploy)
    trainer.config.actor_rollout_ref.rollout.agent.und_hf_config_path = str(tmp_path / "llm_config.json")
    return trainer


def test_und_ar_replica_is_launched_with_real_weights(tmp_path):
    """The regression itself: ``dummy`` here is a whole run of random-weight decoding.

    ``load_format`` decides whether the Thinker stage has weights at all, and nothing in the
    loop can fix it later -- the naive publish path stops at the actor's own GEN server.
    """
    und = _trainer(tmp_path)._build_und_ar_entrypoint_config()
    load_format = str(und.actor_rollout_ref.rollout.get("load_format") or "")
    assert load_format != "dummy", "the UND AR replica must not decode from random weights"
    assert load_format == OmniBagelCoRLTrainerSync.UND_AR_LOAD_FORMAT
    assert load_format == "auto", "vLLM's own default, i.e. load the checkpoint at launch"


def test_the_gen_hybrid_still_takes_its_weights_from_the_actor(tmp_path):
    """The pin must not leak into the entry config: the GEN engine must stay ``dummy``.

    The builder deep-copies the entry config, and the GEN hybrid is the one replica the actor
    *does* publish to -- loading the checkpoint there would only put a second copy of a
    27.5GiB checkpoint on the actor's own card.
    """
    trainer = _trainer(tmp_path)
    _ = trainer._build_und_ar_entrypoint_config()
    assert trainer.config.actor_rollout_ref.rollout.load_format == "dummy"
    assert trainer.config.actor_rollout_ref.rollout.agent.und_n_gpus == 2


def test_the_rest_of_the_ar_config_is_unchanged(tmp_path):
    """A guard against a "fix" that swaps the whole replica config for something else."""
    rollout = _trainer(tmp_path)._build_und_ar_entrypoint_config().actor_rollout_ref.rollout
    assert rollout.name == "vllm_omni"
    assert rollout.tensor_model_parallel_size == 2, "the replica's width declaration"
    assert rollout.max_model_len == 16384, "read back from the AR deploy config"
    assert rollout.engine_kwargs.vllm_omni.output_mode == "ar", "never the GEN diffusion stack"
    assert rollout.engine_kwargs.vllm_omni.deploy_config.endswith("bagel_corl_deploy_ar.yaml")
    assert rollout.engine_kwargs.vllm_omni.stage_overrides == {
        "0": {"tensor_parallel_size": 1},
        "1": {"tensor_parallel_size": 1},
    }
    assert rollout.engine_kwargs.vllm_omni.weight_sync_stage_ids == [0], "the Thinker only, never the DiT"


class _FakeUndEntrypointConfig:
    """A replica config that would be born weightless -- the failure, not the fix."""

    def __init__(self, load_format):
        self.actor_rollout_ref = OmegaConf.create({"model": {"lora_rank": 0}, "rollout": {"load_format": load_format}})


@pytest.mark.parametrize("load_format", ["dummy", "", None], ids=["dummy", "empty", "unset"])
def test_dual_role_init_refuses_a_weightless_ar_replica(tmp_path, monkeypatch, load_format):
    """Fail loud at init instead of losing the UND lane quietly for 200 steps.

    Nothing downstream can detect this: the rollout succeeds, every metric is finite, and
    ``episode/K=0.0`` is also what a legitimately skipped GEN lane looks like.
    """
    trainer = _trainer(tmp_path)
    monkeypatch.setattr(
        OmniBagelCoRLTrainerSync,
        "_build_und_ar_entrypoint_config",
        lambda self: _FakeUndEntrypointConfig(load_format),
    )
    with pytest.raises(RuntimeError, match="random weights"):
        trainer._ensure_dual_role_rollout()


def test_the_pin_has_to_live_in_the_builder_not_the_deploy_yaml():
    """Why this is a builder fix: the recipe's AR deploy config never carried a load format.

    The yaml cannot carry one either -- ``load_format`` is a plain top-level engine arg, so
    the deploy-config path copies it into *every* stage and it would reach the DiT stage too.
    """
    repo_root = Path(__file__).resolve().parents[3]
    deploy = repo_root / "examples/agenticllmgrpo_trainer/bagel/bagel_corl_deploy_ar.yaml"
    if not deploy.is_file():  # pragma: no cover -- tests can run against an installed tree
        pytest.skip(f"recipe deploy config not present at {deploy}")
    text = deploy.read_text()
    assert "pipeline: bagel_think" in text
    assert "load_format" not in text
