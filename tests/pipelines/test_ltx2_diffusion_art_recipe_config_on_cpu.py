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
"""The LTX-2.3 DiffusionART recipe's model pipeline must parse on the rollout server as in the trainer."""

import re
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.pipelines.ltx2_diffusion_art.config import LTXARTPipelineConfig, ltx_guidance_scales
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

_REPO = Path(__file__).resolve().parents[2]
_RECIPE = _REPO / "examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh"


def _recipe_model_pipeline_overrides() -> list[str]:
    """Return the recipe's ``actor_rollout_ref.model.pipeline.*`` overrides with shell variables expanded."""
    script = _RECIPE.read_text()
    variables = dict(re.findall(r"^(\w+)=(\S+)$", script, flags=re.MULTILINE))
    overrides = re.findall(r"^\s*(\+?actor_rollout_ref\.model\.pipeline\.\S+=\S+)\s*\\$", script, flags=re.MULTILINE)
    return [re.sub(r"\$(\w+)", lambda match: variables[match.group(1)], override) for override in overrides]


def test_recipe_model_pipeline_parses_on_the_rollout_server_like_the_trainer(tmp_path):
    overrides = _recipe_model_pipeline_overrides()
    assert any("video_cfg_scale" in override for override in overrides)
    with initialize_config_dir(config_dir=str(_REPO / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={tmp_path}",
                "+actor_rollout_ref.model.architecture=LTX2Pipeline",
                "++actor_rollout_ref.model.load_tokenizer=False",
                *overrides,
            ],
        )

    server = DiffusionStrategy(SimpleNamespace()).init_model_config(cfg.actor_rollout_ref.model)
    trainer = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)

    assert isinstance(server.pipeline, LTXARTPipelineConfig)
    assert ltx_guidance_scales(server.pipeline) == ltx_guidance_scales(trainer.pipeline) == (1.5, 3.0)
