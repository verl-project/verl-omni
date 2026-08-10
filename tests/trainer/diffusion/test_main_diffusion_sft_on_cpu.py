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

import os
from pathlib import Path

os.environ.setdefault("VERL_OMNI_SKIP_AUTO_IMPORTS", "1")


def test_main_omni_routes_sft_trainer_in_source():
    root = Path(__file__).resolve().parents[3]
    main_omni = (root / "verl_omni" / "trainer" / "main_omni.py").read_text()
    omni_trainer = (root / "verl_omni" / "trainer" / "omni" / "ray_omni_trainer.py").read_text()
    diffusion_trainer = (root / "verl_omni" / "trainer" / "diffusion" / "ray_diffusion_trainer.py").read_text()

    assert 'trainer_type == "sft"' in main_omni
    assert "return SFTRayTrainer" in main_omni
    assert 'trainer_type in {"direct_preference", "sft"}' in main_omni
    assert "class SFTRayTrainer" in omni_trainer
    assert "class SFTRayTrainer" not in diffusion_trainer
