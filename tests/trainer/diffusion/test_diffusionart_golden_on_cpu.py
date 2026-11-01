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
"""Golden-value equivalence of verl-omni's DiffusionART math with the authors' implementation.

Necessity: ``diffusion_art.py`` is a re-implementation from the paper, not a copy of the authors' code.
The fixture ``fixtures/diffusionart_golden_8652c8a.json`` was produced by running vanzll/DiffusionART at
commit ``8652c8a100b01a5124781124a8ccdfc4d6bef158`` (``art_scalar_covariance`` with the paper
configuration ``unit_normalized_covariance`` / ``zero_signal_mode=exact`` and ``art_loss`` with
``velocity_linear_sigma`` / ``scheduler_reference_metric`` / ``nu=0.8`` / ``delta=1e-5``) on seeded random
inputs: 4 prompts x 8 endpoints (one prompt with constant rewards), latent shape (4, 5, 5), five flow times
including ``t = 1`` and ``t_next = 0``, and current velocities with small drift plus large drift on every
third row. No ``visual_rl`` dependency is needed here; only the inputs and outputs are stored, as JSON
(float32 values written with their exact repr; see the fixture's ``provenance`` field).
"""

import json
from pathlib import Path

import torch

from verl_omni.trainer.diffusion.diffusion_art import art_guidance, art_loss

FIXTURE = Path(__file__).parent / "fixtures" / "diffusionart_golden_8652c8a.json"
_DTYPES = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}


def load_fixture() -> dict:
    raw = json.loads(FIXTURE.read_text())
    assert "8652c8a100b01a5124781124a8ccdfc4d6bef158" in raw["provenance"]["reference"]

    def decode(section: dict) -> dict:
        return {
            key: torch.tensor(value["data"], dtype=_DTYPES[value["dtype"]]) if isinstance(value, dict) else value
            for key, value in section.items()
        }

    return {name: decode(raw[name]) for name in ("inputs", "guidance", "loss")}


def test_guidance_matches_authors_covariance_directions():
    fixture = load_fixture()
    inputs = fixture["inputs"]
    directions, active = art_guidance(
        inputs["endpoints"], inputs["rewards"], inputs["group_index"], inputs["num_groups"]
    )
    assert active.tolist() == fixture["guidance"]["active"].tolist()
    assert active.tolist() == [True, False, True, True]
    torch.testing.assert_close(directions, fixture["guidance"]["directions"], atol=1e-5, rtol=0)


def test_loss_mask_and_gradient_match_authors_art_loss():
    fixture = load_fixture()
    inputs = fixture["inputs"]
    expected = fixture["loss"]
    directions, _ = art_guidance(inputs["endpoints"], inputs["rewards"], inputs["group_index"], inputs["num_groups"])
    current = inputs["current_velocity"].clone().requires_grad_(True)
    terms = art_loss(
        current,
        inputs["old_velocity"],
        directions[inputs["group_index"]],
        inputs["sigma"],
        inputs["sigma_next"],
        guidance_gain=1.0,
        loss_space="velocity_linear_sigma",
        mask_metric="sde_kl",
        mask_threshold=1e-5,
        reference_noise_level=0.8,
    )
    assert terms["keep_mask"].tolist() == expected["keep_mask"].tolist()
    assert terms["masked"].tolist() == expected["masked"].tolist()
    assert terms["masked"].any() and terms["keep_mask"].any()  # the fixture exercises both branches
    torch.testing.assert_close(terms["per_item_loss"], expected["per_item_loss"], atol=1e-5, rtol=0)
    torch.testing.assert_close(terms["target"], expected["target"], atol=1e-5, rtol=0)
    torch.testing.assert_close(terms["movement_score"], expected["movement_score"], atol=1e-5, rtol=0)
    torch.testing.assert_close(terms["outward_alignment"], expected["outward_alignment"], atol=1e-5, rtol=0)
    (grad,) = torch.autograd.grad(terms["per_item_loss"].mean(), current)
    torch.testing.assert_close(grad, expected["grad_current_velocity"], atol=1e-5, rtol=0)
