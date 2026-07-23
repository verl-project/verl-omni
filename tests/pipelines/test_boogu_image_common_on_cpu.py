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

"""CPU unit tests for the Boogu-Image convention-bridge helpers.

These pin the two invariants GPU validation surfaced:

1. Boogu time-shift parameters must be read from the **raw** scheduler
   config JSON, because diffusers drops the custom keys from
   ``scheduler.config`` (an empty/filtered mapping must therefore degrade to
   a no-op, not silently mis-schedule).
2. The scheduler-native timestep -> Boogu ``t`` mapping is ``t = 1 - sigma``.
"""

import json

import torch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    boogu_timestep_from_scheduler,
    load_boogu_shift_config,
    resolve_time_shift,
)

# Released Base/Edit scheduler config (verified against the HF checkpoints).
_RELEASED_STATIC_V1 = {
    "do_shift": True,
    "dynamic_time_shift": False,
    "time_shift_version": "v1",
    "seq_len": 4096,
    "num_train_timesteps": 1000,
}


def test_static_v1_resolves_to_max_shift_mu():
    """seq_len=4096 sits at the top of the linear map -> mu == max_shift (1.15)."""
    mu, shift = resolve_time_shift(_RELEASED_STATIC_V1, num_tokens=128 * 128)
    assert mu is not None
    assert shift == 1.0
    assert abs(mu - 1.15) < 1e-6


def test_dropped_config_degrades_to_noop():
    """This is the regression: diffusers filters the custom keys off
    ``scheduler.config``, leaving a mapping without ``do_shift`` etc. Resolving
    against that empty view must NOT silently apply an unshifted schedule as if
    it were correct — it returns the no-op sentinel so callers that wrongly pass
    ``scheduler.config`` are visibly unshifted rather than subtly wrong."""
    mu, shift = resolve_time_shift({}, num_tokens=128 * 128)
    assert mu is None
    assert shift == 1.0


def test_v2_dynamic_multiplicative_shift():
    cfg = {"do_shift": True, "dynamic_time_shift": True, "time_shift_version": "v2"}
    mu, shift = resolve_time_shift(cfg, num_tokens=4096)
    assert mu is None
    # m = sqrt(num_tokens) / (half_scaling * 2) = 64 / 120
    assert abs(shift - (64.0 / 120.0)) < 1e-6


def test_no_shift_config():
    mu, shift = resolve_time_shift({"do_shift": False}, num_tokens=4096)
    assert mu is None
    assert shift == 1.0


def test_load_shift_config_reads_raw_json(tmp_path):
    sched_dir = tmp_path / "scheduler"
    sched_dir.mkdir()
    (sched_dir / "scheduler_config.json").write_text(json.dumps(_RELEASED_STATIC_V1))
    loaded = load_boogu_shift_config(str(tmp_path))
    assert loaded["seq_len"] == 4096
    assert loaded["time_shift_version"] == "v1"


def test_load_shift_config_missing_returns_empty(tmp_path):
    assert load_boogu_shift_config(str(tmp_path)) == {}


def test_timestep_mapping_inverts_sigma():
    # Scheduler-native timesteps are sigma * N (descending); Boogu t = 1 - sigma.
    n = 1000
    timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0])
    boogu_t = boogu_timestep_from_scheduler(timesteps, n)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.allclose(boogu_t, expected, atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                import tempfile

                with tempfile.TemporaryDirectory() as d:
                    import pathlib

                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"{name} OK")
