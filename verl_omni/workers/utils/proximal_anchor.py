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

"""Reuse a qualified pre-update actor output as its detached proximal anchor.

This loss requires an active actual-writer lease. It is not a rollout-logprob
bypass and does not enable the optimization by itself. The coordinator must
establish algorithm, model, data and execution-shape eligibility first.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested

from verl_omni.workers.utils.parameter_revision import ParameterRevision

_ANCHOR_LEASE = ContextVar("verl_proximal_anchor_lease", default=None)
_DIAGNOSTIC_ENTROPY = ContextVar("verl_proximal_diagnostic_entropy", default=False)


@contextmanager
def proximal_anchor_loss_scope(tracker: ParameterRevision, *, diagnostic_entropy: bool = False):
    """Allow native PPO loss reuse only within one pending, non-overlapped update."""
    if _ANCHOR_LEASE.get() is not None:
        raise RuntimeError("Nested proximal anchor scope")
    if not isinstance(diagnostic_entropy, bool):
        raise TypeError("Diagnostic entropy policy must be explicit")
    tracker.assert_update_pending()
    token = _ANCHOR_LEASE.set(tracker)
    entropy_token = _DIAGNOSTIC_ENTROPY.set(diagnostic_entropy)
    try:
        yield
    finally:
        _DIAGNOSTIC_ENTROPY.reset(entropy_token)
        _ANCHOR_LEASE.reset(token)


def prepare_proximal_loss(config, model_output, data):
    """Supply an anchor only when the worker has entered its qualified lease."""
    tracker = _ANCHOR_LEASE.get()
    if tracker is None:
        return model_output, data
    tracker.assert_update_pending()
    if _DIAGNOSTIC_ENTROPY.get() and (config.entropy_coeff != 0 or config.calculate_entropy):
        raise ValueError("Diagnostic entropy policy conflicts with the actual loss")
    if config.ppo_epochs != 1:
        raise ValueError("Proximal reuse requires exactly one optimizer update")
    if any(key in data for key in ("old_log_probs", "rollout_is_weights")):
        raise ValueError("A stale anchor or old-dependent importance weight was supplied")
    if config.policy_loss.get("loss_mode", "vanilla") not in {"vanilla", "gspo"}:
        raise ValueError("Unqualified policy loss for proximal reuse")
    if any(key.startswith("sc_") for key in model_output):
        raise ValueError("Score centering must use the ordinary old-policy path")
    # Use the native response alignment: the token predicting the first answer
    # is the last prompt token. Never use rollout probabilities as this anchor.
    log_probs = model_output["log_probs"]
    padded = no_padding_2_padding(log_probs.detach(), data)
    mask = data.select("response_mask").to_padded_tensor()["response_mask"].bool()
    if padded.shape != mask.shape or not torch.isfinite(padded[mask]).all():
        raise ValueError("Invalid pre-update response log probabilities")
    view = data.clone(recurse=False)
    view["old_log_probs"] = response_to_nested(padded, data["response_mask"])
    # Computing diagnostic entropy must not add a zero-weight backward graph.
    # A nonzero entropy objective remains the original autograd expression.
    if config.entropy_coeff == 0 and not config.calculate_entropy and "entropy" in model_output:
        model_output = {key: value for key, value in model_output.items() if key != "entropy"}
    return model_output, view


def proximal_ppo_loss(config, model_output, data, dp_group=None):
    """Call the native PPO loss with a leased, detached pre-update anchor."""
    from verl.workers.utils.losses import ppo_loss

    model_output, data = prepare_proximal_loss(config, model_output, data)
    return ppo_loss(config, model_output, data, dp_group=dp_group)
