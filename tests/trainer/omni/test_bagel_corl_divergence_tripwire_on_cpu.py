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

"""CPU tests for the Bagel Co-RL divergence tripwire.

``bagel_corl_rm1_0123_20261012_084724`` (devices 0-3) spent its last ~150 steps producing
an unusable policy and said nothing: ``actor/grad_norm`` ran 0.32 -> 310 and
``rollout_corr/kl`` 0.03 -> 175 while ``actor/entropy`` fell to 0.02. Neither number is
trended anywhere, so the only way to notice was to read the run afterwards. The recipe has
no KL anchor and no entropy bonus, so a policy past this point cannot recover on its own --
the tripwire exists to make that visible on the step it happens.
"""

import logging

import pytest
import torch

from verl_omni.trainer.omni.bagel_corl_trainer import (
    OmniBagelCoRLTrainerSync,
    _as_ray_actor_class,
    _metric_scalar,
)


class _PlainActorFixture:
    """Stands in for ``GlobalRequestLoadBalancer`` as the *installed* ``verl`` ships it."""


def test_load_balancer_actor_class_is_decorated_only_when_the_class_is_plain():
    """Both ``verl`` shapes must work, and neither may be decorated twice.

    The installed ``verl`` ships ``GlobalRequestLoadBalancer`` as a *plain* class and wraps
    it at instantiation itself -- ``ray.remote(GlobalRequestLoadBalancer).remote(...)`` in
    ``verl.experimental.teacher_loop.teacher_model`` -- while the older sibling shipped it
    already ``@ray.remote``-decorated. Calling ``.remote`` on the bare class is what
    aborted ``on_init_end`` on devices 4-7 (``bagel_corl_prfix_t07_20261015_105824_a1``):

        AttributeError: type object 'GlobalRequestLoadBalancer' has no attribute
        'remote'. Did you mean: '_return_value'?

    Re-decorating the decorated one fails too, so the presence of ``remote`` -- the only
    difference between the two shapes -- has to be the version marker.
    """
    import ray

    assert not hasattr(_PlainActorFixture, "remote"), "the fixture must start un-decorated"
    assert hasattr(_as_ray_actor_class(_PlainActorFixture), "remote"), "plain class must be wrapped"

    decorated = ray.remote(_PlainActorFixture)
    assert _as_ray_actor_class(decorated) is decorated, "an actor class must not be re-decorated"


def _tripwire_trainer():
    # ``__new__`` rather than ``__init__``: the tripwire reads only ``global_steps`` and
    # its own thresholds, and constructing the real trainer needs a full Hydra config.
    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.global_steps = 51
    return trainer


def test_metric_scalar_unwraps_floats_tensors_and_metric_objects():
    """The step metrics dict mixes all three; a tripwire that read one would miss the run."""
    from verl.utils.metric import Metric

    metrics = {
        "plain": 0.5,
        "tensor": torch.tensor(2.0),
        "metric": Metric(value=3.0, aggregation="mean"),
        "non_scalar_tensor": torch.zeros(2),
        "not_a_number": "n/a",
    }
    assert _metric_scalar(metrics, "plain") == 0.5
    assert _metric_scalar(metrics, "tensor") == 2.0
    assert _metric_scalar(metrics, "metric") == 3.0
    assert _metric_scalar(metrics, "non_scalar_tensor") is None
    assert _metric_scalar(metrics, "not_a_number") is None
    assert _metric_scalar(metrics, "absent") is None


def test_the_tripwire_warns_on_kl_and_publishes_the_metric(caplog):
    """``_084724`` ran 150 steps of this without saying anything; the tripwire must say so."""
    trainer = _tripwire_trainer()
    metrics = {"rollout_corr/kl": trainer.DIVERGENCE_KL_LIMIT + 1.0, "actor/grad_norm": 1.5}

    with caplog.at_level(logging.WARNING):
        trainer._divergence_tripwire(metrics)

    assert metrics["bagel_corl/divergence_warning"] == 1.0
    assert "actor/rollout mismatch or gradient health threshold exceeded" in caplog.text
    assert "left its trust region" not in caplog.text
    assert "rollout_corr/kl=21" in caplog.text


def test_the_tripwire_warns_on_grad_norm_alone(caplog):
    trainer = _tripwire_trainer()
    metrics = {"rollout_corr/kl": 0.4, "actor/grad_norm": trainer.DIVERGENCE_GRAD_NORM_LIMIT + 1.0}

    with caplog.at_level(logging.WARNING):
        trainer._divergence_tripwire(metrics)

    assert metrics["bagel_corl/divergence_warning"] == 1.0
    assert "actor/grad_norm" in caplog.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_the_tripwire_reports_nonfinite_as_runaway(value):
    """``nan > limit`` is False, so a non-finite reading must be caught explicitly."""
    trainer = _tripwire_trainer()
    metrics = {"rollout_corr/kl": value, "actor/grad_norm": 1.5}

    trainer._divergence_tripwire(metrics)

    assert metrics["bagel_corl/divergence_warning"] == 1.0


def test_the_tripwire_stays_quiet_and_clears_when_healthy(caplog):
    trainer = _tripwire_trainer()
    hot = {"rollout_corr/kl": trainer.DIVERGENCE_KL_LIMIT + 1.0}
    with caplog.at_level(logging.WARNING):
        trainer._divergence_tripwire(hot)
    assert trainer._divergence_warned is True

    # Same excursion: keep the series, do not repeat the line every step.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        trainer._divergence_tripwire(hot)
    assert "actor/rollout mismatch or gradient health threshold exceeded" not in caplog.text

    healthy = {"rollout_corr/kl": 0.4, "actor/grad_norm": 1.2}
    trainer._divergence_tripwire(healthy)
    assert healthy["bagel_corl/divergence_warning"] == 0.0
    assert trainer._divergence_warned is False

    # A second excursion after recovery warns again.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        trainer._divergence_tripwire(hot)
    assert "actor/rollout mismatch or gradient health threshold exceeded" in caplog.text


def test_the_tripwire_aborts_only_when_opted_in(monkeypatch):
    hot = {"rollout_corr/kl": _tripwire_trainer().DIVERGENCE_KL_LIMIT + 1.0}

    # Default is warn-only; a cross-engine mismatch alone does not prove policy divergence.
    default = _tripwire_trainer()
    default._divergence_tripwire(dict(hot))
    assert default._divergence_warned is True

    monkeypatch.setenv("BAGEL_CORL_DIVERGENCE_ABORT", "1")
    with pytest.raises(RuntimeError, match="actor/rollout mismatch or gradient health threshold exceeded"):
        _tripwire_trainer()._divergence_tripwire(dict(hot))


def test_the_tripwire_ignores_missing_keys():
    """A step that skipped the actor (or a CPU-only test) has neither key."""
    trainer = _tripwire_trainer()
    metrics = {}
    trainer._divergence_tripwire(metrics)
    assert metrics["bagel_corl/divergence_warning"] == 0.0
