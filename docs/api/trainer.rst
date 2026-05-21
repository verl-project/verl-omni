Trainer Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni provides Ray-based trainers for diffusion / multimodal RL.
:class:`~verl_omni.trainer.main_diffusion.TaskRunner` builds worker mappings and
dispatches to a trainer subclass selected by ``algorithm.trainer_type``:

- ``policy_gradient`` → :class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.PolicyGradientRayTrainer`
  (FlowGRPO, MixGRPO, DanceGRPO, GRPO-Guard; multi-timestep reverse-process PG)
- ``direct_preference`` → :class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.DirectPreferenceRayTrainer`
  (DPO, DiffusionNFT, AWM; single forward-timestep preference updates)

Both subclasses inherit shared worker init from
:class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.BaseRayDiffusionTrainer`.
Rollout and reward engines are initialized only when ``algorithm.sample_source=online``.

.. autosummary::
   :nosignatures:

   verl_omni.trainer.diffusion.ray_diffusion_trainer.BaseRayDiffusionTrainer
   verl_omni.trainer.diffusion.ray_diffusion_trainer.PolicyGradientRayTrainer
   verl_omni.trainer.diffusion.ray_diffusion_trainer.DirectPreferenceRayTrainer
   verl_omni.trainer.main_diffusion.TaskRunner

Base Ray Diffusion Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.BaseRayDiffusionTrainer`
owns colocated actor/ref worker setup, dataloaders, validation helpers, and
checkpointing. ``init_workers`` always builds actor/ref workers; rollout and
reward engines are added only when ``algorithm.sample_source=online``.

.. autoclass:: verl_omni.trainer.diffusion.ray_diffusion_trainer.BaseRayDiffusionTrainer
   :members: __init__, init_workers

Policy Gradient Ray Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.PolicyGradientRayTrainer`
implements the online training loop for FlowGRPO-style algorithms: rollout
generation, reward scoring, advantage estimation over denoising timesteps, and
actor updates.

.. autoclass:: verl_omni.trainer.diffusion.ray_diffusion_trainer.PolicyGradientRayTrainer
   :members: fit

Direct Preference Ray Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.DirectPreferenceRayTrainer`
is the extension point for direct-preference algorithms (DPO, DiffusionNFT, AWM)
that train with single forward-timestep updates rather than a full multi-step
SDE trajectory. The ``fit`` implementation is not yet available in-tree.

.. autoclass:: verl_omni.trainer.diffusion.ray_diffusion_trainer.DirectPreferenceRayTrainer
   :members: fit

.. autofunction:: verl_omni.trainer.diffusion.ray_diffusion_trainer.compute_advantage

Entry Point
~~~~~~~~~~~~~~~~~

.. automodule:: verl_omni.trainer.main_diffusion
   :members: main, run_diffusion, TaskRunner

Diffusion Algorithms
~~~~~~~~~~~~~~~~~~~~~

The :mod:`verl_omni.trainer.diffusion.diffusion_algos` module provides the
loss-function and advantage-estimator registries used by the trainer. Custom
losses and advantage estimators can be registered via the decorators below.

.. automodule:: verl_omni.trainer.diffusion.diffusion_algos
   :members: DiffusionAdvantageEstimator,
             DiffusionLossFn,
             DiffusionLossResult,
             register_diffusion_loss,
             get_diffusion_loss_fn,
             register_diffusion_adv_est,
             get_diffusion_adv_estimator_fn,
             compute_flow_grpo_outcome_advantage,
             FlowGRPOLoss,
             GRPOGuardLoss,
             KLLoss,

Trainer Config
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.trainer.config.algorithm.DiffusionAlgoConfig
   :members:

Metrics
~~~~~~~

.. automodule:: verl_omni.trainer.diffusion.diffusion_metric_utils
   :members: compute_data_metrics_diffusion,
             compute_timing_metrics_diffusion,
             compute_throughput_metrics_diffusion
