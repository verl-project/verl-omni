Trainer Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni provides Ray-based trainers for diffusion / multimodal RL. Today,
:class:`~verl_omni.trainer.diffusion.ray_diffusion_trainer.RayDiffusionOnPolicyTrainer`
is the primary entrypoint and orchestrates Flow-GRPO training across actor,
rollout, reference policy, and reward workers.

.. autosummary::
   :nosignatures:

   verl_omni.trainer.diffusion.ray_diffusion_trainer.RayDiffusionOnPolicyTrainer
   verl_omni.trainer.main_diffusion.TaskRunner

Core Trainer
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.trainer.diffusion.ray_diffusion_trainer.RayDiffusionOnPolicyTrainer
   :members: __init__, init_workers, fit

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
             register_diffusion_loss,
             get_diffusion_loss_fn,
             register_diffusion_adv_est,
             get_diffusion_adv_estimator_fn,
             compute_flow_grpo_outcome_advantage,
             compute_diffusion_loss_flow_grpo,
             compute_diffusion_loss_grpo_guard,
             kl_penalty_image

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
