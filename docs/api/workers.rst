Workers Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni workers wrap the Diffusers / FSDP training engine, the rollout
engine (vLLM-Omni), and the optional reference policy. The single-controller
trainer drives them through a unified RPC layer.

.. autosummary::
   :nosignatures:

   verl_omni.workers.engine_workers.TrainingWorker
   verl_omni.workers.engine_workers.ActorRolloutRefWorker
   verl_omni.workers.engine.fsdp.diffusers_impl.DiffusersFSDPEngine
   verl_omni.workers.config.DiffusionRolloutAlgoConfig
   verl_omni.workers.config.DiffusionPipelineConfig
   verl_omni.workers.config.DiffusionSamplingConfig

Engine Workers
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.workers.engine_workers.TrainingWorker
   :members: __init__, reset, set_loss_fn, to,
             train_mini_batch, train_batch, infer_batch,
             save_checkpoint, load_checkpoint

.. autoclass:: verl_omni.workers.engine_workers.ActorRolloutRefWorker
   :members: __init__, init_model,
             compute_log_prob, compute_ref_log_prob, update_actor,
             update_weights, save_checkpoint, load_checkpoint

Diffusers FSDP Engine
~~~~~~~~~~~~~~~~~~~~~~

The Diffusers FSDP engine implements the
:class:`verl.workers.engine.base.BaseEngine` interface for diffusion
transformer backbones (e.g. Qwen-Image), including LoRA, mixed precision,
and parameter / optimizer offloading.

.. autoclass:: verl_omni.workers.engine.fsdp.diffusers_impl.DiffusersFSDPEngine
   :members: __init__, initialize,
             train_mode, eval_mode,
             forward_step, forward_backward_batch,
             prepare_model_inputs, prepare_model_outputs,
             optimizer_step, optimizer_zero_grad, lr_scheduler_step,
             get_data_parallel_rank, get_data_parallel_size, get_data_parallel_group,
             save_checkpoint, load_checkpoint, get_per_tensor_param,
             to, disable_adapter

Loss Functions
~~~~~~~~~~~~~~~~~

.. automodule:: verl_omni.workers.utils.losses
   :members: diffusion_loss

Padding Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: verl_omni.workers.utils.padding
   :members: embeds_padding_2_no_padding

Worker Configs
~~~~~~~~~~~~~~~~~

The configs below are dataclass mirrors of the YAML / Hydra options consumed
by the engine workers. They are typically built from
:class:`omegaconf.DictConfig` via :func:`verl.utils.config.omega_conf_to_dataclass`.

.. autoclass:: verl_omni.workers.config.DiffusionModelConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionActorConfig
   :members:

.. autoclass:: verl_omni.workers.config.FSDPDiffusionActorConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionLossConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionRolloutConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionRolloutAlgoConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionPipelineConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionSamplingConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionPipelineConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionRolloutAlgoConfig
   :members:

.. autoclass:: verl_omni.workers.config.DiffusionSamplingConfig
   :members:
