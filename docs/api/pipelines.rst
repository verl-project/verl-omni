Pipelines Interface
================================

Last updated: |today| (API docstrings are auto-generated).

A *pipeline* in VeRL-Omni packages everything needed to plug a particular
diffusion model architecture into the training loop:

- a **training-side adapter** subclassing
  :class:`~verl_omni.pipelines.model_base.DiffusionModelBase` that handles
  scheduler setup, model-input construction, and the per-step forward /
  reverse-sampling logic used by RL algorithms (e.g. FlowGRPO);
- an optional **rollout-side adapter** registered via
  :class:`~verl_omni.pipelines.model_base.VllmOmniPipelineBase` that hooks
  into vLLM-Omni's diffusion serving stack to expose log-probabilities.

Adapters are auto-selected by matching ``DiffusionModelConfig.architecture``
(read from the model's ``model_index.json``) against the registered name.

.. autosummary::
   :nosignatures:

   verl_omni.pipelines.model_base.DiffusionModelBase
   verl_omni.pipelines.model_base.VllmOmniPipelineBase
   verl_omni.pipelines.qwen_image_flow_grpo.QwenImage
   verl_omni.pipelines.schedulers.flow_match_sde.FlowMatchSDEDiscreteScheduler

Model Base
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.pipelines.model_base.DiffusionModelBase
   :members: register, get_class,
             build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step

.. autoclass:: verl_omni.pipelines.model_base.VllmOmniPipelineBase
   :members: register, get_class, get_pipeline_path

Pipeline Helpers
~~~~~~~~~~~~~~~~~

Convenience wrappers that dispatch to the registered subclass for the
current architecture. The Diffusers FSDP engine and the agent loop call
into these helpers rather than touching the registry directly.

.. automodule:: verl_omni.pipelines.utils
   :members: build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step

Schedulers
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.pipelines.schedulers.flow_match_sde.FlowMatchSDEDiscreteScheduler
   :members: step, sample_previous_step

.. autoclass:: verl_omni.pipelines.schedulers.flow_match_sde.FlowMatchSDEDiscreteSchedulerOutput
   :members:

Built-in Pipelines
~~~~~~~~~~~~~~~~~~~

Qwen-Image (FlowGRPO)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: verl_omni.pipelines.qwen_image_flow_grpo.QwenImage
   :members: build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step

.. autoclass:: verl_omni.pipelines.qwen_image_flow_grpo.QwenImagePipelineWithLogProb
   :members:
