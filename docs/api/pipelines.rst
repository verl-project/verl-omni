Pipelines Interface
================================

Last updated: |today| (API docstrings are auto-generated).

A *pipeline* in VeRL-Omni packages everything needed to plug a particular
model architecture into the training loop. Two adapter families are available:

- autoregressive omni models use a training-side
  :class:`~verl_omni.pipelines.model_base.OmniModelBase` and an optional
  rollout-side :class:`~verl_omni.pipelines.model_base.OmniRolloutPipelineBase`;
- diffusion models use a training-side adapter subclassing
  :class:`~verl_omni.pipelines.model_base.DiffusionModelBase` that handles
  scheduler setup, model-input construction, and the per-step forward /
  reverse-sampling logic used by RL algorithms (e.g. FlowGRPO);
- their optional rollout-side adapter is registered via
  :class:`~verl_omni.pipelines.model_base.VllmOmniPipelineBase` that hooks
  into vLLM-Omni's diffusion serving stack to expose log-probabilities.

Autoregressive training adapters are selected by ``(architecture, model_stage)``;
their rollout adapters are selected by the vLLM-Omni ``pipeline_name``. Diffusion
adapters are selected by matching
``(DiffusionModelConfig.architecture, DiffusionModelConfig.algorithm)`` against a
registered ``(architecture, algorithm)`` key. Diffusion architecture is read from
``model_index.json`` and the algorithm from
``actor_rollout_ref.model.algorithm``.

.. autosummary::
   :nosignatures:

   verl_omni.pipelines.model_base.OmniModelBase
   verl_omni.pipelines.model_base.OmniRolloutPipelineBase
   verl_omni.pipelines.model_base.DiffusionModelBase
   verl_omni.pipelines.model_base.VllmOmniPipelineBase
   verl_omni.pipelines.qwen_image_flow_grpo.QwenImage
   verl_omni.pipelines.qwen_image_mix_grpo.QwenImageMixGRPO
   verl_omni.pipelines.sd3_dpo.StableDiffusion3DPO
   verl_omni.pipelines.schedulers.flow_match_sde.FlowMatchSDEDiscreteScheduler

Model Base
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.pipelines.model_base.OmniModelBase
   :members: register, get_class, get_class_by_name,
             register_auto_classes,
             get_strip_modules, configure_processor, configure_tokenizer,
             configure_model, prepare_model_inputs

.. autoclass:: verl_omni.pipelines.model_base.OmniRolloutPipelineBase
   :members: register, get_class,
             build_stage_configs, rollout_flags, weight_sync_stage_ids, policy_stage_id,
             get_pipeline_id, ensure_pipeline_registered, get_engine_hf_overrides,
             get_stage_engine_extras, prepare_engine_prompt,
             postprocess_agent_loop_output,
             combine_engine_outputs

.. autoclass:: verl_omni.pipelines.model_base.DiffusionModelBase
   :members: register, get_class,
             build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step,
             validate_lora_config

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

Qwen-Image (MixGRPO)
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: verl_omni.pipelines.qwen_image_mix_grpo.QwenImageMixGRPO
   :members: build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step

.. autoclass:: verl_omni.pipelines.qwen_image_mix_grpo.QwenImageMixGRPOPipelineWithLogProb
   :members:

SD3 DPO
^^^^^^^^

.. autoclass:: verl_omni.pipelines.sd3_dpo.StableDiffusion3DPO
   :members: build_scheduler, set_timesteps,
             prepare_model_inputs, forward_and_sample_previous_step
