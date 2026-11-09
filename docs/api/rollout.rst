Rollout & Agent Loop
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni rollout is built on top of `vLLM-Omni
<https://github.com/vllm-project/vllm-omni>`_ for concurrent diffusion and
multimodal generation. The agent loop streams per-sample generation requests
to one or more rollout replicas, optionally fanning reward computation into
asynchronous reward-loop workers.

.. autosummary::
   :nosignatures:

   verl_omni.agent_loop.DiffusionAgentLoopWorker
   verl_omni.agent_loop.DiffusionSingleTurnAgentLoop
   verl_omni.agent_loop.DiffusionAgentLoopOutput
   verl_omni.workers.rollout.replica.DiffusionOutput

Diffusion Agent Loop
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.agent_loop.DiffusionAgentLoopWorker
   :members: __init__, generate_sequences

.. autoclass:: verl_omni.agent_loop.DiffusionSingleTurnAgentLoop
   :members: run

.. autoclass:: verl_omni.agent_loop.DiffusionAgentLoopOutput
   :members:

Variable-Length Reference Rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diffusion pipelines may return per-sample ``condition_video_rows`` and
``condition_audio_rows`` with different lengths. The Agent Loop validates each
``condition_*_row_count``, pads both fields to ``max_prompt_embed_length``, and
emits ``condition_*_rows_mask`` tensors. This fixed transport shape lets the
standard :class:`verl.experimental.agent_loop.AgentLoopManager` concatenate
outputs from multiple rollout workers without model-specific manager logic.

Before Actor training,
:func:`verl_omni.workers.utils.padding.embeds_padding_2_no_padding` removes the
global padding and stores the rows as jagged tensors. The DiffusionNFT engine
restores only the minibatch-local maximum immediately before model input. This
path raises an error for invalid tensor shapes, count or mask mismatches, or row
counts above ``max_prompt_embed_length``; it never truncates reference rows
silently.

Rollout Replica
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.workers.rollout.replica.DiffusionOutput
   :members:

vLLM-Omni Async Server
~~~~~~~~~~~~~~~~~~~~~~~~

The async server adapters wire the
:class:`verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer` and
:class:`verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMReplica`
classes to vLLM-Omni's diffusion-aware backend:

* ``verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server.vLLMOmniHttpServer``:
  subclass of vLLM's HTTP server that swaps the model config for
  :class:`~verl_omni.workers.config.DiffusionModelConfig`, skips LLM-only
  validation, and exposes a PIL → tensor converter for image responses.
* ``verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server.vLLMOmniReplica``:
  Ray actor wrapper that boots a vLLM-Omni engine per replica and forwards
  generation / weight-update / sleep & resume RPCs from the trainer.

These classes are heavy-weight wrappers that depend on running vLLM-Omni at
import time, so they are not introspected by autodoc. See
``verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py`` for full
source.

vLLM-Omni Utilities
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.utils.vllm_omni.utils.OmniTensorLoRARequest
   :members:

.. autoclass:: verl_omni.utils.vllm_omni.utils.VLLMOmniHijack
   :members:

.. autoclass:: verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension
   :members: update_weights_from_ipc
