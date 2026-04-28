Rollout & Agent Loop
================================

Last updated: 04/28/2026 (API docstrings are auto-generated).

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
