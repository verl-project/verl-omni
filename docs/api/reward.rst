Reward Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni reward pipelines support both rule-based scoring (e.g. JPEG
compressibility) and model-based generative reward models (e.g. OCR via a
vision-language model served behind an OpenAI-compatible router). Reward
computation is dispatched per sample by reward managers. The input-agnostic
:class:`~verl_omni.reward_loop.reward_manager.MultiRewardManager` provides the
shared named-term dispatch, per-term outputs, failure handling, and weighted
aggregation used by input-specific subclasses. Modality-specific managers include
:class:`~verl_omni.reward_loop.reward_manager.VisualRewardManager` and
:class:`~verl_omni.reward_loop.reward_manager.AudioRewardManager`. They plug
into :class:`~verl_omni.reward_loop.reward_loop.OmniRewardLoopManager` — verl's
:class:`~verl.experimental.reward_loop.RewardLoopManager` extended with
profiler control over the reward-model rollout servers.

.. autosummary::
   :nosignatures:

   verl_omni.reward_loop.reward_loop.OmniRewardLoopManager
   verl_omni.reward_loop.reward_manager.MultiRewardManager
   verl_omni.reward_loop.reward_manager.VisualRewardManager
   verl_omni.reward_loop.reward_manager.AudioRewardManager
   verl_omni.utils.reward_score.default_compute_score_image
   verl_omni.utils.reward_score.http_scorer_client.compute_score
   verl_omni.utils.reward_score.audio_http_scorer_client.compute_score
   verl_omni.utils.reward_score.vlm_as_a_judge.compute_score_unified_reward

Reward Loop Manager
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_loop.OmniRewardLoopManager
   :members: start_profile, stop_profile

Reward Manager
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_manager.MultiRewardManager
   :members: __init__, run_single

.. autoclass:: verl_omni.reward_loop.reward_manager.VisualRewardManager
   :members: __init__, run_single

.. autoclass:: verl_omni.reward_loop.reward_manager.AudioRewardManager
   :members: __init__, run_single

``AudioRewardManager`` reads ``audio`` and ``audio_sample_rate`` from rollout
``extra_info``, validates a finite CPU float waveform, and calls a synchronous
or asynchronous custom scorer with ``solution_audio=(waveform, sample_rate)``.

Default Score Dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: verl_omni.utils.reward_score
   :members: default_compute_score_image

Built-in Reward Scorers
~~~~~~~~~~~~~~~~~~~~~~~~

JPEG Compressibility
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.jpeg_compressibility
   :members: jpeg_compressibility, jpeg_incompressibility, compute_score

VLM-as-a-Judge Scorers
^^^^^^^^^^^^^^^^^^^^^^

The generative-judge scorers live in one module. ``genrm_ocr`` and
``unified_reward`` remain importable aliases for the shared entrypoints.

When ``VisualRewardManager`` calls this repository's
``compute_score_unified_reward`` (the shared module or the ``unified_reward``
compatibility path), a reward rollout ``response_length`` is not forwarded as
``max_tokens``: UnifiedReward keeps its 512-token default, while tuned
``temperature``, ``top_p``, ``top_k`` and ``seed`` are still forwarded. Set an
explicit judge length through the existing reward kwargs, which take precedence:

.. code-block:: bash

   +reward.custom_reward_function.reward_kwargs.sampling_params.max_tokens=1024

Other scorers, including OCR, keep the rollout-length behavior.

.. automodule:: verl_omni.utils.reward_score.vlm_as_a_judge
   :members: compute_score_ocr, compute_score_unified_reward

HTTP Scorer Client
^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.http_scorer_client
   :members: compute_score

Audio HTTP Scorer Client
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.audio_http_scorer_client
   :members: compute_score

Reward Utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.reward_utils
   :members: video_tensor_to_pil_frames, pil_image_to_base64
