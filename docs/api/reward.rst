Reward Interface
================================

Last updated: |today| (API docstrings are auto-generated).

VeRL-Omni reward pipelines support both rule-based scoring (e.g. JPEG
compressibility) and model-based generative reward models (e.g. OCR via a
vision-language model served behind an OpenAI-compatible router). Reward
computation is dispatched per sample by the
:class:`~verl_omni.reward_loop.reward_manager.VisualRewardManager`, which
plugs into the standard :class:`verl.experimental.reward_loop.RewardLoopManager`.

.. autosummary::
   :nosignatures:

   verl_omni.reward_loop.reward_manager.VisualRewardManager
   verl_omni.utils.reward_score.default_compute_score_image
   verl_omni.utils.reward_score.http_scorer_client.compute_score
   verl_omni.utils.reward_score.unified_reward.compute_score_unified_reward

Reward Manager
~~~~~~~~~~~~~~~~~

.. autoclass:: verl_omni.reward_loop.reward_manager.VisualRewardManager
   :members: __init__, run_single

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

GRM-based OCR Reward
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.genrm_ocr
   :members: compute_score_ocr

HTTP Scorer Client
^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.http_scorer_client
   :members: compute_score

UnifiedReward Scorer
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.unified_reward
   :members: compute_score_unified_reward

Reward Utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: verl_omni.utils.reward_score.reward_utils
   :members: pil_image_to_base64
