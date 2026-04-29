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
