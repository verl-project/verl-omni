Utilities
============

Last updated: 04/28/2026 (API docstrings are auto-generated).

This section documents the small set of cross-cutting utilities provided by
VeRL-Omni. Most other utilities (e.g. tracking, FSDP helpers, dataset
collation) are inherited from upstream
`verl <https://verl.readthedocs.io/en/latest/api/utils.html>`_ and are
documented there.

File System Utilities
------------------------

.. automodule:: verl_omni.utils.fs
   :members: resolve_model_local_dir

Diffusion Padding Utilities
----------------------------

See :func:`verl_omni.workers.utils.padding.embeds_padding_2_no_padding` in
the :doc:`workers` section.

vLLM-Omni LoRA Hooks
----------------------

See :class:`verl_omni.utils.vllm_omni.utils.OmniTensorLoRARequest` and
:class:`verl_omni.utils.vllm_omni.utils.VLLMOmniHijack` in the
:doc:`rollout` section.
