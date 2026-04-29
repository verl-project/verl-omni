Utilities
============

Last updated: |today| (API docstrings are auto-generated).

This section documents the small set of cross-cutting utilities provided by
VeRL-Omni. Most other utilities (e.g. tracking, dataset collation) are
inherited from upstream
`verl <https://verl.readthedocs.io/en/latest/api/utils.html>`_ and are
documented there.

File System Utilities
------------------------

.. automodule:: verl_omni.utils.fs
   :members: resolve_model_local_dir

FSDP Utilities
----------------

VeRL-Omni reuses :mod:`verl.utils.fsdp_utils` verbatim and only overrides a
single helper to support layered LoRA collection on diffusers transformer-block
models (e.g. Qwen-Image).

.. autofunction:: verl_omni.utils.fsdp_utils.collect_lora_params

Dataset Utilities
------------------

VeRL-Omni's RLHF dataset class is a thin subclass of
:class:`verl.utils.dataset.rl_dataset.RLHFDataset` that adds an optional
``negative_prompt`` channel for classifier-free guidance. The ``collate_fn``,
``get_dataset_class``, ``create_rl_dataset`` and ``create_rl_sampler``
helpers are re-exported from upstream so callers can keep importing from a
single module.

.. autoclass:: verl_omni.utils.dataset.rl_dataset.RLHFDataset
   :members: __init__, __getitem__

.. autofunction:: verl_omni.utils.dataset.rl_dataset.get_dataset_class

.. autofunction:: verl_omni.utils.dataset.rl_dataset.create_rl_dataset

Diffusion Padding Utilities
----------------------------

See :func:`verl_omni.workers.utils.padding.embeds_padding_2_no_padding` in
the :doc:`workers` section.

vLLM-Omni LoRA Hooks
----------------------

See :class:`verl_omni.utils.vllm_omni.utils.OmniTensorLoRARequest` and
:class:`verl_omni.utils.vllm_omni.utils.VLLMOmniHijack` in the
:doc:`rollout` section.
