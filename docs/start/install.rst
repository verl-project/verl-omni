.. _installation:

Installation
============

Requirements
------------

.. list-table::
   :widths: 20 30
   :header-rows: 1

   * - Dependency
     - Version
   * - Python
     - >= 3.10
   * - CUDA
     - >= 12.1
   * - GPU
     - NVIDIA GPU (≥ 24 GB VRAM recommended)

Install from Source
-------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/verl-project/verl-omni.git
   cd verl-omni
   pip install -e .

vLLM-Omni Rollout Backend
--------------------------

verl-omni uses `vLLM-Omni <https://github.com/vllm-project/vllm-omni>`_ for multimodal rollout.
Install it separately:

.. code-block:: bash

   pip install "vllm==0.18" "vllm-omni==0.18"

.. note::
   Inference frameworks may override your existing PyTorch installation.
   Install vLLM-Omni **before** other packages, or verify compatibility afterwards.

Post-Installation Verification
-------------------------------

.. code-block:: bash

   python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
   python -c "import vllm; print('vllm', vllm.__version__)"
   python -c "import verl; print('verl-omni ready')"
