# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Fallback for CPU-only environments where vLLM-Omni current_omni_platform.device_type is empty.
# This prevents RuntimeError: Device string must not be empty when importing modules with torch.amp.autocast.
# TODO: Remove when vllm-omni initializes device_type on CPU platforms by default.
try:
    import vllm_omni.platforms

    if not vllm_omni.platforms.current_omni_platform.device_type:
        vllm_omni.platforms.current_omni_platform.device_type = "cpu"
except Exception:
    pass


# TODO: Remove this FusedMoE compatibility patch after verl is upgraded to 0.9.0
# (which natively supports vLLM 0.27 MoE modular refactor).
# vLLM 0.27 compatibility: alias FusedMoE in vllm.model_executor.layers.fused_moe.layer for verl
try:
    import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

    if not hasattr(_fused_moe_layer, "FusedMoE"):
        import vllm.model_executor.layers.fused_moe as _fused_moe

        _fused_moe_layer.FusedMoE = getattr(
            _fused_moe,
            "FusedMoEExpertsModular",
            getattr(_fused_moe, "FusedMoEMethodBase", object),
        )

    # Persist _vllm_moe_compat.py + .pth into site-packages so Ray worker subprocesses also inherit the alias
    try:
        import site

        _compat_py = (
            "import importlib.util, sys\n"
            "from importlib.abc import Loader, MetaPathFinder\n"
            "\n"
            "class _MoEHookFinder(MetaPathFinder):\n"
            "    def find_spec(self, fullname, path, target=None):\n"
            "        if fullname == 'vllm.model_executor.layers.fused_moe.layer':\n"
            "            sys.meta_path.remove(self)\n"
            "            try:\n"
            "                spec = importlib.util.find_spec(fullname)\n"
            "                if spec and spec.loader:\n"
            "                    orig_loader = spec.loader\n"
            "                    class _HookedLoader(Loader):\n"
            "                        def create_module(self, spec):\n"
            "                            return orig_loader.create_module(spec)\n"
            "                        def exec_module(self, module):\n"
            "                            orig_loader.exec_module(module)\n"
            "                            if not hasattr(module, 'FusedMoE'):\n"
            "                                import vllm.model_executor.layers.fused_moe as _m\n"
            "                                _target = getattr(_m, 'FusedMoEExpertsModular', None)\n"
            "                                module.FusedMoE = _target or getattr(_m, 'FusedMoEMethodBase', object)\n"
            "                    spec.loader = _HookedLoader()\n"
            "                return spec\n"
            "            finally:\n"
            "                sys.meta_path.insert(0, self)\n"
            "        return None\n"
            "\n"
            "sys.meta_path.insert(0, _MoEHookFinder())\n"
        )
        for _site_dir in set(site.getsitepackages()):
            if os.path.exists(_site_dir):
                _py_path = os.path.join(_site_dir, "_vllm_moe_compat.py")
                _pth_path = os.path.join(_site_dir, "verl_omni_vllm_compat.pth")
                with open(_py_path, "w") as _f:
                    _f.write(_compat_py)
                if not os.path.exists(_pth_path):
                    with open(_pth_path, "w") as _f:
                        _f.write("import _vllm_moe_compat\n")
    except Exception:
        pass
except Exception:
    pass


# Import pipelines / rollout / reward loop / engines to auto-register them
# Apply model patches and auto-register pipelines / rollout / reward loop / engines
import verl_omni.models  # noqa: E402, F401
import verl_omni.pipelines  # noqa: E402, F401
import verl_omni.reward_loop  # noqa: E402, F401
import verl_omni.trainer  # noqa: E402, F401
import verl_omni.workers.engine  # noqa: E402, F401
import verl_omni.workers.rollout  # noqa: E402, F401
