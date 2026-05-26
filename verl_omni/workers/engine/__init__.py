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
from .fsdp import (  # noqa: F401
    DiffusersFSDPEngine,
    DPODiffusersFSDPEngine,
    NFTDiffusersFSDPEngine,
    PPODiffusersFSDPEngine,
)

try:
    from .veomni import VeOmniDiffusionEngine  # noqa: F401
except ImportError as _veomni_import_err:
    import warnings

    warnings.warn(
        f"verl_omni.workers.engine.veomni failed to import; "
        f"`backend=veomni` will be unavailable. Root cause: "
        f"{type(_veomni_import_err).__name__}: {_veomni_import_err}",
        RuntimeWarning,
        stacklevel=2,
    )
    VeOmniDiffusionEngine = None

__all__ = [
    "PPODiffusersFSDPEngine",
    "DPODiffusersFSDPEngine",
    "NFTDiffusersFSDPEngine",
    "DiffusersFSDPEngine",
    "VeOmniDiffusionEngine",
]
