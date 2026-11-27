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

"""Exercise review-derived Python rules with legal and illegal source snippets."""

import ast

import pytest
from check_ast_rules import findings

CORE = "verl_omni/trainer/diffusion/example.py"
ADAPTER = "verl_omni/pipelines/example/training_adapter.py"
CONFIG = "verl_omni/workers/config/diffusion/example.py"
MANAGER = "verl_omni/reward_loop/reward_manager/example.py"
TEST = "tests/workers/test_example_on_cpu.py"
SHARED_IMPORT = "from verl_omni.pipelines.model_base import DiffusionModelBase"
ADAPTER_IMPORT = "from verl_omni.pipelines.qwen_image_flow_grpo import QwenImage"
QWEN_ROLLOUT = "verl_omni/pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py"
SDE_HELPER = "verl_omni/pipelines/example/common.py"
REQUEST_SEED = "generator.manual_seed(hash(request.request_id))\n"
MASK_LENGTH = "txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()\n"
SMOKE_PATH = "tests/workers/test_example.py"


@pytest.mark.parametrize(
    "statement",
    [
        ADAPTER_IMPORT,
        "import verl_omni.pipelines.qwen_image_flow_grpo as adapter",
        "from verl_omni.pipelines import qwen_image_flow_grpo as adapter",
        "from ...pipelines.qwen_image_flow_grpo import QwenImage",
        f"def get_adapter():\n    {ADAPTER_IMPORT}\n    return QwenImage",
    ],
)
def test_ast_flags_concrete_pipeline_imports_in_shared_core(statement: str) -> None:
    diagnostics = list(findings(CORE, ast.parse(statement)))
    assert len(diagnostics) == 1
    assert diagnostics[0][1] == "VO001"
    assert "verl_omni.pipelines.qwen_image_flow_grpo" in diagnostics[0][2]


def test_ast_allows_shared_interfaces_and_public_adapter_reuse() -> None:
    assert not list(findings(CORE, ast.parse(f"{SHARED_IMPORT}\n# {ADAPTER_IMPORT}")))
    assert not list(findings("verl_omni/pipelines/sibling/training_adapter.py", ast.parse(ADAPTER_IMPORT)))


@pytest.mark.parametrize(
    ("path", "statement", "expected_codes"),
    [
        ("verl_omni/workers/diffusion_tq.py", ADAPTER_IMPORT, {"VO001"}),
        ("verl_omni/pipelines/rollout_media.py", ADAPTER_IMPORT, {"VO001"}),
        ("verl_omni/pipelines/__init__.py", ADAPTER_IMPORT, set()),
        ("verl_omni/workers/rollout/strategy.py", "from .. import engine", {"VO002"}),
        (
            "verl_omni/pipelines/sibling/vllm_omni_rollout_adapter.py",
            "import verl_omni.workers.engine.fsdp as fsdp",
            {"VO002"},
        ),
        (
            "verl_omni/pipelines/sibling/omni_rollout_adapter.py",
            "from verl_omni.workers import engine",
            {"VO002"},
        ),
        (
            "verl_omni/pipelines/sibling/rollout_kernels.py",
            "from verl_omni.workers.engine import fsdp",
            {"VO002"},
        ),
        (
            "verl_omni/pipelines/sibling/training_adapter.py",
            "from ..qwen_image_flow_grpo.common import _prepare",
            {"VO003"},
        ),
        (
            "verl_omni/pipelines/sibling/training_adapter.py",
            "import verl_omni.pipelines.qwen_image_flow_grpo._private",
            {"VO003"},
        ),
        ("verl_omni/pipelines/sibling/training_adapter.py", "from .common import _prepare", set()),
    ],
)
def test_ast_applies_rules_by_module_role(path: str, statement: str, expected_codes: set[str]) -> None:
    assert {code for _line, code, _message in findings(path, ast.parse(statement))} == expected_codes


@pytest.mark.parametrize(
    ("path", "statement", "expected_code"),
    [
        (SDE_HELPER, "rng = random.Random(hash(request_id))", "VO004"),
        (SDE_HELPER, "rng = random.Random(hash(request_id) % (2**32))", "VO004"),
        (SDE_HELPER, "generator.manual_seed(seed=int(hash(request.req_id)))", "VO004"),
        (SDE_HELPER, REQUEST_SEED, "VO004"),
        (SDE_HELPER, "rng = random.Random(window_seed)", None),
        (SDE_HELPER, "bucket = hash(request_id) % capacity", None),
        (QWEN_ROLLOUT, MASK_LENGTH, "VO005"),
        (QWEN_ROLLOUT, "state.negative_txt_seq_lens = [int(mask.sum()) for mask in masks]", "VO005"),
        (QWEN_ROLLOUT, "txt_seq_lens = [int(m.sum()) for m in prompt_embeds_mask]", "VO005"),
        (QWEN_ROLLOUT, "encode(txt_seq_lens=attention_mask.sum(dim=-1))", "VO005"),
        (QWEN_ROLLOUT, "txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)", None),
        (QWEN_ROLLOUT, "valid_token_count = prompt_embeds_mask.sum(dim=1)", None),
        ("verl_omni/pipelines/example/vllm_omni_rollout_adapter.py", MASK_LENGTH, None),
    ],
)
def test_ast_flags_known_rollout_pitfalls(path: str, statement: str, expected_code: str | None) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(statement))] == (
        [expected_code] if expected_code else []
    )


@pytest.mark.parametrize(
    ("statement", "expected_codes"),
    [
        ("import pytest\npytest.main()", ["VO006"]),
        ("import pytest as runner\nrunner.main()", ["VO006"]),
        ("from pytest import main as run_tests\nrun_tests()", ["VO006"]),
        ("import pytest\nraise SystemExit(pytest.main())", []),
        ("import pytest\nassert pytest.main() == 0", []),
        ("from another_module import main\nmain()", []),
    ],
)
def test_ast_preserves_smoke_exit_status(statement: str, expected_codes: list[str]) -> None:
    assert [code for _line, code, _message in findings(SMOKE_PATH, ast.parse(statement))] == expected_codes


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/workers/test_example_cpu.py", ["VO007"]),
        ("tests/workers/test_example_on_cpu.py", []),
        ("tests/workers/test_example_gpu.py", []),
        ("tests/special_sanity/test_example_cpu.py", []),
    ],
)
def test_checks_cpu_test_filename(path: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(""))] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pytest\npytestmark = pytest.mark.cpu", ["VO007"]),
        ("import pytest as pt\npytestmark = pt.mark.cpu", ["VO007"]),
        ("from pytest import mark as marker\npytestmark = marker.cpu", ["VO007"]),
        ("import pytest\n@pytest.mark.cpu\ndef test_worker():\n    pass", ["VO007"]),
        ("import other\npytestmark = other.mark.cpu", []),
        ('message = "pytest.mark.cpu"', []),
    ],
)
def test_cpu_marker_cannot_hide_a_misnamed_cpu_test(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(SMOKE_PATH, ast.parse(source))] == expected
    assert not list(findings("tests/workers/test_example_on_cpu.py", ast.parse(source)))
    assert not list(findings("tests/special_sanity/test_example.py", ast.parse(source)))


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (CORE, "from verl_omni.utils.tracking import _export_video", ["VO003"]),
        (CORE, "from ...utils.tracking import _export_video as export", ["VO003"]),
        (CORE, "from .utils import _local", []),
        (CORE, "from . import _local", []),
        (CORE, "from verl_omni.utils.tracking import export_video", []),
        (CORE, "from verl_omni.utils.tracking import __all__", []),
        (ADAPTER, "import verl_omni.pipelines.sibling._private as shared", ["VO003"]),
        (ADAPTER, "from ..sibling.common import _prepare", ["VO003"]),
        (ADAPTER, "from .common import _prepare", []),
        (ADAPTER, "from verl_omni.pipelines.sibling.common import prepare", []),
        (ADAPTER, "from external import _prepare", []),
        ("verl_omni/utils/reward_score/desync.py", "from .clap import _get_audio", ["VO003"]),
        ("verl_omni/utils/reward_score/desync.py", "from . import _clap", ["VO003"]),
        ("verl_omni/utils/reward_score/scorers/first.py", "from .second import _Scorer", ["VO003"]),
        ("verl_omni/utils/reward_score/scorers/first.py", "from ..reward_utils import _get_audio", []),
        ("verl_omni/utils/reward_score/desync.py", "from .reward_utils import _get_audio", []),
        ("verl_omni/pipelines/foo/sub/mod.py", "from ..common import _helper", []),
        (TEST, "from verl_omni.utils.tracking import _export_video", []),
    ],
)
def test_private_imports_respect_module_boundaries(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (CORE, 'if model == "qwen_image":\n    run()', ["VO008"]),
        (CORE, 'if config.model in ("bagel", "qwen3-omni"):\n    run()', ["VO008"]),
        (CORE, 'if "MiniMaxH3ImagePipeline" == architecture:\n    run()', ["VO008"]),
        (CORE, 'if model.startswith(("LTX2", "Wan22")):\n    run()', ["VO008"]),
        (CORE, 'if model.endswith("qwen-image"):\n    run()', ["VO008"]),
        (CORE, 'if model == "FluxPipeline":\n    run()', ["VO008"]),
        (CORE, 'match model:\n    case "qwen_image" | "bagel":\n        run()', ["VO008"]),
        (CORE, 'match mode:\n    case "video":\n        run()', []),
        (CORE, 'if mode == "video":\n    run()', []),
        (CORE, 'if model == "squwen_image":\n    run()', []),
        (CORE, 'if mode == "flux_capacitor_enabled":\n    run()', []),
        (CORE, 'if new_model == "unknown_future_model":\n    run()', []),
        (CORE, 'raise ValueError("qwen_image is unsupported")', []),
        (CORE, 'logger.info("qwen_image")\n# model == "qwen_image"', []),
        (CORE, 'if make_message("qwen_image") == message:\n    run()', []),
        (ADAPTER, 'if model == "qwen_image":\n    run()', []),
    ],
)
def test_shared_model_dispatch_uses_known_literal_names(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("class Model(DiffusionModelBase):\n    def __init__(self):\n        pass", ["VO009"]),
        ("class Model(OmniModelBase):\n    def forward(self):\n        pass", ["VO009"]),
        ("class Model(DiffusionModelARBase):\n    async def forward(self):\n        pass", ["VO009"]),
        ("class Model(DiffusionI2IModelBase):\n    @classmethod\n    def forward(cls):\n        pass", []),
        ("class Model(DiffusionModelBase):\n    @staticmethod\n    def forward():\n        pass", []),
        ("class Model(DiffusionModelBase):\n    @property\n    def config(self):\n        return None", []),
        ("class Model(RolloutPipeline):\n    def __init__(self):\n        pass", []),
        (
            "from verl_omni.pipelines.model_base import OmniModelBase as Base\n"
            "class Model(Base):\n    def forward(self):\n        pass",
            ["VO009"],
        ),
        (
            "import verl_omni.pipelines.model_base as bases\n"
            "class Model(bases.OmniModelBase):\n    def forward(self):\n        pass",
            ["VO009"],
        ),
        (
            "from ..model_base import OmniModelBase as Base\nclass Model(Base):\n    def forward(self):\n        pass",
            ["VO009"],
        ),
        (
            "from builtins import staticmethod as static\n"
            "class Model(OmniModelBase):\n    @static\n    def forward():\n        pass",
            [],
        ),
        (
            "class Parent(DiffusionModelBase):\n    pass\nclass Child(Parent):\n    def forward(self):\n        pass",
            ["VO009"],
        ),
        (
            '@DiffusionModelBase.register("Example", algorithm="grpo")\n'
            'class Child(ImportedParent):\n    """Adapter."""\n    def forward(self):\n        pass',
            ["VO009"],
        ),
        (
            '@OmniRolloutPipelineBase.register("example")\n'
            'class RolloutAdapter(OmniRolloutPipelineBase):\n    """Rollout adapter."""\n'
            "    def __init__(self):\n        pass",
            ["VO009"],
        ),
        (
            "from verl_omni.pipelines.non_diffusers_model_base import VllmOmniPipelineBase\n"
            "class RolloutAdapter(VllmOmniPipelineBase):\n    def __call__(self):\n        pass",
            [],
        ),
        (
            "class RolloutAdapter(OmniRolloutPipelineBase):\n"
            '    """Rollout adapter."""\n    @classmethod\n    def build(cls):\n        pass',
            [],
        ),
        (
            "class Model(DiffusionModelBase):\n    @staticmethod\n    def __init__():\n        pass",
            ["VO009"],
        ),
        (
            "class Model(DiffusionModelBase):\n"
            "    def __init_subclass__(cls, **kwargs):\n        super().__init_subclass__(**kwargs)\n"
            "    def __class_getitem__(cls, item):\n        return cls\n",
            [],
        ),
    ],
)
def test_training_adapters_remain_stateless(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(ADAPTER, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("@dataclass\nclass ExampleConfig:\n    pass", ["VO011"]),
        ("@dataclass\nclass ExampleConfig(object):\n    pass", ["VO011"]),
        ("@dataclass(frozen=True)\nclass ExampleConfig(BaseConfig):\n    pass", ["VO011"]),
        ('__all__ = ["ExampleConfig"]\n@dataclass(frozen=False)\nclass ExampleConfig(ExistingConfig):\n    pass', []),
        ("@dataclass\nclass ExampleConfig(BaseConfig):\n    steps: int = 1\n", ["VO011"]),
        ("from dataclasses import dataclass as dc\n@dc\nclass ExampleConfig:\n    pass", ["VO011"]),
        ("import dataclasses as dc\n@dc.dataclass\nclass ExampleConfig:\n    pass", ["VO011"]),
        ("__all__ = []\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass", ["VO011"]),
        ("__all__: list[str] = []\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass", ["VO011"]),
        ('__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass', []),
        ('__all__ = []\n__all__ += ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass', []),
        ('__all__ = ["Other"] + ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass', []),
        (
            '__all__ = ["Other"]\n__all__.append("ExampleConfig")\n'
            "@dataclass\nclass ExampleConfig(BaseConfig):\n    pass",
            [],
        ),
        (
            '__all__ = ["Other"]\n__all__.append("Other2")\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass',
            ["VO011"],
        ),
        ("__all__ = list(parent.__all__)\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass", []),
        ("__all__ = [*parent.__all__]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass", []),
        ("__all__ = []\n__all__ += parent.__all__\n@dataclass\nclass ExampleConfig(BaseConfig):\n    pass", []),
        ("__all__ = []\n@dataclass\nclass _PrivateConfig(BaseConfig):\n    pass", []),
        ("class ExampleConfig:\n    pass", []),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    def __post_init__(self):\n        pass",
            [],
        ),
    ],
)
def test_config_dataclass_contract_has_precise_scope(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(CONFIG, ast.parse(source))] == expected
    assert "VO011" not in [code for _line, code, _message in findings("verl_omni/utils/example.py", ast.parse(source))]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import aiohttp", ["VO012"]),
        ("import requests as client", ["VO012"]),
        ("from httpx import AsyncClient", ["VO012"]),
        ("import urllib", ["VO012"]),
        ("import urllib.request as client", ["VO012"]),
        ("from urllib import request", ["VO012"]),
        ("from urllib.request import urlopen", ["VO012"]),
        ("from urllib.parse import urlparse", []),
        ("import urllib.parse", []),
        ("from urllib import parse", []),
        ("from .media import artifact", []),
    ],
)
def test_reward_network_io_belongs_to_scorers(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(MANAGER, ast.parse(source))] == expected
    assert not list(findings("verl_omni/utils/reward_score/example.py", ast.parse(source)))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('import pytest\nimport torch\npytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")', ["VO013"]),
        ('import pytest as pt\nimport torch as t\npt.mark.skipif(not t.cuda.is_available(), reason="gpu")', ["VO013"]),
        (
            "from pytest import mark\nfrom torch.cuda import is_available as available\n"
            'mark.skipif(condition=not available(), reason="gpu")',
            ["VO013"],
        ),
        (
            "import pytest\nfrom torch import cuda as accelerator\n"
            'pytest.mark.skipif(accelerator.is_available() is False, reason="gpu")',
            ["VO013"],
        ),
        ('import pytest\npytest.mark.skipif(enabled_gpu_count() < 2, reason="two GPUs")', []),
        ('import pytest\npytest.mark.skipif(no_gpu, reason="torch.cuda.is_available()")', []),
        ('import pytest\npytest.mark.skipif(not available(), reason="other dependency")', []),
        ('import pytest\npytest.mark.skipif(not torch.npu.is_available(), reason="npu")', []),
        ('import other\nother.mark.skipif(not torch.cuda.is_available(), reason="gpu")', []),
    ],
)
def test_cuda_skips_use_topology_helpers(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(TEST, ast.parse(source))] == expected
    assert not list(findings("verl_omni/utils/example.py", ast.parse(source)))


def test_tests_conftest_with_cpu_mark_is_not_flagged_and_nested_import_does_not_shadow() -> None:
    assert not list(findings("tests/workers/conftest.py", ast.parse("import pytest\nCPU_MARK = pytest.mark.cpu\n")))
    shadowed = (
        "import pytest\nimport torch\n"
        '@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu")\n'
        "def test_gpu():\n    from unittest.mock import MagicMock as torch\n    return torch\n"
    )
    assert [code for _ln, code, _msg in findings(TEST, ast.parse(shadowed))] == ["VO013"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    steps: int\n', ["VO011"]),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    steps: int = field(metadata={'a': 1})\n",
            ["VO011"],
        ),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    steps: int = field(default=MISSING)\n",
            ["VO011"],
        ),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    steps: int = field(default_factory=MISSING)\n",
            ["VO011"],
        ),
        (
            '__all__ = ["ExampleConfig"]\nfrom omegaconf import MISSING\n@dataclass\n'
            "class ExampleConfig(BaseConfig):\n    path: str = MISSING\n",
            [],
        ),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    mode: Literal['a', 'b'] = 'a'\n",
            ["VO011"],
        ),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n'
            "    mode: \"Literal['a', 'b']\" = \"a\"\n",
            ["VO011"],
        ),
        ('__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    TAG: ClassVar[str]\n', []),
        ('__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    TAG: "ClassVar[str]"\n', []),
        (
            '__all__ = ["ExampleConfig"]\n@dataclass\nclass ExampleConfig(BaseConfig):\n    steps: int = 1\n'
            "    def __post_init__(self):\n        pass\n",
            [],
        ),
        (
            '__all__ = ["SubConfig"]\n@dataclass\nclass SubConfig(ParentConfig):\n'
            "    def __post_init__(self):\n        self.x = 1\n",
            ["VO011"],
        ),
        (
            '__all__ = ["SubConfig"]\n@dataclass\nclass SubConfig(ParentConfig):\n'
            "    def __post_init__(self):\n        super().__post_init__()\n",
            [],
        ),
    ],
)
def test_config_dataclass_rules_cover_defaults_literal_and_super(source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(CONFIG, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("verl_omni/pipelines/rollout_request.py", "def build_request() -> None:\n    return None\n", ["VO015"]),
        (
            "verl_omni/pipelines/rollout_request.py",
            "from typing import overload\n@overload\ndef build(x: int) -> int: ...\n"
            '@overload\ndef build(x: str) -> str: ...\ndef build(x):\n    """Build."""\n    return x\n',
            [],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @property\n    def size(self):\n        return 1\n'
            "    @size.setter\n    def size(self, v):\n        self._v = v\n",
            [],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            "class Request:\n    @property\n    def size(self):\n        return 1\n",
            ["VO015"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    def run(self) -> None:\n        pass\n'
            "    @property\n    def size(self):\n        return 1\n",
            ["VO015"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    def run(self) -> None:\n'
            '        """Run request."""\n        pass\n',
            [],
        ),
        (
            "verl_omni/utils/reward_score/custom.py",
            "def compute_score_custom() -> float:\n    return 0.0\n",
            ["VO015"],
        ),
        (
            "verl_omni/utils/reward_score/custom.py",
            'def compute_score_custom() -> float:\n    """Score one sample."""\n    return 0.0\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["PublicFn"]\ndef PublicFn() -> None:\n    pass\ndef internal_unlisted():\n    pass\n',
            ["VO015"],
        ),
        (
            "verl_omni/pipelines/demo/omni_rollout_adapter.py",
            '@OmniRolloutPipelineBase.register("demo")\nclass DemoRolloutAdapter(OmniRolloutPipelineBase):\n'
            '    supports_async_chunk = True\n    """Misplaced docstring after attribute."""\n    pass\n',
            ["VO015"],
        ),
        (
            "verl_omni/pipelines/demo/omni_rollout_adapter.py",
            '@OmniRolloutPipelineBase.register("demo")\nclass DemoRolloutAdapter(OmniRolloutPipelineBase):\n'
            '    """Rollout adapter."""\n    supports_async_chunk = True\n',
            [],
        ),
        (
            "verl_omni/reward_loop/reward_manager/custom.py",
            '@register_reward_manager("custom")\nclass CustomRewardManager:\n    pass\n',
            ["VO015"],
        ),
        ("verl_omni/utils/custom.py", "def _private_helper():\n    pass\n", []),
    ],
)
def test_public_api_docstrings_have_zero_private_noise(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["scale"]\ndef scale(x):\n    """Scale input."""\n    return x\n',
            ["VO018"],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["scale"]\ndef scale(x: float) -> float:\n    """Scale input."""\n    return x\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["run"]\n'
            'async def run(x: int, /, y, *, flag: bool = True) -> None:\n    """Run."""\n    return None\n',
            ["VO018"],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["run"]\n'
            'async def run(x: int, /, y: str, *, flag: bool = True) -> None:\n    """Run."""\n    return None\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["wrap"]\ndef wrap(x: int, *args, **kwargs) -> int:\n    """Wrap."""\n    return x\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["wrap"]\ndef wrap(x, *args, **kwargs) -> int:\n    """Wrap."""\n    return x\n',
            ["VO018"],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["build"]\nfrom typing import overload\n'
            "@overload\ndef build(x: int) -> int: ...\n"
            "@overload\ndef build(x: str) -> str: ...\n"
            'def build(x):\n    """Build."""\n    return x\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["build"]\nfrom typing import overload\n'
            '@overload\ndef build(x):\n    """Build."""\n    return x\n',
            ["VO018"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'def build_request(x):\n    """Build."""\n    return x\n',
            ["VO018"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    def run(self, x):\n        """Run."""\n        pass\n',
            ["VO018"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    def run(self, x: int) -> None:\n'
            '        """Run."""\n        pass\n',
            [],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @classmethod\n'
            '    def from_cfg(cls, cfg):\n        """Build."""\n        pass\n',
            ["VO018"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @classmethod\n'
            '    def from_cfg(cls, cfg: object) -> None:\n        """Build."""\n        pass\n',
            [],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @staticmethod\n'
            '    def build(cfg) -> None:\n        """Build."""\n        pass\n',
            ["VO018"],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @staticmethod\n'
            '    def build(cfg: object) -> None:\n        """Build."""\n        pass\n',
            [],
        ),
        (
            "verl_omni/pipelines/rollout_request.py",
            'class Request:\n    """Contract."""\n    @property\n    def size(self):\n        return 1\n'
            "    @size.setter\n    def size(self, v):\n        self._v = v\n"
            "    def _helper(self, x):\n        return x\n",
            [],
        ),
        (
            "verl_omni/utils/reward_score/custom.py",
            'def compute_score(data_source, solution_str) -> float:\n    """Score."""\n    return 0.0\n',
            ["VO018"],
        ),
        (
            "verl_omni/utils/reward_score/custom.py",
            'def compute_score_custom(x: str):\n    """Score."""\n    return 0.0\n',
            ["VO018"],
        ),
        (
            "verl_omni/utils/reward_score/custom.py",
            'def compute_score_custom(x: str) -> float:\n    """Score."""\n    return 0.0\n',
            [],
        ),
        (
            "verl_omni/trainer/config/example.py",
            '__all__ = ["build_cfg"]\ndef build_cfg(x):\n    return x\n',
            [],
        ),
        (
            "verl_omni/utils/custom.py",
            '__all__ = ["Public"]\nclass Public:\n    """Public."""\n    pass\n'
            "def unlisted(x):\n    return x\ndef _priv(x):\n    return x\n",
            [],
        ),
    ],
)
def test_public_api_type_annotations_cover_exported_contract_and_scorer_apis(
    path: str, source: str, expected: list[str]
) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("verl_omni/utils/reward_score/ex.py", "try:\n    run()\nexcept Exception:\n    return 0.0\n", ["VO016"]),
        ("verl_omni/utils/reward_score/ex.py", "try:\n    run()\nexcept Exception:\n    return -1.0\n", ["VO016"]),
        (
            "verl_omni/utils/reward_score/ex.py",
            'try:\n    run()\nexcept Exception:\n    return {"score": 0.0, "raw": 0.0}\n',
            ["VO016"],
        ),
        (
            "verl_omni/reward_loop/reward_manager/ex.py",
            "try:\n    run()\nexcept Exception:\n    score = 0.0\n",
            ["VO016"],
        ),
        ("verl_omni/utils/reward_score/ex.py", "try:\n    run()\nexcept:\n    return 0.0\n", ["VO016"]),
        (
            "verl_omni/reward_loop/reward_manager/ex.py",
            "try:\n    run()\nexcept ArtifactContractError:\n    logger.warning('bad')\n",
            ["VO016"],
        ),
        (
            "verl_omni/reward_loop/reward_manager/ex.py",
            "try:\n    run()\nexcept Exception as exc:\n    if required:\n        raise RuntimeError('bad') from exc\n"
            "    score = 0.0\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept ArtifactContractError:\n    raise\nexcept Exception:\n    return 0.0\n",
            [],
        ),
        (
            "verl_omni/reward_loop/reward_manager/ex.py",
            "try:\n    run()\nexcept ArtifactContractError:\n    raise\n"
            "except Exception as exc:\n    if required:\n        raise RuntimeError('bad') from exc\n"
            "    score = 0.0\n",
            [],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception as exc:\n    if isinstance(exc, ArtifactContractError):\n"
            '        raise\n    return {"score": 0.0}\n',
            [],
        ),
        (
            "verl_omni/utils/reward_score/clap.py",
            "for i, item in enumerate(items):\n    try:\n        out = score(item)\n"
            "    except Exception as error:\n        results[i] = error\n",
            [],
        ),
        (
            "verl_omni/reward_loop/native_fsdp.py",
            "try:\n    terminate()\nexcept BaseException as exc:\n    errors.append(exc)\n",
            [],
        ),
        (
            "verl_omni/utils/reward_score/pickscore_reward.py",
            "try:\n    batch_score()\nexcept BaseException as error:\n    future.set_exception(error)\n",
            [],
        ),
        (
            "verl_omni/reward_loop/reward_model_executor.py",
            "try:\n    teardown()\nexcept BaseException:\n    logger.exception('cleanup failed')\n",
            [],
        ),
        ("verl_omni/utils/reward_score/ex.py", "try:\n    run()\nexcept ValueError:\n    return 0.0\n", []),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept:\n    if required:\n        raise\n    score = 0.0\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept ArtifactContractError:\n    if debug:\n        raise\n    return 0.0\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception:\n    if fallback:\n        return 0.0\n    raise\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception as exc:\n"
            "    if not isinstance(exc, ArtifactContractError):\n        raise\n    return 0.0\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception as exc:\n"
            "    if isinstance(exc, ArtifactContractError) and strict:\n        raise\n    return 0.0\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception:\n    score: float = 0.0\n",
            ["VO016"],
        ),
        ("verl_omni/utils/reward_score/ex.py", "try:\n    run()\nexcept Exception:\n    return 0.0, {}\n", ["VO016"]),
        (
            "verl_omni/utils/reward_score/ex.py",
            "try:\n    run()\nexcept Exception:\n    scores.append(0.0)\n",
            ["VO016"],
        ),
        (
            "verl_omni/utils/reward_score/ex.py",
            "def _parse_timeout(raw):\n    try:\n        return int(raw)\n    except Exception:\n        return 30\n",
            [],
        ),
        ("verl_omni/utils/other.py", "try:\n    run()\nexcept Exception:\n    return 0.0\n", []),
    ],
)
def test_artifact_contract_error_must_propagate_in_rewards(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (
            TEST,
            "from pathlib import Path\nADAPTER = Path('verl_omni/pipelines/demo/adapter.py')\n"
            "def test_flag():\n    src = ADAPTER.read_text()\n    assert 'foo' in src\n",
            ["VO017"],
        ),
        (
            TEST,
            "import unittest\nfrom pathlib import Path\nPKG = Path('verl_omni/pipelines/demo')\n"
            "class T(unittest.TestCase):\n    def test_flag(self):\n"
            "        src = (PKG / 'adapter.py').read_text()\n        self.assertIn('foo', src)\n",
            ["VO017"],
        ),
        (
            TEST,
            "class T:\n    def _load(self):\n        return Path('verl_omni/pipelines/demo/a.py').read_text()\n"
            "    def test_flag(self):\n        src = self._load()\n        assert 'foo' in src\n",
            ["VO017"],
        ),
        (
            TEST,
            "def test_open():\n    with open('verl_omni/pipelines/demo/a.py') as f:\n        src = f.read()\n"
            "    assert 'foo' in src\n",
            ["VO017"],
        ),
        (
            TEST,
            "import re\nfrom pathlib import Path\n"
            "def test_regex():\n    src = Path('verl_omni/pipelines/demo/a.py').read_text()\n"
            "    assert re.search('def foo', src)\n",
            ["VO017"],
        ),
        (
            TEST,
            "import ast\nfrom pathlib import Path\n"
            "def test_ast():\n    src = Path('verl_omni/pipelines/demo/a.py').read_text()\n"
            "    assert 'foo' in ast.unparse(ast.parse(src))\n",
            [],
        ),
        (
            TEST,
            "from pathlib import Path\nSCRIPT = Path('examples/demo/run.sh')\n"
            "def test_recipe():\n    src = SCRIPT.read_text()\n"
            "    assert 'verl_omni/utils/reward_score/foo.py' in src\n",
            [],
        ),
        (
            TEST,
            "def test_fixture(tmp_path):\n    p = tmp_path / 'sample.py'\n"
            "    p.write_text('x = 1')\n    assert 'x = 1' in p.read_text()\n",
            [],
        ),
        (
            "tests/special_sanity/test_demo.py",
            "from pathlib import Path\ndef test_ok():\n    assert 'x' in Path('verl_omni/x.py').read_text()\n",
            [],
        ),
    ],
)
def test_tests_do_not_grep_production_source_as_unit_tests(path: str, source: str, expected: list[str]) -> None:
    assert [code for _line, code, _message in findings(path, ast.parse(source))] == expected
