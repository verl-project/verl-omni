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

"""Check Python architecture contracts without importing candidate code."""

import ast
import re
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import PurePosixPath

CORE_DIRECTORIES = ("verl_omni/trainer/", "verl_omni/workers/", "verl_omni/agent_loop/")
SHARED_PIPELINE_MODULES = {
    "diffusion_rollout_output",
    "model_base",
    "non_diffusers_model_base",
    "request_batch",
    "rollout_media",
    "rollout_request",
    "schedulers",
    "utils",
}
TRAINING_BASES = {"DiffusionModelBase", "DiffusionModelARBase", "DiffusionI2IModelBase", "OmniModelBase"}
MODEL_NAME = re.compile(
    r"(?<![a-z0-9])(?:qwen[-_]?image|qwen3[-_]?(?:omni|tts)|minimax[-_]?h3|h3[-_]?image|"
    r"boogu|bagel|wan22|wanpipeline|ltx[-_]?2|minicpm|stable[-_]?diffusion|sd3(?:5|(?=[_\-]|$))|"
    r"flux(?:1|pipeline|[-_]dance[-_]grpo|$)|z[-_]?image)",
    re.IGNORECASE,
)


def _qualified(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else ""
    return ""


def _imports(tree: ast.AST, package: tuple[str, ...]) -> Iterator[tuple[ast.AST, str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name, ""
        elif isinstance(node, ast.ImportFrom) and node.level <= len(package):
            parent = resolve_name("." * node.level + (node.module or ""), ".".join(package))
            for alias in node.names:
                yield node, parent, alias.name


def _aliases(tree: ast.AST, package: tuple[str, ...]) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.level <= len(package):
            parent = resolve_name("." * node.level + (node.module or ""), ".".join(package))
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{parent}.{alias.name}"
    return aliases


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _import_boundaries(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    source_path = PurePosixPath(path)
    package = source_path.parent.parts
    in_pipelines = package[:2] == ("verl_omni", "pipelines")
    pipeline = package[2] if in_pipelines and len(package) > 2 else None
    shared_pipeline = in_pipelines and (pipeline or source_path.stem) in SHARED_PIPELINE_MODULES
    is_core = path.startswith(CORE_DIRECTORIES) or shared_pipeline or path == "verl_omni/utils/diffusion_tq.py"
    is_adapter = pipeline is not None and not shared_pipeline
    is_rollout = path.startswith("verl_omni/workers/rollout/") or (
        is_adapter
        and (
            source_path.name in {"omni_rollout_adapter.py", "vllm_omni_rollout_adapter.py"}
            or source_path.name.startswith("rollout_")
        )
    )
    for node, parent, symbol in _imports(tree, package):
        module = f"{parent}.{symbol}" if symbol else parent
        parts = module.split(".")
        if is_rollout and parts[:3] == ["verl_omni", "workers", "engine"]:
            yield node.lineno, "VO002", f"Use a shared rollout contract instead of importing {module}."
        if (
            is_core
            and len(parts) >= 3
            and parts[:2] == ["verl_omni", "pipelines"]
            and parts[2] not in SHARED_PIPELINE_MODULES
        ):
            yield node.lineno, "VO001", f"Use a registry or shared interface instead of importing {module}."


def _reference_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _mask_loop_vars(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.comprehension) and _reference_name(child.iter).split("_")[-1] == "mask":
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
    return names


def _rollout_findings(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    qwen_image = path.startswith("verl_omni/pipelines/qwen_image_") and path.endswith("/vllm_omni_rollout_adapter.py")
    length_fields = {"txt_seq_lens", "negative_txt_seq_lens"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _reference_name(node.func) in {"Random", "seed", "manual_seed"}:
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            for argument in arguments:
                for call in ast.walk(argument):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "hash"
                        and len(call.args) == 1
                        and _reference_name(call.args[0]) in {"request_id", "req_id"}
                    ):
                        yield call.lineno, "VO004", "Use an explicit sampling seed instead of hashing a request ID."
                        break

        if not qwen_image:
            continue
        if isinstance(node, ast.Assign) and any(_reference_name(target) in length_fields for target in node.targets):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and _reference_name(node.target) in length_fields:
            value = node.value
        elif isinstance(node, ast.keyword) and node.arg in length_fields:
            value = node.value
        else:
            continue
        if value is not None:
            mask_vars = _mask_loop_vars(value)
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "sum"
                and (
                    _reference_name(call.func.value).split("_")[-1] == "mask"
                    or isinstance(call.func.value, ast.Name)
                    and call.func.value.id in mask_vars
                )
                for call in ast.walk(value)
            ):
                yield node.lineno, "VO005", "Use txt_seq_lens_from_embeds() for the padded Qwen-Image RoPE width."


def _cpu_test_names(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    source_path = PurePosixPath(path)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest" and node.level == 0:
            aliases.update({alias.asname or alias.name: f"pytest.{alias.name}" for alias in node.names})
    cpu_marked = any(
        isinstance(node, ast.Attribute)
        and node.attr == "cpu"
        and (
            isinstance(node.value, ast.Name)
            and aliases.get(node.value.id) == "pytest.mark"
            or isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            and isinstance(node.value.value, ast.Name)
            and aliases.get(node.value.value.id) == "pytest"
        )
        for node in ast.walk(tree)
    )
    if (
        not source_path.parts[1].startswith("special_")
        and (cpu_marked or source_path.name.startswith("test_") and source_path.name.endswith("_cpu.py"))
        and not source_path.name.endswith("_on_cpu.py")
    ):
        yield 1, "VO007", "Name CPU tests test_*_on_cpu.py so the L1 workflow collects them."


def _smoke_exit_status(tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    pytest_modules = set()
    pytest_mains = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pytest_modules.update(alias.asname or alias.name for alias in node.names if alias.name == "pytest")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "pytest":
            pytest_mains.update(alias.asname or alias.name for alias in node.names if alias.name == "main")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        direct = isinstance(function, ast.Name) and function.id in pytest_mains
        qualified = (
            isinstance(function, ast.Attribute)
            and function.attr == "main"
            and isinstance(function.value, ast.Name)
            and function.value.id in pytest_modules
        )
        if direct or qualified:
            yield node.lineno, "VO006", "Check the pytest exit status or propagate it with SystemExit."


def _private_imports(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    package = PurePosixPath(path).parent.parts
    for node, parent, symbol in _imports(tree, package):
        parts = tuple(parent.split("."))
        if parts[0] != "verl_omni":
            continue
        full_parts = (*parts, symbol) if symbol else parts
        if not any(_private(part) for part in full_parts[1:]):
            continue
        same_package = parts == package or parts[:-1] == package
        sibling_scorer = (
            package[:3] == parts[:3] == ("verl_omni", "utils", "reward_score")
            and parts != package
            and parts[-1] != "reward_utils"
        )
        if not same_package or sibling_scorer:
            target = ".".join(full_parts)
            yield node.lineno, "VO003", f"Use a public shared helper instead of importing internals: {target}."


def _literals(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.List | ast.Tuple | ast.Set):
        for element in node.elts:
            yield from _literals(element)


def _model_dispatch(tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            line = node.lineno
            values = [node.left, *node.comparators]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"startswith", "endswith"}:
                continue
            line = node.lineno
            values = node.args
        elif isinstance(node, ast.match_case):
            line = node.pattern.lineno
            values = [sub.value for sub in ast.walk(node.pattern) if isinstance(sub, ast.MatchValue)]
        else:
            continue
        if any(MODEL_NAME.search(value) for expression in values for value in _literals(expression)):
            yield line, "VO008", "Use a registry or adapter hook instead of model-name dispatch in shared core."


def _training_adapters(tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    known_bases = TRAINING_BASES | {f"verl_omni.pipelines.model_base.{name}" for name in TRAINING_BASES}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for node in classes:
        registered = any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "register"
            and _qualified(decorator.func.value, aliases) in known_bases
            for decorator in node.decorator_list
        )
        if not registered and not any(_qualified(base, aliases) in known_bases for base in node.bases):
            continue
        known_bases.add(node.name)
        for method in node.body:
            if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            decorators = {_qualified(decorator, aliases) for decorator in method.decorator_list}
            if method.name == "__init__" or not decorators.intersection(
                {
                    "classmethod",
                    "staticmethod",
                    "property",
                    "builtins.classmethod",
                    "builtins.staticmethod",
                    "builtins.property",
                }
            ):
                yield (
                    method.lineno,
                    "VO009",
                    "Training adapters are stateless: use classmethod/staticmethod hooks and no __init__.",
                )


def _config_dataclasses(tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    exported = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            exported = _literal_exports(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            exported = _literal_exports(node.value) if node.value else None
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            additions = _literal_exports(node.value)
            if exported is not None and additions is not None:
                exported.update(additions)
            else:
                exported = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        dataclasses = [
            decorator
            for decorator in node.decorator_list
            if _qualified(decorator.func if isinstance(decorator, ast.Call) else decorator, aliases)
            in {"dataclass", "dataclasses.dataclass"}
        ]
        if not dataclasses:
            continue
        reasons = []
        if not node.bases or all(_qualified(base, aliases) in {"object", "builtins.object"} for base in node.bases):
            reasons.append("inherit BaseConfig or an existing config subclass")
        if any(
            isinstance(decorator, ast.Call)
            and any(
                keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in dataclasses
        ):
            reasons.append("use BaseConfig mutability instead of frozen=True")
        if exported is not None and not node.name.startswith("_") and node.name not in exported:
            reasons.append("export the public config in __all__")
        if reasons:
            yield node.lineno, "VO011", f"{node.name}: {'; '.join(reasons)}."


def _literal_exports(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.List | ast.Tuple) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts
    ):
        return set(_literals(node))
    return None


def _network_imports(tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    for node, parent, symbol in _imports(tree, ()):
        root = parent.split(".")[0]
        urllib_target = f"{parent}.{symbol}" if symbol else parent
        if root in {"aiohttp", "requests", "httpx"} or (
            root == "urllib" and (urllib_target == "urllib" or urllib_target.startswith("urllib.request"))
        ):
            yield node.lineno, "VO012", "Keep network I/O in reward scorers, not reward managers."


def _cuda_skips(tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _qualified(node.func, aliases) != "pytest.mark.skipif":
            continue
        conditions = list(node.args[:1]) + [keyword.value for keyword in node.keywords if keyword.arg == "condition"]
        if any(
            isinstance(check, ast.Call) and _qualified(check.func, aliases) == "torch.cuda.is_available"
            for condition in conditions
            for check in ast.walk(condition)
        ):
            yield node.lineno, "VO013", "Use the GPU topology test helpers instead of bare CUDA availability skips."


def findings(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    """Yield explicit import, dispatch and class-contract violations for one file."""
    aliases = _aliases(tree, PurePosixPath(path).parent.parts)
    if path.startswith("tests/"):
        yield from _cuda_skips(tree, aliases)
        yield from _cpu_test_names(path, tree)
        yield from _smoke_exit_status(tree)
        return
    if not path.startswith("verl_omni/"):
        return
    yield from _private_imports(path, tree)
    if path.startswith(CORE_DIRECTORIES):
        yield from _model_dispatch(tree)
    if path.startswith("verl_omni/pipelines/") and len(PurePosixPath(path).parts) > 3:
        yield from _training_adapters(tree, aliases)
    if path.startswith(("verl_omni/trainer/config/", "verl_omni/workers/config/")):
        yield from _config_dataclasses(tree, aliases)
    if path.startswith("verl_omni/reward_loop/reward_manager/"):
        yield from _network_imports(tree)
    if path.startswith("verl_omni/pipelines/"):
        yield from _rollout_findings(path, tree)
    yield from _import_boundaries(path, tree)
