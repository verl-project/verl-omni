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
TRAINING_BASES = {
    "DiffusionModelBase",
    "DiffusionModelARBase",
    "DiffusionI2IModelBase",
    "OmniModelBase",
    "OmniRolloutPipelineBase",
}
SHARED_CONTRACT_FILES = {
    "verl_omni/pipelines/model_base.py",
    "verl_omni/pipelines/non_diffusers_model_base.py",
    "verl_omni/pipelines/rollout_request.py",
    "verl_omni/pipelines/rollout_media.py",
    "verl_omni/pipelines/request_batch.py",
    "verl_omni/pipelines/diffusion_rollout_output.py",
}
LITERAL_TYPES = {"Literal", "typing.Literal", "typing_extensions.Literal"}
CLASSVAR_TYPES = {"ClassVar", "typing.ClassVar", "typing_extensions.ClassVar"}
ROOT_CONFIG_BASES = {"BaseConfig", "object", "builtins.object"}
OVERLOAD_DECORATORS = {"overload", "typing.overload", "typing_extensions.overload"}
STATICMETHOD_DECORATORS = {"staticmethod", "builtins.staticmethod"}
PROPERTY_DECORATORS = {"property", "builtins.property", "cached_property", "functools.cached_property"}
DOCSTRING_SKIP_DECORATORS = {
    *PROPERTY_DECORATORS,
    *OVERLOAD_DECORATORS,
}
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
    aliases: dict[str, str] = {}

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.level <= len(package):
                parent = resolve_name("." * node.level + (node.module or ""), ".".join(package))
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{parent}.{alias.name}"
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try | ast.TryStar):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.With | ast.AsyncWith):
                visit(node.body)
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    visit(case.body)

    if isinstance(tree, ast.Module):
        visit(tree.body)
    return aliases


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _import_boundaries(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    source_path = PurePosixPath(path)
    package = source_path.parent.parts
    in_pipelines = package[:2] == ("verl_omni", "pipelines")
    pipeline = package[2] if in_pipelines and len(package) > 2 else None
    shared_pipeline = in_pipelines and (pipeline or source_path.stem) in SHARED_PIPELINE_MODULES
    is_core = path.startswith(CORE_DIRECTORIES) or shared_pipeline
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
    is_test_module = source_path.name.startswith("test_") or source_path.name.endswith("_test.py")
    if (
        not source_path.parts[1].startswith("special_")
        and is_test_module
        and (cpu_marked or source_path.name.endswith("_cpu.py"))
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
        is_reward_utils = package[:3] == ("verl_omni", "utils", "reward_score") and (
            parts == ("verl_omni", "utils", "reward_score", "reward_utils")
            or (parts == ("verl_omni", "utils", "reward_score") and symbol == "reward_utils")
        )
        if is_reward_utils:
            continue
        same_pipeline = (
            len(package) >= 3
            and len(parts) >= 3
            and package[:2] == parts[:2] == ("verl_omni", "pipelines")
            and package[2] == parts[2]
            and package[2] not in SHARED_PIPELINE_MODULES
        )
        same_package = parts == package or parts[:-1] == package or same_pipeline
        sibling_scorer = (
            package[:3] == parts[:3] == ("verl_omni", "utils", "reward_score")
            and (parts != package or (bool(symbol) and _private(symbol)))
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
    known_bases = TRAINING_BASES | {
        f"{mod}.{name}"
        for mod in ("verl_omni.pipelines.model_base", "verl_omni.pipelines.non_diffusers_model_base")
        for name in TRAINING_BASES
    }
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
            if method.name in {"__init_subclass__", "__class_getitem__"}:
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
                    "Pipeline adapters are stateless: use classmethod/staticmethod hooks and no __init__.",
                )


def _declares_all(tree: ast.AST) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return True
        if (
            isinstance(node, ast.AnnAssign | ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            return True
    return False


def _static_exports(tree: ast.AST) -> set[str] | None:
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
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "__all__"
            and node.value.func.attr in {"append", "extend"}
        ):
            if node.value.func.attr == "append" and len(node.value.args) == 1 and not node.value.keywords:
                arg = node.value.args[0]
                if exported is not None and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    exported.add(arg.value)
                else:
                    exported = None
            elif node.value.func.attr == "extend" and len(node.value.args) == 1 and not node.value.keywords:
                additions = _literal_exports(node.value.args[0])
                if exported is not None and additions is not None:
                    exported.update(additions)
                else:
                    exported = None
            else:
                exported = None
    return exported


def _parse_annotation(annotation: ast.AST) -> ast.AST:
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return annotation
    return annotation


def _is_classvar(annotation: ast.AST, aliases: dict[str, str]) -> bool:
    parsed = _parse_annotation(annotation)
    target = parsed.value if isinstance(parsed, ast.Subscript) else parsed
    return _qualified(target, aliases) in CLASSVAR_TYPES


def _uses_literal(annotation: ast.AST, aliases: dict[str, str]) -> bool:
    parsed = _parse_annotation(annotation)
    for sub in ast.walk(parsed):
        target = sub.value if isinstance(sub, ast.Subscript) else sub
        if _qualified(target, aliases) in LITERAL_TYPES:
            return True
    return False


def _has_default_value(stmt: ast.AnnAssign, aliases: dict[str, str]) -> bool:
    if stmt.value is None:
        return False
    if _qualified(stmt.value, aliases) in {"MISSING", "dataclasses.MISSING"}:
        return False
    if isinstance(stmt.value, ast.Call) and _qualified(stmt.value.func, aliases) in {"field", "dataclasses.field"}:
        has_default = any(
            keyword.arg == "default" and _qualified(keyword.value, aliases) not in {"MISSING", "dataclasses.MISSING"}
            for keyword in stmt.value.keywords
        )
        has_factory = any(
            keyword.arg == "default_factory"
            and _qualified(keyword.value, aliases) not in {"MISSING", "dataclasses.MISSING"}
            for keyword in stmt.value.keywords
        )
        return has_default or has_factory
    return True


def _calls_super_post_init(method: ast.FunctionDef, aliases: dict[str, str]) -> bool:
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "__post_init__"
        and isinstance(call.func.value, ast.Call)
        and _qualified(call.func.value.func, aliases) in {"super", "builtins.super"}
        for call in ast.walk(method)
    )


def _config_dataclasses(tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    has_all = _declares_all(tree)
    exported = _static_exports(tree)
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
        base_names = [_qualified(base, aliases) for base in node.bases]
        if not base_names or all(name in {"object", "builtins.object"} for name in base_names):
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
        if not node.name.startswith("_") and (not has_all or (exported is not None and node.name not in exported)):
            reasons.append("export the public config in __all__")
        missing_defaults = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and not _is_classvar(stmt.annotation, aliases)
            and not _has_default_value(stmt, aliases)
        ]
        if missing_defaults:
            reasons.append(f"give every field a default or default_factory ({', '.join(missing_defaults)})")
        if any(isinstance(stmt, ast.AnnAssign) and _uses_literal(stmt.annotation, aliases) for stmt in node.body):
            reasons.append("use str with valid_* validation instead of typing.Literal")
        if any(name and name not in ROOT_CONFIG_BASES and not name.endswith(".BaseConfig") for name in base_names):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.FunctionDef)
                    and stmt.name == "__post_init__"
                    and not _calls_super_post_init(stmt, aliases)
                ):
                    reasons.append("call super().__post_init__() in subclass __post_init__")
                    break
        if reasons:
            yield node.lineno, "VO011", f"{node.name}: {'; '.join(reasons)}."


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    return bool(ast.get_docstring(node))


def _is_property_accessor(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}


def _has_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str], names: set[str]) -> bool:
    return any(_qualified(decorator, aliases) in names for decorator in node.decorator_list)


def _skip_method_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]) -> bool:
    if node.name.startswith("_"):
        return True
    return _has_decorator(node, aliases, DOCSTRING_SKIP_DECORATORS) or any(
        _is_property_accessor(decorator) for decorator in node.decorator_list
    )


def _is_registered_class(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Attribute) and (func.attr == "register" or func.attr.startswith("register_")):
            return True
        if isinstance(func, ast.Name) and func.id.startswith("register_"):
            return True
    return False


def _public_api_scope(path: str, tree: ast.AST) -> tuple[set[str] | None, bool, bool]:
    in_config = path.startswith(("verl_omni/trainer/config/", "verl_omni/workers/config/"))
    exported = None if in_config else _static_exports(tree)
    return (
        exported,
        path in SHARED_CONTRACT_FILES,
        path.startswith("verl_omni/utils/reward_score/"),
    )


def _is_public_api_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    exported: set[str] | None,
    *,
    in_contract: bool,
    in_reward_score: bool,
) -> bool:
    is_exported = exported is not None and not node.name.startswith("_") and node.name in exported
    is_contract_symbol = in_contract and not node.name.startswith("_")
    is_scorer = in_reward_score and (node.name == "compute_score" or node.name.startswith("compute_score_"))
    return is_exported or is_contract_symbol or is_scorer


def _overloaded_names(body: list[ast.stmt], aliases: dict[str, str]) -> set[str]:
    return {
        node.name
        for node in body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and _has_decorator(node, aliases, OVERLOAD_DECORATORS)
    }


def _missing_public_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    is_method: bool,
    is_static: bool,
) -> bool:
    args = list(node.args.posonlyargs) + list(node.args.args)
    if is_method and not is_static and args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    args.extend(node.args.kwonlyargs)
    return node.returns is None or any(arg.annotation is None for arg in args)


def _contract_method_annotations(cls: ast.ClassDef, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    overloaded = _overloaded_names(cls.body, aliases)
    for member in cls.body:
        if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef) or member.name.startswith("_"):
            continue
        if _has_decorator(member, aliases, PROPERTY_DECORATORS) or any(
            _is_property_accessor(decorator) for decorator in member.decorator_list
        ):
            continue
        if not _has_decorator(member, aliases, OVERLOAD_DECORATORS) and member.name in overloaded:
            continue
        is_static = _has_decorator(member, aliases, STATICMETHOD_DECORATORS)
        if _missing_public_annotations(member, is_method=True, is_static=is_static):
            yield (
                member.lineno,
                "VO018",
                f"Annotate shared contract method '{cls.name}.{member.name}' parameters and return type.",
            )


def _public_api_annotations(path: str, tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    exported, in_contract, in_reward_score = _public_api_scope(path, tree)
    body = tree.body if isinstance(tree, ast.Module) else []
    overloaded = _overloaded_names(body, aliases)
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not _has_decorator(node, aliases, OVERLOAD_DECORATORS) and node.name in overloaded:
                continue
            if _is_public_api_function(
                node, exported, in_contract=in_contract, in_reward_score=in_reward_score
            ) and _missing_public_annotations(node, is_method=False, is_static=False):
                yield node.lineno, "VO018", f"Annotate public API '{node.name}' parameters and return type."
        elif in_contract and isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield from _contract_method_annotations(node, aliases)


def _public_api_docstrings(path: str, tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    exported, in_contract, in_reward_score = _public_api_scope(path, tree)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        is_overload = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _has_decorator(
            node, aliases, OVERLOAD_DECORATORS
        )
        is_exported = exported is not None and not node.name.startswith("_") and node.name in exported
        is_contract_symbol = in_contract and not node.name.startswith("_")
        is_scorer = (
            in_reward_score
            and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and (node.name == "compute_score" or node.name.startswith("compute_score_"))
        )
        is_registered = isinstance(node, ast.ClassDef) and not node.name.startswith("_") and _is_registered_class(node)
        needs_doc = is_exported or is_contract_symbol or is_scorer or is_registered
        if not is_overload and needs_doc and not _has_docstring(node):
            yield node.lineno, "VO015", f"Document public API '{node.name}' with a concise contract docstring."
        if in_contract and isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for member in node.body:
                if (
                    isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
                    and not _skip_method_docstring(member, aliases)
                    and not _has_docstring(member)
                ):
                    yield (
                        member.lineno,
                        "VO015",
                        f"Document shared contract method '{node.name}.{member.name}' with a concise docstring.",
                    )


def _caught_exception_names(exc_type: ast.AST, aliases: dict[str, str]) -> list[str]:
    if isinstance(exc_type, ast.Tuple):
        names: list[str] = []
        for element in exc_type.elts:
            names.extend(_caught_exception_names(element, aliases))
        return names
    qualified = _qualified(exc_type, aliases)
    return [qualified] if qualified else []


def _is_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return True
    return (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd) and _is_numeric_literal(node.operand)
    )


def _is_dict_score_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        return any(isinstance(key, ast.Constant) and key.value == "score" for key in node.keys)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and any(keyword.arg == "score" for keyword in node.keywords)
    )


def _node_contains_fallback_score(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Return) and sub.value is not None:
            if _is_numeric_literal(sub.value) or _is_dict_score_expr(sub.value):
                return True
            if isinstance(sub.value, ast.Tuple) and any(_is_numeric_literal(el) for el in sub.value.elts):
                return True
        elif isinstance(sub, ast.Assign) and sub.value is not None:
            if _is_numeric_literal(sub.value) and any(
                isinstance(target, ast.Name) and target.id == "score" for target in sub.targets
            ):
                return True
            if _is_dict_score_expr(sub.value):
                return True
        elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
            if _is_numeric_literal(sub.value) and isinstance(sub.target, ast.Name) and sub.target.id == "score":
                return True
            if _is_dict_score_expr(sub.value):
                return True
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "append"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "scores"
            and sub.args
            and _is_numeric_literal(sub.args[0])
        ):
            return True
    return False


def _returns_or_sets_fallback_score(body: list[ast.stmt]) -> bool:
    return _node_contains_fallback_score(ast.Module(body=body, type_ignores=[]))


def _unconditionally_raises_before_fallback(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if _node_contains_fallback_score(stmt):
            return False
        if isinstance(stmt, ast.Raise):
            return True
    return False


def _is_positive_contract_check(test: ast.AST, exc_name: str, aliases: dict[str, str]) -> bool:
    if isinstance(test, ast.Call):
        return (
            _qualified(test.func, aliases) in {"isinstance", "builtins.isinstance"}
            and len(test.args) >= 2
            and isinstance(test.args[0], ast.Name)
            and test.args[0].id == exc_name
            and any(
                name.split(".")[-1] == "ArtifactContractError"
                for name in _caught_exception_names(test.args[1], aliases)
            )
        )
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        return any(_is_positive_contract_check(value, exc_name, aliases) for value in test.values)
    return False


def _reraises_contract_in_body(handler: ast.ExceptHandler, aliases: dict[str, str]) -> bool:
    if not handler.name:
        return False
    for stmt in handler.body:
        if _node_contains_fallback_score(stmt):
            return False
        if (
            isinstance(stmt, ast.If)
            and _is_positive_contract_check(stmt.test, handler.name, aliases)
            and _unconditionally_raises_before_fallback(stmt.body)
        ):
            return True
    return False


def _is_trivial_scalar_parse_try(try_node: ast.Try | ast.TryStar) -> bool:
    if len(try_node.body) != 1 or try_node.orelse:
        return False
    sole = try_node.body[0]
    call = None
    if isinstance(sole, ast.Return):
        call = sole.value
    elif isinstance(sole, ast.Assign):
        call = sole.value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"int", "float", "bool", "complex"}
    )


def _artifact_contract_errors(tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try | ast.TryStar)]
    for try_node in try_nodes:
        handled_contract = False
        trivial_scalar = _is_trivial_scalar_parse_try(try_node)
        for handler in try_node.handlers:
            unconditional_raise = _unconditionally_raises_before_fallback(handler.body)
            if handler.type is None:
                if not handled_contract and not unconditional_raise and _returns_or_sets_fallback_score(handler.body):
                    yield handler.lineno, "VO016", "Propagate ArtifactContractError before bare except score fallback."
                continue
            caught = _caught_exception_names(handler.type, aliases)
            if any(name.split(".")[-1] == "ArtifactContractError" for name in caught):
                if unconditional_raise:
                    handled_contract = True
                else:
                    yield handler.lineno, "VO016", "Propagate ArtifactContractError instead of swallowing it."
            elif (
                not trivial_scalar
                and any(
                    name in {"Exception", "BaseException", "builtins.Exception", "builtins.BaseException"}
                    for name in caught
                )
                and not handled_contract
                and not unconditional_raise
                and not _reraises_contract_in_body(handler, aliases)
                and _returns_or_sets_fallback_score(handler.body)
            ):
                yield handler.lineno, "VO016", "Propagate ArtifactContractError before fallback score handling."


def _literal_exports(node: ast.AST) -> set[str] | None:
    if isinstance(node, ast.List | ast.Tuple) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts
    ):
        return set(_literals(node))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_exports(node.left)
        right = _literal_exports(node.right)
        if left is not None and right is not None:
            return left | right
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


def _path_strings(node: ast.AST) -> list[str]:
    return [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Constant) and isinstance(sub.value, str)]


def _path_var_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _has_examples_string(strings: list[str]) -> bool:
    return any(item == "examples" or "examples/" in item for item in strings)


def _has_verl_omni_ref(node: ast.AST, strings: list[str], dir_vars: set[str]) -> bool:
    has_direct = any(item == "verl_omni" or "verl_omni/" in item or item.startswith("verl_omni") for item in strings)
    return has_direct or bool(_path_var_names(node) & dir_vars)


def _expr_is_verl_omni_dir(node: ast.AST, dir_vars: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in dir_vars:
        return True
    strings = _path_strings(node)
    if not strings or _has_examples_string(strings) or any(item.endswith(".py") for item in strings):
        return False
    return _has_verl_omni_ref(node, strings, dir_vars)


def _expr_is_verl_omni_py(node: ast.AST, py_path_vars: set[str], dir_vars: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in py_path_vars:
        return True
    strings = _path_strings(node)
    if not strings or _has_examples_string(strings):
        return False
    if not any(item.endswith(".py") for item in strings):
        return False
    return _has_verl_omni_ref(node, strings, dir_vars)


def _bound_open_path(call: ast.Call, py_path_vars: set[str], dir_vars: set[str], aliases: dict[str, str]) -> bool:
    if _qualified(call.func, aliases) in {"open", "builtins.open"} and call.args:
        return _expr_is_verl_omni_py(call.args[0], py_path_vars, dir_vars)
    if isinstance(call.func, ast.Attribute) and call.func.attr == "open":
        return _expr_is_verl_omni_py(call.func.value, py_path_vars, dir_vars)
    return False


def _is_source_read_call(
    node: ast.AST,
    py_path_vars: set[str],
    dir_vars: set[str],
    file_vars: set[str],
    helper_names: set[str],
    aliases: dict[str, str],
) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == "read_text":
        return _expr_is_verl_omni_py(node.func.value, py_path_vars, dir_vars)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "read":
        target = node.func.value
        if isinstance(target, ast.Name) and target.id in file_vars:
            return True
        if isinstance(target, ast.Call):
            return _bound_open_path(target, py_path_vars, dir_vars, aliases)
    if isinstance(node.func, ast.Name) and node.func.id in helper_names:
        return True
    return isinstance(node.func, ast.Attribute) and node.func.attr in helper_names


def _expr_references_source(
    expr: ast.AST,
    local_py: set[str],
    local_dirs: set[str],
    file_vars: set[str],
    src_vars: set[str],
    helper_names: set[str],
    aliases: dict[str, str],
) -> bool:
    if isinstance(expr, ast.Name) and expr.id in src_vars:
        return True
    if _is_source_read_call(expr, local_py, local_dirs, file_vars, helper_names, aliases):
        return True
    if isinstance(expr, ast.Subscript):
        return _expr_references_source(expr.value, local_py, local_dirs, file_vars, src_vars, helper_names, aliases)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return _expr_references_source(
            expr.func.value, local_py, local_dirs, file_vars, src_vars, helper_names, aliases
        )
    return False


def _collect_fn_source_state(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    base_py_vars: set[str],
    base_dir_vars: set[str],
    helper_names: set[str],
    aliases: dict[str, str],
) -> tuple[set[str], set[str], set[str], set[str]]:
    local_py = set(base_py_vars)
    local_dirs = set(base_dir_vars)
    file_vars: set[str] = set()
    src_vars: set[str] = set()

    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.With | ast.AsyncWith):
            for item in stmt.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.optional_vars, ast.Name)
                    and _bound_open_path(item.context_expr, local_py, local_dirs, aliases)
                ):
                    file_vars.add(item.optional_vars.id)
        elif isinstance(stmt, ast.Assign):
            target_names = [target.id for target in stmt.targets if isinstance(target, ast.Name)]
            if _expr_is_verl_omni_py(stmt.value, local_py, local_dirs):
                local_py.update(target_names)
            elif _expr_is_verl_omni_dir(stmt.value, local_dirs):
                local_dirs.update(target_names)
            if _expr_references_source(stmt.value, local_py, local_dirs, file_vars, src_vars, helper_names, aliases):
                src_vars.update(target_names)
    return local_py, local_dirs, file_vars, src_vars


def _test_has_source_grep(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    local_py: set[str],
    local_dirs: set[str],
    file_vars: set[str],
    src_vars: set[str],
    helper_names: set[str],
    aliases: dict[str, str],
) -> bool:
    def _matches(expr: ast.AST | None) -> bool:
        return expr is not None and _expr_references_source(
            expr, local_py, local_dirs, file_vars, src_vars, helper_names, aliases
        )

    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Compare) and any(isinstance(op, ast.In | ast.NotIn) for op in sub.ops):
                    if any(_matches(comp) for comp in sub.comparators):
                        return True
                elif (
                    isinstance(sub, ast.Call)
                    and _qualified(sub.func, aliases) in {"re.search", "re.match", "re.fullmatch", "re.findall"}
                    and (
                        any(_matches(arg) for arg in sub.args[1:2])
                        or any(kw.arg == "string" and _matches(kw.value) for kw in sub.keywords)
                    )
                ):
                    return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"assertIn", "assertNotIn"}
        ):
            container = (
                node.args[1]
                if len(node.args) >= 2
                else next((kw.value for kw in node.keywords if kw.arg == "container"), None)
            )
            if _matches(container):
                return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"assertRegex", "assertNotRegex"}
        ):
            text_arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "text"), None)
            if _matches(text_arg):
                return True
    return False


def _source_grep_tests(path: str, tree: ast.AST, aliases: dict[str, str]) -> Iterator[tuple[int, str, str]]:
    if path.startswith("tests/special_sanity/"):
        return
    if any(isinstance(node, ast.Call) and _qualified(node.func, aliases) == "ast.parse" for node in ast.walk(tree)):
        return
    py_path_vars: set[str] = set()
    dir_vars: set[str] = set()
    for stmt in tree.body if isinstance(tree, ast.Module) else ():
        if isinstance(stmt, ast.Assign):
            targets = [target.id for target in stmt.targets if isinstance(target, ast.Name)]
            if _expr_is_verl_omni_py(stmt.value, py_path_vars, dir_vars):
                py_path_vars.update(targets)
            elif _expr_is_verl_omni_dir(stmt.value, dir_vars):
                dir_vars.update(targets)

    all_funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    helper_names: set[str] = set()
    for fn in all_funcs:
        if fn.name.startswith("test_"):
            continue
        local_py, local_dirs, file_vars, src_vars = _collect_fn_source_state(
            fn, py_path_vars, dir_vars, helper_names, aliases
        )
        if any(
            isinstance(ret, ast.Return)
            and ret.value is not None
            and _expr_references_source(ret.value, local_py, local_dirs, file_vars, src_vars, helper_names, aliases)
            for ret in ast.walk(fn)
        ):
            helper_names.add(fn.name)

    for fn in all_funcs:
        if not fn.name.startswith("test_"):
            continue
        local_py, local_dirs, file_vars, src_vars = _collect_fn_source_state(
            fn, py_path_vars, dir_vars, helper_names, aliases
        )
        if not src_vars and not any(
            _is_source_read_call(sub, local_py, local_dirs, file_vars, helper_names, aliases) for sub in ast.walk(fn)
        ):
            continue
        if _test_has_source_grep(fn, local_py, local_dirs, file_vars, src_vars, helper_names, aliases):
            yield (
                fn.lineno,
                "VO017",
                "Test behavior directly instead of asserting substrings on verl_omni source text.",
            )


def findings(path: str, tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    """Yield explicit import, dispatch and class-contract violations for one file."""
    aliases = _aliases(tree, PurePosixPath(path).parent.parts)
    if path.startswith("tests/"):
        yield from _cuda_skips(tree, aliases)
        yield from _cpu_test_names(path, tree)
        yield from _smoke_exit_status(tree)
        yield from _source_grep_tests(path, tree, aliases)
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
    if path.startswith(("verl_omni/utils/reward_score/", "verl_omni/reward_loop/")):
        yield from _artifact_contract_errors(tree, aliases)
    if path.startswith("verl_omni/pipelines/"):
        yield from _rollout_findings(path, tree)
    yield from _public_api_docstrings(path, tree, aliases)
    yield from _public_api_annotations(path, tree, aliases)
    yield from _import_boundaries(path, tree)
