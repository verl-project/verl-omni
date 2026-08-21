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
"""Regression tests for the prefix-caching CLI fix.

verl's ``build_cli_args_from_config`` drops explicit-``False`` booleans, so
``rollout.enable_prefix_caching=False`` never reaches the omni engine and the
scheduler silently runs with prefix caching ON — which corrupts rollouts after
sleep/wake cycles (stale block-hash table over discarded KV). These tests pin
the shim in
``verl_omni.workers.rollout.vllm_rollout.prefix_caching`` that re-emits
``--no-enable-prefix-caching`` for an explicit ``False``.
"""

import logging
import types

import pytest

from verl_omni.workers.rollout.vllm_rollout.prefix_caching import (
    _PATCH_MARK,
    install_prefix_caching_cli_fix,
    warn_if_prefix_caching_mismatch,
)


def make_fake_verl_server_module() -> types.ModuleType:
    """Build a module whose serializer replicates verl's bool-dropping behavior."""
    mod = types.ModuleType("fake_verl_vllm_async_server")

    def build_cli_args_from_config(config: dict) -> list[str]:
        # Simplified clone of verl's serializer: bool False emits nothing.
        cli_args = []
        for k, v in config.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    cli_args.append(f"--{k}")
            else:
                cli_args.append(f"--{k}")
                cli_args.append(str(v))
        return cli_args

    mod.build_cli_args_from_config = build_cli_args_from_config
    return mod


class TestInstallPrefixCachingCliFix:
    def test_explicit_false_emits_no_flag(self):
        mod = make_fake_verl_server_module()
        install_prefix_caching_cli_fix(mod)
        args = mod.build_cli_args_from_config(
            {"enable_prefix_caching": False, "max_num_seqs": 128, "enforce_eager": True}
        )
        assert "--no-enable-prefix-caching" in args
        assert "--enable-prefix-caching" not in args
        # Other keys keep verl's native serialization (underscore style).
        assert "--max_num_seqs" in args and "128" in args
        assert "--enforce_eager" in args

    def test_explicit_true_keeps_verl_behavior(self):
        mod = make_fake_verl_server_module()
        install_prefix_caching_cli_fix(mod)
        args = mod.build_cli_args_from_config({"enable_prefix_caching": True})
        assert args == ["--enable_prefix_caching"]
        assert "--no-enable-prefix-caching" not in args

    def test_absent_or_none_keeps_verl_behavior(self):
        mod = make_fake_verl_server_module()
        install_prefix_caching_cli_fix(mod)
        assert mod.build_cli_args_from_config({}) == []
        assert mod.build_cli_args_from_config({"enable_prefix_caching": None}) == []

    def test_caller_config_not_mutated(self):
        mod = make_fake_verl_server_module()
        install_prefix_caching_cli_fix(mod)
        config = {"enable_prefix_caching": False}
        mod.build_cli_args_from_config(config)
        assert config == {"enable_prefix_caching": False}

    def test_install_is_idempotent(self):
        mod = make_fake_verl_server_module()
        install_prefix_caching_cli_fix(mod)
        patched_once = mod.build_cli_args_from_config
        install_prefix_caching_cli_fix(mod)
        assert mod.build_cli_args_from_config is patched_once
        # Still exactly one negative flag after repeated installs.
        args = mod.build_cli_args_from_config({"enable_prefix_caching": False})
        assert args.count("--no-enable-prefix-caching") == 1

    def test_install_raises_when_serializer_missing(self):
        mod = types.ModuleType("empty_module")
        with pytest.raises(AttributeError, match="build_cli_args_from_config"):
            install_prefix_caching_cli_fix(mod)

    def test_upstream_fixed_serializer_left_untouched(self):
        # If verl ever ships an explicit-False-aware serializer (marked with
        # the same attribute), install must be a no-op that keeps it in place.
        mod = make_fake_verl_server_module()

        def fixed_serializer(config: dict) -> list[str]:
            return ["--no-enable-prefix-caching"] if config.get("enable_prefix_caching") is False else []

        setattr(fixed_serializer, _PATCH_MARK, True)
        mod.build_cli_args_from_config = fixed_serializer
        install_prefix_caching_cli_fix(mod)
        assert mod.build_cli_args_from_config is fixed_serializer

    def test_install_on_real_verl_module(self):
        """Integration: patch the actual verl module, then restore it."""
        pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")
        import verl.workers.rollout.vllm_rollout.vllm_async_server as verl_server_mod

        orig = verl_server_mod.build_cli_args_from_config
        try:
            install_prefix_caching_cli_fix(verl_server_mod)
            patched = verl_server_mod.build_cli_args_from_config
            assert getattr(patched, _PATCH_MARK, False)
            args = patched({"enable_prefix_caching": False})
            assert "--no-enable-prefix-caching" in args
            assert "--enable-prefix-caching" not in args
        finally:
            verl_server_mod.build_cli_args_from_config = orig


class TestWarnIfPrefixCachingMismatch:
    @staticmethod
    def _fake_engine(effective: bool):
        cache_config = types.SimpleNamespace(enable_prefix_caching=effective)
        return types.SimpleNamespace(vllm_config=types.SimpleNamespace(cache_config=cache_config))

    def test_mismatch_warns(self, caplog):
        config = types.SimpleNamespace(enable_prefix_caching=False)
        with caplog.at_level(logging.WARNING, logger="verl_omni.workers.rollout.vllm_rollout.prefix_caching"):
            warn_if_prefix_caching_mismatch(self._fake_engine(True), config)
        assert any("enable_prefix_caching" in r.message for r in caplog.records)

    def test_match_is_silent(self, caplog):
        config = types.SimpleNamespace(enable_prefix_caching=True)
        with caplog.at_level(logging.WARNING, logger="verl_omni.workers.rollout.vllm_rollout.prefix_caching"):
            warn_if_prefix_caching_mismatch(self._fake_engine(True), config)
        assert not caplog.records

    def test_non_bool_config_is_silent(self, caplog):
        config = types.SimpleNamespace(enable_prefix_caching=None)
        with caplog.at_level(logging.WARNING, logger="verl_omni.workers.rollout.vllm_rollout.prefix_caching"):
            warn_if_prefix_caching_mismatch(self._fake_engine(True), config)
        assert not caplog.records

    def test_engine_without_attrs_never_raises(self):
        config = types.SimpleNamespace(enable_prefix_caching=False)
        warn_if_prefix_caching_mismatch(object(), config)  # must not raise
        warn_if_prefix_caching_mismatch(None, config)
