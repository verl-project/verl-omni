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

from unittest.mock import patch

from omegaconf import OmegaConf

from verl_omni.utils import diffusion_attention as da


def _make_config(attn_backend: str = da.ACTOR_FA3_BACKEND):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"attn_backend": attn_backend},
                "rollout": {"name": "vllm_omni"},
            }
        }
    )


class TestDiffusionAttentionFallback:
    def test_keeps_fa3_when_available(self, monkeypatch):
        monkeypatch.delenv(da.DIFFUSION_ATTENTION_ENV, raising=False)
        config = _make_config()
        with patch.object(da, "fa3_available", return_value=True):
            da.fallback_fa3_if_unavailable(config)
        assert config.actor_rollout_ref.model.attn_backend == da.ACTOR_FA3_BACKEND
        import os

        assert os.environ[da.DIFFUSION_ATTENTION_ENV] == da.ROLLOUT_FA3_BACKEND

    def test_falls_back_to_native_when_fa3_unavailable(self, monkeypatch):
        monkeypatch.delenv(da.DIFFUSION_ATTENTION_ENV, raising=False)
        config = _make_config()
        with patch.object(da, "fa3_available", return_value=False):
            da.fallback_fa3_if_unavailable(config)
        assert config.actor_rollout_ref.model.attn_backend == da.ACTOR_NATIVE_BACKEND
        import os

        assert os.environ[da.DIFFUSION_ATTENTION_ENV] == da.ROLLOUT_NATIVE_BACKEND

    def test_noop_for_native_backend(self, monkeypatch):
        monkeypatch.delenv(da.DIFFUSION_ATTENTION_ENV, raising=False)
        config = _make_config(da.ACTOR_NATIVE_BACKEND)
        with patch.object(da, "fa3_available", return_value=False):
            da.fallback_fa3_if_unavailable(config)
        assert config.actor_rollout_ref.model.attn_backend == da.ACTOR_NATIVE_BACKEND
        import os

        assert da.DIFFUSION_ATTENTION_ENV not in os.environ

    def test_ray_init_injection_in_struct_mode(self, monkeypatch):
        monkeypatch.delenv(da.DIFFUSION_ATTENTION_ENV, raising=False)
        # Create a config with ray_kwargs and ray_init, and set struct mode to True
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": da.ACTOR_FA3_BACKEND},
                    "rollout": {"name": "vllm_omni"},
                },
                "ray_kwargs": {"ray_init": {}},
            }
        )
        OmegaConf.set_struct(config, True)

        with patch.object(da, "fa3_available", return_value=True):
            da.fallback_fa3_if_unavailable(config)

        assert config.ray_kwargs.ray_init.runtime_env.env_vars[da.DIFFUSION_ATTENTION_ENV] == da.ROLLOUT_FA3_BACKEND

    def test_ray_worker_propagation(self, monkeypatch):
        import ray

        monkeypatch.delenv(da.DIFFUSION_ATTENTION_ENV, raising=False)
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"attn_backend": da.ACTOR_FA3_BACKEND},
                    "rollout": {"name": "vllm_omni"},
                },
                "ray_kwargs": {"ray_init": {}},
            }
        )
        OmegaConf.set_struct(config, True)

        with patch.object(da, "fa3_available", return_value=False):
            da.fallback_fa3_if_unavailable(config)

        # Retrieve ray_init container
        ray_init_kwargs = OmegaConf.to_container(config.ray_kwargs.ray_init, resolve=True)

        # Initialize Ray with the generated runtime_env
        if ray.is_initialized():
            ray.shutdown()
        ray.init(**ray_init_kwargs)

        try:

            @ray.remote
            def check_env_on_worker():
                import os

                return os.environ.get(da.DIFFUSION_ATTENTION_ENV)

            # Verify that the environment variable was successfully propagated to Ray workers
            worker_val = ray.get(check_env_on_worker.remote())
            assert worker_val == da.ROLLOUT_NATIVE_BACKEND
        finally:
            ray.shutdown()
