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
import argparse
import logging
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.net_utils import get_free_port
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """Use DiffusionModelConfig instead of HFModelConfig."""
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """No-op: diffusion models don't have max_position_embeddings."""
        pass

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Omni-specific post-init: create PIL→tensor converter, then log."""
        self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """Diffusion models have no LLM sampling params; return empty dict."""
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        # inject multi-stage yaml config
        deploy_config = getattr(args, "deploy_config", None)
        if deploy_config:
            engine_args["deploy_config"] = deploy_config

        import_external_libs(self.config.external_lib)

        self.config.resolve_algorithm(self.model_config)

        pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
            architecture=self.model_config.architecture,
            algorithm=self.model_config.algorithm,
        )
        # TODO (mike): read custom_pipeline from engine_args
        if pipeline_path is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

        if self.config.step_execution:
            engine_args["step_execution"] = True

        diffusion_master_port, diffusion_master_sock = get_free_port("127.0.0.1", with_alive_sock=True)
        diffusion_master_sock.close()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(diffusion_master_port)
        logger.info("Using MASTER_PORT=%s for vLLM-Omni diffusion workers", os.environ["MASTER_PORT"])

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def wake_up(self, tags: list[str] | None = None):
        """Override parent to use collective_rpc instead of engine.wake_up().

        The parent (verl ``1927ad33``+) calls ``self.engine.wake_up(tags=...)``
        which triggers CUDA initialisation in this HTTP server process when
        running under vLLM-Omni (AsyncOmni engine).
        Use ``collective_rpc`` instead.

        # TODO (long): drop this override once vllm-omni wake_up
        without triggering GPU initialisation.
        """
        if self.node_rank != 0:
            return
        await self.engine.collective_rpc(
            "wake_up", kwargs={"tags": tags if tags is not None else self._get_wake_up_tags()}
        )

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        priority: int = 0,
    ) -> DiffusionOutput:
        """Generate sequence with token-in-image-out."""
        prompt_ids = normalize_token_ids(prompt_ids)
        default_params_list = self.engine.default_sampling_params_list

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # Build OmniCustomPrompt with pre-tokenized IDs
        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if prompt_mask is not None:
            custom_prompt["prompt_mask"] = prompt_mask
        if len(default_params_list) > 1:
            # Multi-stage pipelines tag the diffusion stage so the orchestrator can route inputs correctly.
            custom_prompt["modalities"] = ["image"]
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["multi_modal_data"] = multi_modal_data
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        # Build OmniDiffusionSamplingParams from the incoming dict
        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Build sampling params list: multi-stage models use defaults for non-diffusion stages
        sampling_params_list = default_params_list[:-1] + [diffusion_sampling_params]

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        def _maybe_unbatch(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[0] if value.dim() > 0 else value
            if isinstance(value, np.ndarray):
                return value[0] if value.ndim > 0 else value
            if isinstance(value, list | tuple):
                return value[0] if value else None
            return value

        extra_fields = {k: _maybe_unbatch(v) for k, v in mm_output.items() if k != "all_log_probs"}
        extra_fields["global_steps"] = self.global_steps

        # Determine stop reason from finish_reason
        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig,
        model_config: DiffusionModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
