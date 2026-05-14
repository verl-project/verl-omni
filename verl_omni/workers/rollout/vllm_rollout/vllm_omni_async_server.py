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
from dataclasses import asdict
from typing import Any, Optional

import ray
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
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

        import_external_libs(self.config.external_lib)
        pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
            architecture=self.model_config.architecture,
            algorithm=self.model_config.algorithm,
        )
        # TODO (mike): read custom_pipeline from engine_args
        if pipeline_path is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

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
        priority: int = 0,
    ) -> DiffusionOutput:
        """Generate sequence with token-in-image-out."""
        outputs = await self._generate_engine_call(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            num_outputs_per_prompt=1,
            image_data=image_data,
            video_data=video_data,
            negative_prompt_ids=negative_prompt_ids,
        )
        return outputs[0]

    async def generate_batched(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        num_outputs_per_prompt: int,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
    ) -> list[DiffusionOutput]:
        """Submit one engine request that produces ``num_outputs_per_prompt`` samples
        in a single B=N transformer forward pass.

        This bypasses the orchestrator's per-request serialization
        (``StageDiffusionClient`` runs each ``add_request_async`` through a
        ``ThreadPoolExecutor(max_workers=1)``) by letting the underlying
        ``QwenImagePipelineWithLogProb`` forward batch ``N`` latents together,
        the same way ``diffusers.QwenImagePipeline`` does with
        ``num_images_per_prompt=N``.

        Returns:
            list[DiffusionOutput]: One entry per generated sample. Each entry
            has the same shape as a single :meth:`generate` call would return
            (``diffusion_output`` is CHW, ``log_probs`` and ``extra_fields[...]``
            are sliced per-sample).
        """
        if num_outputs_per_prompt < 1:
            raise ValueError(f"num_outputs_per_prompt must be >= 1, got {num_outputs_per_prompt}")
        return await self._generate_engine_call(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            num_outputs_per_prompt=num_outputs_per_prompt,
            image_data=image_data,
            video_data=video_data,
            negative_prompt_ids=negative_prompt_ids,
        )

    async def _generate_engine_call(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        num_outputs_per_prompt: int,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
    ) -> list[DiffusionOutput]:
        """Shared implementation for :meth:`generate` and :meth:`generate_batched`.

        Builds the ``OmniCustomPrompt`` + ``OmniDiffusionSamplingParams``, calls
        the async engine once, then slices the resulting ``OmniRequestOutput``
        into ``num_outputs_per_prompt`` per-sample ``DiffusionOutput`` objects.
        """
        prompt_ids = normalize_token_ids(prompt_ids)

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
        custom_prompt: OmniCustomPrompt = {"prompt_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        # Build OmniDiffusionSamplingParams from the incoming dict, overriding
        # num_outputs_per_prompt so the pipeline runs a B=N transformer forward.
        effective_sp: dict[str, Any] = dict(sampling_params)
        effective_sp["num_outputs_per_prompt"] = num_outputs_per_prompt

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in effective_sp.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=[diffusion_sampling_params],
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        if len(final_res.images) < num_outputs_per_prompt:
            raise RuntimeError(
                f"Expected {num_outputs_per_prompt} images for request {request_id}, "
                f"got {len(final_res.images)} from the engine."
            )

        return self._unpack_batched_result(
            final_res=final_res,
            sampling_params=sampling_params,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )

    def _unpack_batched_result(
        self,
        *,
        final_res: OmniRequestOutput,
        sampling_params: dict[str, Any],
        num_outputs_per_prompt: int,
    ) -> list[DiffusionOutput]:
        """Slice a batched ``OmniRequestOutput`` into per-sample ``DiffusionOutput``s.

        Per-sample tensors in ``custom_output`` are expected to have shape
        ``(N, ...)`` where ``N == num_outputs_per_prompt``; index ``i`` is
        used for the *i*-th returned ``DiffusionOutput``.
        """
        mm_output = final_res.custom_output or {}

        all_log_probs = None
        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")

        # Determine stop reason from finish_reason (shared across all samples
        # in this batched request).
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

        def _take(tensor, index):
            return tensor[index] if tensor is not None else None

        outputs: list[DiffusionOutput] = []
        for i in range(num_outputs_per_prompt):
            diffusion_output = self._to_tensor(final_res.images[i]).float() / 255.0
            extra_fields = {
                "all_latents": _take(all_latents, i),
                "all_timesteps": _take(all_timesteps, i),
                "prompt_embeds": _take(prompt_embeds, i),
                "prompt_embeds_mask": _take(prompt_embeds_mask, i),
                "negative_prompt_embeds": _take(negative_prompt_embeds, i),
                "negative_prompt_embeds_mask": _take(negative_prompt_embeds_mask, i),
                "global_steps": self.global_steps,
            }
            outputs.append(
                DiffusionOutput(
                    diffusion_output=diffusion_output,
                    log_probs=_take(all_log_probs, i),
                    stop_reason=stop_reason,
                    num_preempted=num_preempted,
                    extra_fields=extra_fields,
                )
            )

        return outputs

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
