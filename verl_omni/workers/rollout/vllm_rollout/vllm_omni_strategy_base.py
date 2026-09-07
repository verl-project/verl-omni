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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from vllm_omni.lora.request import LoRARequest

if TYPE_CHECKING:
    from argparse import Namespace

    from verl.workers.rollout.replica import TokenOutput

    from verl_omni.workers.rollout.replica import DiffusionOutput
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class OmniStrategyBase(ABC):
    """Abstract base class for a vLLM-Omni generation *mode*.

    ``vLLMOmniHttpServer`` supports two very different rollout modes -- an
    autoregressive/thinker mode and a diffusion mode -- that require different
    configs, engine arguments, request payloads, and output types. Rather than
    branch on the mode throughout the server, the server selects exactly one
    concrete subclass once in ``_init_config`` and delegates every mode-specific
    hook to it, while shared engine lifecycle (sleep/wake/abort, LoRA cache,
    replica management) stays on the server.

    The full generation flow (:meth:`generate`) lives here so both modes share
    prompt normalization, multimodal assembly, and LoRA resolution; only the
    three per-request steps -- :meth:`preprocess_input`, :meth:`run_generation`,
    and :meth:`process_output` -- differ.

    To add a mode, subclass this ABC and:

    * set the two config dataclass attributes (:attr:`rollout_config_cls` and
      :attr:`model_config_cls`);
    * implement the abstract hooks (:meth:`worker_extension_cls`,
      :meth:`prepare_engine_args`, :meth:`preprocess_input`,
      :meth:`run_generation`, :meth:`process_output`);
    * optionally override the concrete hooks (:meth:`init_config`,
      :meth:`init_model_config`, :meth:`validate_configs`, :meth:`post_init`,
      :meth:`apply_quantization`, :meth:`override_generation_config`,
      :meth:`preprocess_engine_kwargs`) when the mode needs behavior beyond the
      shared defaults.

    The two concrete subclasses are
    :class:`~verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy.ARStrategy`
    and
    :class:`~verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy.DiffusionStrategy`.
    """

    #: Rollout-config dataclass the mode expects. Subclasses must set this.
    rollout_config_cls: type
    #: Model-config dataclass the mode expects. Subclasses must set this.
    model_config_cls: type

    def __init__(self, server: vLLMOmniHttpServer) -> None:
        self.server = server

    def init_config(self, config: Any) -> Any:
        """Convert the raw rollout config into :attr:`rollout_config_cls`.

        Called once during ``vLLMOmniHttpServer`` initialization. Override only
        if the mode needs more than a plain dataclass conversion.

        Args:
            config: The raw (OmegaConf) rollout config.

        Returns:
            The mode's rollout-config dataclass instance.
        """
        return omega_conf_to_dataclass(config, dataclass_type=self.rollout_config_cls)

    def init_model_config(self, model_config: Any) -> Any:
        """Convert the raw model config into :attr:`model_config_cls`.

        Args:
            model_config: The raw (OmegaConf) model config.

        Returns:
            The mode's model-config dataclass instance.
        """
        return omega_conf_to_dataclass(model_config, dataclass_type=self.model_config_cls)

    def validate_configs(self) -> None:
        """Validate/normalize ``self.server.config`` after it is built.

        Default no-op. Override to enforce mode-specific config invariants
        (for example, deriving ``max_model_len`` from prompt/response lengths).
        """
        return None

    def post_init(self, cuda_visible_devices: str) -> None:
        """Run at the end of ``vLLMOmniHttpServer`` post-initialization.

        Default no-op. Override to set up mode-specific server state.

        Args:
            cuda_visible_devices: The device string visible to this server.
        """
        return None

    def apply_quantization(self) -> tuple[str | None, dict[str, Any]]:
        """Return the ``(quantization, extra_kwargs)`` pair for the engine.

        Default returns ``(None, {})`` (no quantization). Override for modes
        that reuse the base vLLM quantization path.
        """
        return None, {}

    def override_generation_config(self) -> dict[str, Any]:
        """Return generation-config overrides merged into the engine defaults.

        Default returns ``{}``. Override for modes that must force specific
        generation settings.
        """
        return {}

    @abstractmethod
    def worker_extension_cls(self, device_type: str) -> str:
        """Return the import path of the worker-extension class for *device_type*.

        Args:
            device_type: The runtime device family, e.g. ``cuda`` or ``npu``.

        Returns:
            Fully-qualified class path of the colocate worker extension.
        """

    def preprocess_engine_kwargs(self, engine_kwargs: dict[str, Any]) -> None:
        """Mutate ``engine_kwargs`` in place before the engine is created.

        The base implementation strips the mode selector (``output_mode``).
        Override (calling ``super().preprocess_engine_kwargs(...)``) to consume
        additional mode-specific kwargs.

        Args:
            engine_kwargs: The engine keyword-argument dict, mutated in place.
        """
        engine_kwargs.pop("output_mode", None)

    @abstractmethod
    def prepare_engine_args(self, engine_args: dict[str, Any], args: Namespace) -> None:
        """Mutate ``engine_args`` in place with mode-specific engine arguments.

        Args:
            engine_args: The engine argument dict, mutated in place.
            args: The parsed launch namespace.
        """

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        """Shared generation template used by both modes.

        Normalizes the prompt ids, assembles multimodal data, resolves the LoRA
        request, then runs the mode-specific pipeline
        :meth:`preprocess_input` -> :meth:`run_generation` -> :meth:`process_output`.
        Subclasses should not override this; they customize those three steps.

        Returns:
            ``TokenOutput`` for AR mode or ``DiffusionOutput`` for diffusion mode.
        """
        prompt_ids = normalize_token_ids(prompt_ids)
        multi_modal_data = self._build_multi_modal_data(image_data, video_data, audio_data)
        lora_request = await self.server._resolve_lora_request()
        prompt, params = self.preprocess_input(
            prompt_ids,
            sampling_params,
            multi_modal_data,
            lora_request,
            negative_prompt_ids,
            prompt_mask,
            mm_processor_kwargs,
            extra_prompt_ids,
            negative_extra_prompt_ids,
        )
        final_res = await self.run_generation(prompt, params, request_id, lora_request, priority)
        return self.process_output(final_res, params, sampling_params)

    @staticmethod
    def _build_multi_modal_data(
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        audio_data: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Assemble vLLM multimodal inputs from the public rollout arguments."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data
        return multi_modal_data

    @staticmethod
    def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
        """Map a vLLM finish reason to verl's stop-reason vocabulary."""
        if finish_reason == "abort":
            return "aborted"
        if finish_reason in ("stop", "length"):
            return "completed"
        return finish_reason

    @staticmethod
    async def _collect_last_output(generator: Any) -> Any:
        """Drain an engine output async-generator and return its final item."""
        final_res = None
        async for output in generator:
            final_res = output
        return final_res

    @staticmethod
    def _extract_num_preempted(req_output: Any) -> Optional[int]:
        """Read ``num_preempted`` from an output or its first completion, if present."""
        outputs = getattr(req_output, "outputs", None)
        if outputs and hasattr(outputs[0], "num_preempted"):
            return outputs[0].num_preempted
        if hasattr(req_output, "num_preempted"):
            return req_output.num_preempted
        return None

    @abstractmethod
    def preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        lora_request: Optional[LoRARequest],
        negative_prompt_ids: Optional[list[int]],
        prompt_mask: torch.BoolTensor | None = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
    ) -> tuple[Any, Any]:
        """Build the engine prompt and sampling params for this mode.

        Called by :meth:`generate` after prompt normalization and multimodal
        assembly. AR mode returns a token-centric prompt plus ``SamplingParams``;
        diffusion mode returns an ``OmniCustomPrompt`` plus a diffusion sampling
        params list.

        Returns:
            A ``(prompt, params)`` pair understood by :meth:`run_generation`.
        """

    @abstractmethod
    async def run_generation(
        self,
        prompt: Any,
        params: Any,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
    ) -> Any:
        """Submit the request to the engine and return the final raw output.

        Implementations build the mode-specific ``engine.generate(...)`` call and
        pass the resulting async generator to :meth:`_collect_last_output`.

        Returns:
            The last object yielded by the engine generator (or ``None``).
        """

    @abstractmethod
    def process_output(self, final_res: Any, params: Any, sampling_params: dict[str, Any]) -> Any:
        """Convert the raw engine output into the mode's rollout output type.

        Args:
            final_res: The final raw engine output from :meth:`run_generation`.
            params: The engine params produced by :meth:`preprocess_input`.
            sampling_params: The original public sampling params dict.

        Returns:
            ``TokenOutput`` for AR mode or ``DiffusionOutput`` for diffusion mode.
        """
