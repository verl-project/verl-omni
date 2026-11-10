import logging

from .diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT

__all__ = ["MiniMaxH3ImageDiffusionNFT"]

_IMAGE_PIPELINE_MODULE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image"

try:
    from .vllm_omni_rollout_adapter import MiniMaxH3ImageDiffusionNFTPipeline  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != _IMAGE_PIPELINE_MODULE:
        raise
    logging.getLogger(__name__).warning(
        "MiniMaxH3ImagePipeline rollout adapter is not registered: installed vLLM-Omni lacks %s.",
        _IMAGE_PIPELINE_MODULE,
    )
else:
    __all__.append("MiniMaxH3ImageDiffusionNFTPipeline")
