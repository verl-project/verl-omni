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

"""DeSync reward adapted from zghhui/OmniNFT."""

import importlib
import math
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from .reward_utils import audio_info_from_batch, get_audio, load_torch_state_dict, resample_audio

_TARGET_VIDEO_FPS = 25.0
_TARGET_AUDIO_RATE = 16_000
_VIDEO_FRAMES = 200
_AUDIO_SAMPLES = 128_000
_VIDEO_SEGMENT = 16
_VIDEO_STEP = 8
_AUDIO_SEGMENT = 10_240
_AUDIO_STEP = 5_120
_SEGMENTS = 24
_COMPARE_SEGMENTS = 14
_MEL_TIME = 66
_CLASS_GRID = torch.linspace(-2.0, 2.0, 21)
_MHA_FASTPATH_LOCK = threading.Lock()
_SOURCE_IMPORT_LOCK = threading.Lock()


@contextmanager
def _source_import_path(source_root: Path):
    """Temporarily prepend a process-wide import path; caller owns serialization."""
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        sys.path.remove(str(source_root))


def _legacy_find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        shifted_head = head - sum(pruned_head < head for pruned_head in already_pruned_heads)
        mask[shifted_head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(mask.shape[0])[mask].long()
    return heads, index


def _legacy_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.ndim == 1:
        head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.ndim == 2:
        head_mask = head_mask[:, None, :, None, None]
    if head_mask.ndim != 5:
        raise ValueError("head_mask must have dimension 1, 2, or 5.")
    head_mask = head_mask.to(dtype=self.dtype)
    return head_mask.unsqueeze(-1) if is_attention_chunked else head_mask


@contextmanager
def _temporary_transformers_ast_import_compat():
    """Expose a missing legacy Transformers symbol for the duration of import.

    This changes the process-wide module, not one model instance. Remove only
    the symbol installed here on exit; the caller holds ``_SOURCE_IMPORT_LOCK``.
    """
    from transformers import pytorch_utils

    installed = not hasattr(pytorch_utils, "find_pruneable_heads_and_indices")
    if installed:
        pytorch_utils.find_pruneable_heads_and_indices = _legacy_find_pruneable_heads_and_indices
    try:
        yield
    finally:
        if installed:
            del pytorch_utils.find_pruneable_heads_and_indices


def _import_synchformer(root: Path, module_path: Path):
    """Import one Synchformer source tree under a process-local lock.

    Temporary path/symbol changes are restored after import. Imported modules
    remain cached. Reject an already imported different source path; the path
    check does not verify source contents.
    """
    module_name = "flow_grpo.audio_video_align.synchformer.synchformer"
    ast_module_name = "flow_grpo.audio_video_align.synchformer.hf_src.modeling_ast"
    with _SOURCE_IMPORT_LOCK:
        existing = sys.modules.get(module_name)
        if existing is not None and Path(existing.__file__).resolve() != module_path:
            raise RuntimeError("A different Synchformer source_root is already imported in this process.")
        with _source_import_path(root), _temporary_transformers_ast_import_compat():
            module = existing or importlib.import_module(module_name)
            importlib.import_module(ast_module_name)
    return module


def _install_ast_head_mask_compat(model) -> None:
    """Add the removed Transformers method only to this Synchformer instance."""
    ast_base = sys.modules["flow_grpo.audio_video_align.synchformer.hf_src.modeling_ast"].ASTPreTrainedModel
    for component in model.modules():
        if isinstance(component, ast_base) and not hasattr(component, "get_head_mask"):
            component.get_head_mask = MethodType(_legacy_get_head_mask, component)


def _load_components(model_path: str, source_root: str) -> tuple[Any, Any]:
    """Load the configured Synchformer source/state strictly and create a 16 kHz mel transform.

    Check the imported module's path and tensor-only checkpoint, freeze model
    parameters, and return model/mel components without moving them to an accelerator.
    """
    root = Path(source_root).expanduser().resolve()
    module_path = root / "flow_grpo/audio_video_align/synchformer/synchformer.py"
    config_path = module_path.parent / "divided_224_16x4.yaml"
    if not module_path.is_file() or not config_path.is_file():
        raise ValueError("DeSync source_root is missing the Synchformer source or fixed config.")
    module = _import_synchformer(root, module_path)
    if Path(module.__file__).resolve() != module_path:
        raise RuntimeError("Imported Synchformer does not belong to the configured source_root.")

    model = module.Synchformer()
    _install_ast_head_mask_compat(model)
    state_dict = load_torch_state_dict(model_path)
    if not isinstance(state_dict, dict) or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("DeSync checkpoint must be a tensor state dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=_TARGET_AUDIO_RATE, win_length=400, hop_length=160, n_fft=1024, n_mels=128
    )
    return model, mel


def _temporal_resample_video(video: torch.Tensor, source_fps: float) -> torch.Tensor:
    duration = video.shape[0] / source_fps
    frame_count = int(math.floor(duration * _TARGET_VIDEO_FPS + 1e-9))
    if frame_count <= 0:
        raise ValueError("DeSync video is too short for 25 fps resampling.")
    frame_count = min(frame_count, _VIDEO_FRAMES)
    # ffmpeg `fps=25` (torio): sample each output slot at its center.
    output_index = torch.arange(frame_count, dtype=torch.float64)
    indices = torch.ceil((output_index + 0.5) * source_fps / _TARGET_VIDEO_FPS).long() - 1
    return video[indices.clamp(0, video.shape[0] - 1)]


def _resize_crop_video(video: torch.Tensor) -> torch.Tensor:
    height, width = video.shape[-2:]
    if min(height, width) <= 0:
        raise ValueError("DeSync video spatial dimensions must be positive.")
    if height <= width:
        size = (224, int(width * 224 / height))
    else:
        size = (int(height * 224 / width), 224)
    video = F.interpolate(video.float().div(255), size=size, mode="bicubic", align_corners=False, antialias=True)
    top = (size[0] - 224) // 2
    left = (size[1] - 224) // 2
    return video[:, :, top : top + 224, left : left + 224].sub(0.5).div(0.5)


def _prepare_video(video: torch.Tensor, source_fps: float) -> torch.Tensor:
    video = _resize_crop_video(_temporal_resample_video(video, source_fps))
    if video.shape[0] < _VIDEO_FRAMES:
        video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, _VIDEO_FRAMES - video.shape[0]), value=-1.0)
    return video


def _prepare_audio(audio: torch.Tensor, source_rate: int) -> torch.Tensor:
    waveform = resample_audio(audio.float().mean(dim=0), source_rate, _TARGET_AUDIO_RATE)[:_AUDIO_SAMPLES]
    return F.pad(waveform, (0, _AUDIO_SAMPLES - waveform.shape[0]))


def _pad_mel_time(mel: torch.Tensor) -> torch.Tensor:
    if mel.shape[-1] < _MEL_TIME:
        return F.pad(mel, (0, _MEL_TIME - mel.shape[-1]))
    return mel[..., :_MEL_TIME]


async def compute_score(
    data_source=None,
    solution_image=None,
    ground_truth=None,
    extra_info=None,
    *,
    reward_model,
    batch=None,
    fps: float | None = None,
    **kwargs,
) -> dict[str, float]:
    """Score AV synchrony as 1 / (1 + mean absolute predicted offset).

    Read audio/rates from extra_info or a single-sample batch. The checkpoint's
    21 offset classes and fixed segment preprocessing define this model's score.
    """
    del data_source, ground_truth, kwargs
    extra_info = audio_info_from_batch(extra_info, batch, scorer="DeSync")
    if batch is not None:
        item = batch[0]
        if solution_image is None:
            solution_image = item.batch["responses"]
        if "fps" in item.batch:
            extra_info["fps"] = item.batch["fps"]
        elif "fps" in item.non_tensor_batch:
            extra_info["fps"] = item.non_tensor_batch["fps"]
    fps = float(extra_info["fps"] if fps is None else fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("DeSync fps must be finite and positive.")
    if solution_image.ndim != 4 or solution_image.shape[1] != 3 or solution_image.dtype != torch.uint8:
        raise ValueError("DeSync requires uint8 RGB video with shape [T,3,H,W].")
    waveform, source_rate = get_audio(extra_info)
    video = _prepare_video(solution_image.detach().cpu(), fps)
    audio = _prepare_audio(waveform.unsqueeze(0), source_rate)
    logits = await reward_model.infer(video.unsqueeze(0), audio.unsqueeze(0))
    if not isinstance(logits, torch.Tensor) or logits.shape != (2, 1, 21):
        raise ValueError("DeSync logits must have shape (2, 1, 21).")
    offsets = _CLASS_GRID[logits.argmax(dim=-1)].abs()
    score = (1.0 / (1.0 + offsets.mean())).float()
    if not torch.isfinite(score):
        raise ValueError("DeSync score must be finite.")
    return {"score": float(score)}


class DeSyncModel:
    """Raw Synchformer inference adapter owned by a native reward executor."""

    def __init__(self, model_path: str, device, source_root: str) -> None:
        self.model, self.mel = _load_components(model_path, source_root)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mel.to(self.device)

    def close(self) -> None:
        """Drop model/mel references; reuse requires constructing a new adapter."""
        self.model = None
        self.mel = None
        self.device = None

    def _infer_micro_batch(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Expand each prepared AV sample into 24 overlapping aligned segments.

        Extract video and normalized log-mel features on the active device, then
        compare the first and last 14 segments separately. Return detached CPU
        logits ``[2, B, 21]`` in model-output dtype; the caller sets inference mode
        and the MHA compatibility switch.
        """
        batch_size = video.shape[0]
        video_segments = video.unfold(1, _VIDEO_SEGMENT, _VIDEO_STEP).movedim(-1, 2)
        audio_segments = audio.unfold(1, _AUDIO_SEGMENT, _AUDIO_STEP)
        if video_segments.shape[1] != _SEGMENTS or audio_segments.shape[1] != _SEGMENTS:
            raise ValueError("DeSync preprocessing must produce exactly 24 AV segments.")

        visual = video_segments.reshape(-1, _VIDEO_SEGMENT, *video.shape[2:]).unsqueeze(1).to(self.device)
        visual = self.model.extract_vfeats(visual)
        visual = visual.reshape(batch_size, _SEGMENTS, *visual.shape[2:])

        audio_segments = audio_segments.to(self.device)
        mel = _pad_mel_time(torch.log(self.mel(audio_segments) + 1e-6))
        # Match OmniNFT's Synchformer preprocessing:
        # https://github.com/zghhui/OmniNFT/blob/master/flow_grpo/audio_video_align/av_desync.py
        mel = (mel - (-4.2677393)) / (2 * 4.5689974)
        auditory = self.model.extract_afeats(mel.unsqueeze(2))

        logits_batches = []
        for start in (0, _SEGMENTS - _COMPARE_SEGMENTS):
            logits = self.model.compare_v_a(
                visual[:, start : start + _COMPARE_SEGMENTS], auditory[:, start : start + _COMPARE_SEGMENTS]
            )
            if not isinstance(logits, torch.Tensor) or logits.shape != (batch_size, 21):
                raise ValueError(f"DeSync logits must have shape ({batch_size}, 21).")
            if not torch.isfinite(logits).all():
                raise ValueError("DeSync logits must contain only finite values.")
            logits_batches.append(logits.detach().cpu())
        return torch.stack(logits_batches)

    @torch.inference_mode()
    def infer(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Infer prepared video ``[B, 200, 3, 224, 224]`` and audio ``[B, 128000]``.

        Return detached CPU offset logits ``[2, B, 21]`` without gradients.
        On NPU, disable the process-wide MHA fastpath for the forward because the
        Synchformer attention path is unsupported there. A module
        lock serializes these calls and ``finally`` restores the previous flag;
        unrelated callers that do not use this lock can observe the temporary flag.
        """
        if self.device.type != "npu":
            return self._infer_micro_batch(video, audio)
        with _MHA_FASTPATH_LOCK:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                return self._infer_micro_batch(video, audio)
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
