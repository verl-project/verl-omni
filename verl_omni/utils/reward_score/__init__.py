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

"""Visual (image) reward scoring functions for VeRL-Omni."""


def default_compute_score_image(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Compute the reward score for a visual (image) response.

    Args:
        data_source (str): Dataset identifier that determines the scoring method.
        solution_image: The generated image, as a ``torch.Tensor`` in shape
            ``(C, H, W)`` or ``(N, C, H, W)``.
        ground_truth (str): Ground-truth answer (may be unused for rule-based
            rewards such as ``jpeg_compressibility``).
        extra_info (dict, optional): Additional metadata passed by the reward
            manager.

    Returns:
        float or dict: The computed score (or a dict with a ``"score"`` key).

    Raises:
        NotImplementedError: If no scorer is registered for *data_source*.
    """
    if data_source == "jpeg_compressibility":
        from . import jpeg_compressibility

        res = jpeg_compressibility.compute_score(solution_image)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


__all__ = ["default_compute_score_image"]
