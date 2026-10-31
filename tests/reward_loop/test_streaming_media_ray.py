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
"""Real-Ray named-media transport with CPU JPEG scoring; no GPU inference."""

import os
import tempfile
from pathlib import Path

import pytest
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.protocol import DataProto

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_loop import OmniRewardLoopWorker
from verl_omni.reward_loop.streaming import create_async_reward_client
from verl_omni.workers.config.reward import StreamingRewardConfig


def score_image_shapes(data_source, solution_image, ground_truth, extra_info):
    """Require pipeline geometry at the real reward worker's scorer boundary."""
    shapes = extra_info["img_shapes"]
    assert shapes == [(1, 4, 6), (1, 8, 8)]
    return {"score": float(sum(height * width for _, height, width in shapes))}


@pytest.mark.asyncio
async def test_real_ray_named_media_matches_batch_and_rejects_invalid_preview():
    """Exercise serialization and both worker entrypoints with an optional scorer."""
    with tempfile.TemporaryDirectory(prefix="vosmedia-") as temp_dir:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=temp_dir,
            object_store_memory=128 * 1024 * 1024,
        )
        try:
            with initialize_config_dir(config_dir=str(Path("verl_omni/trainer/config").resolve()), version_base=None):
                config = compose(config_name="diffusion_trainer")
            config.actor_rollout_ref.model.tokenizer_path = os.environ["STREAMING_TEST_TOKENIZER"]
            config.reward.reward_manager.name = "MultiVisualRewardManager"
            config.reward.reward_functions = OmegaConf.create(
                {
                    "jpeg": {
                        "path": "pkg://verl_omni.utils.reward_score.jpeg_compressibility",
                        "name": "compute_score",
                        "weight": 0.3,
                        "required": False,
                    },
                    "geometry": {
                        "path": str(Path(__file__).resolve()),
                        "name": "score_image_shapes",
                        "weight": 0.7,
                        "required": True,
                    },
                }
            )
            worker = ray.remote(OmniRewardLoopWorker).options(num_cpus=1).remote(config)
            client = create_async_reward_client({"jpeg": (worker,)}, StreamingRewardConfig(extra_fields=[]))
            try:
                latent = torch.ones(16, 2, 2)
                fields = artifact_fields(
                    {
                        "image_latent": MediaArtifact(MediaSpec("image", "latent", "CHW"), latent),
                        "image_preview": MediaArtifact(
                            MediaSpec("image", "decoded", "CHW"), torch.zeros(3, 8, 8, dtype=torch.uint8)
                        ),
                    },
                    "image_latent",
                    "image_preview",
                )
                fields["img_shapes"] = [(1, 4, 6), (1, 8, 8)]
                sample = DataProto.from_dict(
                    tensors={"responses": latent.unsqueeze(0)},
                    non_tensors={
                        "data_source": ["ray"],
                        "reward_model": [{"ground_truth": "prompt"}],
                        "tool_extra_fields": [fields],
                    },
                )
                batch = await worker.compute_score_batch.remote(sample)
                assert batch[0]["reward_extra_info"]["reward/geometry"] == 88.0
                async with client.sample_slot():
                    assert await client.compute_score(sample) == batch[0]
                fields["preview_artifact"] = "absent"
                for invoke in (worker.compute_score_batch.remote, client.compute_score):
                    with pytest.raises(ray.exceptions.RayTaskError, match="absent"):
                        async with client.sample_slot():
                            await invoke(sample)
            finally:
                await client.close()
                ray.kill(worker)
        finally:
            ray.shutdown()
