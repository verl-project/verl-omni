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
"""Compare production batch and streaming scoring on identical saved images."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.streaming import create_async_reward_client
from verl_omni.workers.config.reward import get_streaming_reward_config


def main():
    """Load real generated images and use the same resident reward workers twice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.run / "config.yaml")
    config.actor_rollout_ref.model.tokenizer_path = str(Path(config.actor_rollout_ref.model.path) / "tokenizer")
    rows = [json.loads(line) for line in (args.run / "rollouts/1.jsonl").read_text().splitlines()]
    images = [torch.from_numpy(np.array(Image.open(row["output"]).convert("RGB"))).permute(2, 0, 1) for row in rows]
    samples = []
    for row, image in zip(rows, images, strict=True):
        fields = artifact_fields(
            {"image_preview": MediaArtifact(MediaSpec("image", "decoded", "CHW"), image)},
            "image_preview",
            "image_preview",
        )
        samples.append(
            DataProto.from_dict(
                tensors={"responses": image.unsqueeze(0)},
                non_tensors={
                    "data_source": ["same_image_parity"],
                    "reward_model": [{"ground_truth": row["gts"]}],
                    "tool_extra_fields": [fields],
                },
            )
        )
    data = DataProto.concat(samples)
    ray.init(address="local", num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
    try:
        pool = RayResourcePool([1], use_gpu=True, max_colocate_count=2, name_prefix="parity_reward")
        manager = OmniRewardLoopManager(config, rm_resource_pool=pool)

        async def compare():
            client = create_async_reward_client(manager.reward_loop_worker_handles, get_streaming_reward_config(config))
            try:
                batch = await manager.async_compute_rm_score(data)

                async def score(sample):
                    async with client.sample_slot():
                        return await client.compute_score(sample)

                streamed = await asyncio.gather(*(score(data[i : i + 1]) for i in range(len(data))))
                scores = torch.tensor([item["reward_score"] for item in streamed])
                torch.testing.assert_close(scores, batch.batch["rm_scores"].flatten(), rtol=1e-5, atol=1e-6)
                for i, item in enumerate(streamed):
                    info = item["reward_extra_info"]
                    for key, value in info.items():
                        np.testing.assert_allclose(value, batch.non_tensor_batch[key][i], rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(
                        item["reward_score"], 0.7 * info["reward/pickscore"] + 0.3 * info["reward/jpeg"], atol=1e-8
                    )
                args.report.write_text(
                    json.dumps(
                        {
                            "samples": len(rows),
                            "groups": list(manager.reward_loop_worker_handles),
                            "max_score_delta": (scores - batch.batch["rm_scores"].flatten()).abs().max().item(),
                            "batch_scores": batch.batch["rm_scores"].flatten().tolist(),
                            "streaming_scores": scores.tolist(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                await client.close()
                await manager.multi_reward_model_manager.close_native_models()

        asyncio.run(compare())
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
