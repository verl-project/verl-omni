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
"""Two-step SD3 FlowGRPO with real native PickScore and a separate JPEG group.

Run with two visible GPUs and local model/processor checkpoints. The SD3 model
may be the checkpoint from build_sd3_tiny_random.py or pretrained SD3.5 weights.
Use --actor-offload for pretrained SD3.5 on a single 48GB actor/rollout GPU.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.trainer.main_diffusion_v1 import run_diffusion_v1


def main():
    """Create a small dataset and run the production V1 trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pickscore-model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--actor-offload", action="store_true")
    parser.add_argument("--agent-loop-manager")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompts = ["a red circle on a white background", "a blue square on a black background"]
    dataset = output / "data.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "streaming_e2e",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "text_to_image",
                "reward_model": {"style": "model", "ground_truth": prompt},
                "extra_info": {"index": i},
            }
            for i, prompt in enumerate(prompts)
        ]
    ).to_parquet(dataset)
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    overrides = [
        f"data.train_files={dataset}",
        f"data.val_files={dataset}",
        "data.train_batch_size=2",
        "data.val_batch_size=2",
        "data.val_max_samples=2",
        "data.max_prompt_length=128",
        "actor_rollout_ref.model.algorithm=flow_grpo",
        f"actor_rollout_ref.model.path={args.model.resolve()}",
        "actor_rollout_ref.model.custom_chat_template="
        "\"{% for message in messages %}{{ message['content'] }}{% endfor %}\"",
        "actor_rollout_ref.model.extra_tokenizers="
        "{clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 32}}",
        "actor_rollout_ref.model.attn_backend=native",
        "actor_rollout_ref.model.lora_rank=4",
        "actor_rollout_ref.model.lora_alpha=8",
        "actor_rollout_ref.model.target_modules=['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']",
        "actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(args.actor_offload).lower()}",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "actor_rollout_ref.rollout.seed=42",
        "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA",
        "actor_rollout_ref.rollout.pipeline.height=128",
        "actor_rollout_ref.rollout.pipeline.width=128",
        "actor_rollout_ref.rollout.pipeline.num_inference_steps=4",
        "actor_rollout_ref.rollout.pipeline.guidance_scale=1.0",
        "actor_rollout_ref.rollout.pipeline.max_sequence_length=32",
        "actor_rollout_ref.rollout.max_prompt_embed_length=109",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.algo.sde_window_size=2",
        "actor_rollout_ref.rollout.algo.sde_window_range=[0,3]",
        "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4",
        "actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=4",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.distributed_executor_backend=uni",
        "reward.num_workers=1",
        "reward.reward_model.enable=false",
        "reward.reward_model.enable_resource_pool=true",
        "reward.reward_model.nnodes=1",
        "reward.reward_model.n_gpus_per_node=1",
        "reward.reward_manager.name=MultiVisualRewardManager",
        f"reward.streaming.enabled={str(not args.batch).lower()}",
        "reward.streaming.max_inflight=2",
        "+reward.models.pickscore.backend=native",
        "+reward.models.pickscore.offload=false",
        f"+reward.models.pickscore.model_path={args.pickscore_model.resolve()}",
        "+reward.models.pickscore.placement.devices=[0]",
        "+reward.models.pickscore.executor.model=verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel",
        f"+reward.models.pickscore.executor.kwargs.processor_path={args.processor.resolve()}",
        "+reward.reward_functions.pickscore.path=pkg://verl_omni.utils.reward_score.pickscore_reward",
        "+reward.reward_functions.pickscore.name=compute_score_pickscore_native",
        "+reward.reward_functions.pickscore.weight=0.7",
        "+reward.reward_functions.pickscore.required=true",
        "+reward.reward_functions.jpeg.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility",
        "+reward.reward_functions.jpeg.name=compute_score",
        "+reward.reward_functions.jpeg.weight=0.3",
        "+reward.reward_functions.jpeg.required=true",
        "trainer.logger=[console]",
        "trainer.project_name=verl-test",
        "trainer.experiment_name=named-streaming",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_training_steps=2",
        "trainer.total_epochs=3",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.val_before_train=false",
        "trainer.log_val_generations=0",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={output / 'checkpoints'}",
        f"trainer.validation_data_dir={output / 'validation'}",
        f"trainer.rollout_data_dir={output / 'rollouts'}",
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "ray_kwargs.ray_init.num_cpus=24",
        "++ray_kwargs.ray_init.object_store_memory=1073741824",
        "++ray_kwargs.ray_init.include_dashboard=false",
    ]
    if args.agent_loop_manager:
        overrides.append(f"+actor_rollout_ref.rollout.agent.agent_loop_manager_class={args.agent_loop_manager}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=overrides)
    OmegaConf.save(config, output / "config.yaml")
    (output / "overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    try:
        run_diffusion_v1(config)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
