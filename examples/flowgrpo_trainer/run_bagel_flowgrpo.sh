# Bagel LoRA RL, vllm_omni rollout (FlowGRPO)
#
# Prerequisites:
#   1. A Bagel model (e.g. BAGEL-7B-MoT) at $BAGEL_MODEL_PATH
#   2. A vllm-omni deploy-config YAML at $BAGEL_DEPLOY_CONFIG (we ship one
#      next to this script at ``bagel_deploy_config.yaml``)
#   3. ``BagelDiffusion`` registered as ``OmniBagelForConditionalGeneration``
#      via ``verl_omni.pipelines.bagel_flow_grpo`` (auto-imported)
#   4. A reward VLM model at $REWARD_MODEL_PATH
#   5. OCR training data at $OCR_TRAIN_PATH / $OCR_TEST_PATH
#      (generate via: ``python examples/flowgrpo_trainer/data_process/qwenimage_ocr.py``)
#
# Usage:
#   export BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT
#   export REWARD_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo.sh
#
#   # Override any param via CLI:
#   bash examples/flowgrpo_trainer/run_bagel_flowgrpo.sh trainer.n_gpus_per_node=8
#
# Default uses 4 GPUs with ``tensor_model_parallel_size=1`` (4 single-GPU
# rollout replicas) to mirror the Qwen-Image recipe. Be aware of a TOCTOU
# race in vllm-omni's per-actor ``MASTER_PORT`` picker (``settle_port`` in
# ``vllm_omni/diffusion/data.py``): every concurrent ``vLLMOmniHttpServer``
# Ray actor independently calls ``is_port_available(p)`` and may pick the
# same port before any of them ``bind()``s, with collision probability
# scaling by the number of concurrent actors (~4% at 4, ~18% at 8 in the
# default 100-port window) and amplified further when retries land inside
# the prior run's TIME_WAIT window (≈60s). If a launch dies during
# ``init_distributed_environment`` with ``EADDRINUSE`` on a port in
# 30005-30105, wait ~60s and re-launch; the upstream bug is tracked at
# vllm-project/vllm-omni#TBD.

set -x

# --------------- Paths (override via environment) ---------------
BAGEL_MODEL_PATH=${BAGEL_MODEL_PATH:-$HOME/models/BAGEL-7B-MoT}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-$SCRIPT_DIR/bagel_deploy_config.yaml}

REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$HOME/models/Qwen3-VL-8B-Instruct}

ocr_train_path=${OCR_TRAIN_PATH:-$HOME/data/ocr/train.parquet}
ocr_test_path=${OCR_TEST_PATH:-$HOME/data/ocr/test.parquet}

ENGINE=vllm_omni
REWARD_ENGINE=vllm

reward_path=verl_omni/utils/reward_score/genrm_ocr.py

python3 -m verl_omni.trainer.diffusion.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$BAGEL_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$BAGEL_MODEL_PATH \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.pipeline.height=512 \
    actor_rollout_ref.model.pipeline.width=512 \
    actor_rollout_ref.model.pipeline.num_inference_steps=15 \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=15 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=15 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.num_workers=1 \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$REWARD_MODEL_PATH \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=4 \
    +reward.reward_model.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    reward.custom_reward_function.path=$reward_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=bagel_ocr_lora \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=50 "$@"
