#!/usr/bin/env bash
# Offline-built tiny SD3: CRD rollout/training, independent EMA, and FSDP2 checkpoint regression.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
NUM_GPUS=${NUM_GPUS:-1}
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer." >&2
    exit 1
fi
MODEL_PATH=${MODEL_PATH:-"$HOME/models/tiny-random/stable-diffusion-3-tiny-random"}
smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/crd-e2e.XXXXXXXX")
max_prompt_length=64
python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "$MODEL_PATH"
python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "$smoke_dir/data" --train_size "$((4 * NUM_GPUS))" --val_size 2
custom_chat_template="{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
python3 -m verl_omni.trainer.main_diffusion_v1 \
    "data.train_files=$smoke_dir/data/train.parquet" \
    "data.val_files=$smoke_dir/data/test.parquet" \
    data.train_batch_size="$NUM_GPUS" data.max_prompt_length="$max_prompt_length" \
    algorithm.trainer_type=direct_preference algorithm.sample_source=online \
    algorithm.global_std=True algorithm.train_timestep_range='[0,3]' algorithm.train_timestep_count=2 \
    algorithm.old_policy_decay=0.2 algorithm.reference_policy_decay=0.7 \
    actor_rollout_ref.model.algorithm=crd actor_rollout_ref.model.model_type=diffusion_nft_model \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.model.custom_chat_template=\"$custom_chat_template\"" \
    "actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: $max_prompt_length}}" \
    actor_rollout_ref.model.attn_backend=native actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.model.lora_rank=8 actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.model.policy_state_adapters='["default","old","moving_reference"]' \
    actor_rollout_ref.model.pipeline.reference_guidance_scale=4.5 \
    actor_rollout_ref.actor.diffusion_loss.crd_beta=1.0 actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-4 actor_rollout_ref.actor.optim.total_training_steps=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$NUM_GPUS" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm_omni actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.agent.num_workers=1 actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.pipeline.height=128 actor_rollout_ref.rollout.pipeline.width=128 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length="$max_prompt_length" \
    actor_rollout_ref.rollout.max_prompt_embed_length="$((77 + max_prompt_length))" \
    actor_rollout_ref.rollout.algo.noise_level=0.0 actor_rollout_ref.rollout.enforce_eager=True \
    reward.num_workers=1 reward.reward_model.enable=False \
    reward.custom_reward_function.path=tests/special_e2e/crd_smoke_reward.py \
    trainer.logger=console trainer.project_name=verl-test trainer.experiment_name=crd-sd3-e2e \
    trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 \
    "trainer.default_local_dir=$smoke_dir/trainer" \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=1 \
    trainer.resume_mode=disable trainer.total_epochs=4 trainer.total_training_steps=2 \
    trainer.use_v1=True trainer.v1.trainer_mode=sync trainer.v1.sampler.drop_incomplete_groups=True \
    "$@" 2>&1 | tee "$smoke_dir/train.log"
python3 tests/special_e2e/check_crd_run.py --log "$smoke_dir/train.log" \
    --output-dir "$smoke_dir/trainer" --steps 1 2
python3 -m torch.distributed.run --standalone --nproc-per-node="$NUM_GPUS" \
    tests/workers/crd_fsdp_checkpoint.py --model "$MODEL_PATH" --checkpoint-dir "$smoke_dir/engine"
echo "CRD end-to-end checks passed; artifacts: $smoke_dir"
