#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker on-policy distillation (OPD): the GSPO AVQA recipe
# plus a frozen vllm_omni teacher pool that scores every student rollout.
#
# DATASET selects the data and reward (default avqa):
#   DATASET=avqa   audio/video QA with the AVQA recipe's own data and reward.
#   DATASET=mmk12  image + text -> text with the rule-based mmk12_reward.py, as in
#                  the Qwen3-Omni OPD recipe. Prepare the data once:
#                    pip install math-verify    # required by mmk12_reward.py
#                    python examples/gspo_trainer/data_process/mmk12.py \
#                        --local_dataset_path <path_to_raw_mmk12> \
#                        --local_save_dir ~/data/mmk12
#                  TRAIN_FILE, VAL_FILE and REWARD_FUNCTION_PATH override the defaults.
#   DATASET=gsm8k  text-only math scored by verl's built-in gsm8k reward —
#                  the verification task: the thinker's initial accuracy is
#                  high, so reward curves are interpretable from the first
#                  steps. Prepare the data once (no extra deps):
#                    python examples/gspo_trainer/data_process/gsm8k.py \
#                        --local_save_dir ~/data/gsm8k
#                  TRAIN_FILE, VAL_FILE override the defaults.
#
# Student data, actor, and rollout settings come from
# examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh
# (prepare its AVQA data first); this wrapper selects the student checkpoint and
# adds the distillation group, as the Qwen3-Omni OPD recipe does for its GSPO base.
# The teacher must share the student's tokenizer and multimodal preprocessing.
#
# Defaults: four student GPUs (from the base recipe) plus TEACHER_GPUS teacher GPUs
# on the same node. Teacher prompt/response lengths must match the student's.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a MiniCPM-o 4.5 checkpoint}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a frozen checkpoint sharing the student tokenizer}"

DATASET=${DATASET:-avqa}
case "${DATASET}" in
    avqa)
        DATASET_ARGS=()
        EXPERIMENT_NAME=minicpm_o_4_5_thinker_lora_avqa_opd
        ;;
    mmk12)
        export TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
        export VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}
        REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"${REPO_ROOT}/verl_omni/utils/reward_score/mmk12_reward.py"}
        DATASET_ARGS=(
            reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}"
            reward.custom_reward_function.name=compute_score
        )
        EXPERIMENT_NAME=minicpm_o_4_5_thinker_lora_mmk12_opd
        ;;
    gsm8k)
        export TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k/train.parquet"}
        export VAL_FILE=${VAL_FILE:-"$HOME/data/gsm8k/test.parquet"}
        # data_source="openai/gsm8k" dispatches to verl's built-in gsm8k reward;
        # null out the AVQA base's choice reward so the dispatch takes over.
        # The OPD path runs unpacked (use_remove_padding=false below), so the
        # log-prob/update passes materialize dense bsz x response_len x vocab
        # tensors: the base recipe's 20480-token chunks and AVQA windows OOM
        # GSM8K batches (~30 GiB single softmaxes, vocab 151748). GSM8K prompts
        # are <=~280 tokens with template and CoT answers < ~400, so sized
        # windows and 4096-token chunks keep transients under ~7 GiB. Other
        # short-task datasets need the same sizing (see README).
        DATASET_ARGS=(
            reward.custom_reward_function.path=null
            data.max_prompt_length=512
            data.max_response_length=1024
            actor_rollout_ref.rollout.prompt_length=640
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
            distillation.teacher_models.teacher_model.inference.prompt_length=640
            distillation.teacher_models.teacher_model.inference.response_length=1024
            distillation.teacher_models.teacher_model.inference.max_model_len=2048
        )
        EXPERIMENT_NAME=minicpm_o_4_5_thinker_lora_gsm8k_opd
        ;;
    *)
        echo "DATASET must be avqa, mmk12, or gsm8k, got '${DATASET}'." >&2
        exit 1
        ;;
esac

TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-2}
if (( TEACHER_GPUS <= 0 || TEACHER_TP <= 0 )) || (( TEACHER_GPUS % TEACHER_TP != 0 )); then
    echo "TEACHER_GPUS (${TEACHER_GPUS}) must be a positive multiple of TEACHER_TP (${TEACHER_TP})." >&2
    exit 1
fi

MODEL_PATH="${STUDENT_MODEL}" exec bash "${REPO_ROOT}/examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh" \
    actor_rollout_ref.model.use_remove_padding=false \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm_omni \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
    distillation.teacher_models.teacher_model.inference.max_model_len=16640 \
    distillation.teacher_models.teacher_model.inference.prompt_length=4160 \
    distillation.teacher_models.teacher_model.inference.response_length=12288 \
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=false \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.output_mode="ar" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.trust_remote_code=true \
    distillation.distillation_loss.loss_mode=kl \
    distillation.distillation_loss.use_policy_gradient=true \
    trainer.project_name=opd \
    ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "$@"
