export LD_LIBRARY_PATH=${CONDA_PREFIX}/cuda-compat:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
# export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
export VLLM_USE_DEEP_GEMM=0
export VERL_LOGGING_LEVEL=INFO


# bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh > train.log 2>&1
# bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_sp2.sh > train.log 2>&1
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh > train.log 2>&1
# bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_one_step_off.sh > train.log 2>&1
# bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_fsdp2_fa3.sh > train.log 2>&1
# bash examples/grpoguard_trainer/run_qwen_image_ocr_lora.sh > train.log 2>&1
# bash examples/mixgrpo_trainer/run_qwen_image_ocr_lora_mixgrpo.sh > train.log 2>&1
# bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_rollout_corr.sh > train.log 2>&1
# bash examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh > train.log 2>&1
