(flowgrpo_quickstart)=
# Quickstart: FlowGRPO training on Qwen-Image OCR dataset

Last updated: 04/23/2026

Post-train a diffusion image generation model with FlowGRPO.

## Introduction

In this example, we post-train a `Qwen-Image` policy with FlowGRPO for OCR-style image generation tasks. The rollout uses `vllm-omni` for multimodal generation, and the reward is computed by a visual generative reward model (*Qwen3-VL-8B-Instruct* in this example) that compares OCR text extracted from generated images against the dataset ground truth.

## Prerequisite

- Install VeRL-Omni and its dependencies following the {doc}`installation guide <install>`. Also install the FlowGRPO-specific reward dependency:

```bash
pip install Levenshtein
```

- Use a machine with `4` GPUs for the provided example script.
- Run the commands below from the repository root.

## Dataset Introduction

We use the OCR dataset from the original Flow-GRPO repository: [dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr). Each sample asks the model to generate an image that contains specific text, and the reward model scores the generated image by reading the rendered text and comparing it with the reference OCR string.

The raw dataset is a plain-text file (`train.txt` / `test.txt`) where each line is one generation prompt. The OCR target — the text the model must render in the image — is enclosed in double quotes within the prompt. A few representative samples:

```text
A close-up of a medicine bottle with a clear, red warning label that reads "Take With Food" prominently displayed, set against a neutral background.
A close-up of a robot's chest panel, with a digital display blinking "System Override Active" in red, set against a dimly lit industrial background.
A detailed textbook diagram labeled "Photosynthesis Process", viewed under a high-powered microscope, showcasing the intricate cellular structures and chemical reactions involved.
An ancient, leather-bound wizard's spellbook lies open, revealing a worn, yellowed page. A delicate bookmark rests precisely on "Page 666", casting a subtle glow that illuminates the arcane text.
An astronaut's boot print on the Martian surface, clearly reading "First Steps", surrounded by the red, dusty terrain under a pale, distant sky.
```

The preprocessing script converts the raw dataset into parquet files that contain:

- the multimodal prompt used for image generation,
- a negative prompt for true CFG sampling,
- OCR ground truth stored under `reward_model.ground_truth`,
- auxiliary metadata such as split and sample index.

## Step 1: Prepare the dataset

Set the `WORKSPACE` environment variable to any writable directory you prefer (defaults to `$HOME` if unset):

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
```

Obtain the raw OCR dataset from the original Flow-GRPO repository and place it under `$WORKSPACE/data/ocr`. Then preprocess it into `train.parquet` and `test.parquet`:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

The command above writes:

- `$WORKSPACE/data/ocr/train.parquet`
- `$WORKSPACE/data/ocr/test.parquet`

These parquet files are the inputs consumed by the FlowGRPO training script.

### Preparing a custom dataset

To train on your own OCR-style data, create `train.txt` and `test.txt` following the same one-prompt-per-line convention. Each prompt must contain the target OCR string enclosed in double quotes — the preprocessing script extracts the text between the first pair of quotes as the ground truth. For example:

```text
A vintage storefront sign above the door reads "Open 24 Hours" in bold neon letters.
A handwritten sticky note on a refrigerator says "Buy milk" in blue ink.
```

Place the files in `$WORKSPACE/data/ocr/` (or any directory you prefer) and run the same preprocessing command, adjusting `--input_dir` and `--output_dir` as needed:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

For datasets with a different ground-truth extraction scheme (e.g. a CSV with an explicit label column), modify `extract_solution` and the `process_fn` function in `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` to match your format, then re-run the script to regenerate the parquet files.

## Step 2: Obtain models for RL training

In this example, we train `Qwen/Qwen-Image` with LoRA and use `Qwen/Qwen3-VL-8B-Instruct` as the OCR reward model.

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_path` variable in the script directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required — Hugging Face will cache it automatically on first run. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

The run script exposes the following environment variable:

```bash
WORKSPACE              # base directory for data (default: $HOME)
```

## Step 3: Perform FlowGRPO training

The provided example script launches `python3 -m verl_omni.trainer.main_flowgrpo` with the FlowGRPO-specific config needed for this OCR task:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `reward.reward_manager.name=visual`
- `reward.custom_reward_function.name=compute_score_ocr`
- LoRA fine-tuning on `Qwen-Image`
- a single-node, `4`-GPU layout

Run the training script:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

Optional KL loss tuning:

- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`

The script uses `$WORKSPACE` (default: `$HOME`) as the base directory. Override any path via the environment variables described in Step 2, or set `WORKSPACE` to point to a volume with enough free space before launching.

You are expected to see training, validation, actor, critic, and reward metrics logged through the configured backends. By default, checkpoints are saved under:

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

## Wandb logging

The provided script already enables:

```bash
trainer.logger='["console", "wandb"]' \
trainer.project_name=flow_grpo \
trainer.experiment_name=qwen_image_ocr_lora
```

Set your W&B credentials before launching if you want remote tracking:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

You can also override `trainer.project_name` and `trainer.experiment_name` from the command line to organize runs under your own project names.

## Diffusion-specific metrics

The following metrics are specific to diffusion FlowGRPO training.

**critic/rewards/zero_std_ratio** — the fraction of prompt groups (out of
`train_batch_size` prompts) where every one of the `n` generated images
received the same reward, giving a within-group standard deviation of zero.
GRPO derives its learning signal from *relative* rewards within a group, so a
group with zero std contributes no gradient regardless of the absolute reward
value. A persistently high ratio (e.g. above 0.5) means the reward model is
saturated or the task difficulty is poorly calibrated — either all images are
rewarded or none are — and the policy is not receiving useful training signal.

**critic/rewards/std_mean** — the mean of the per-prompt reward standard
deviations across all prompt groups in the batch. Complements
`zero_std_ratio`: while `zero_std_ratio` flags completely collapsed groups,
`std_mean` tracks the average spread of rewards within a group across the
whole batch. A healthy, rising `std_mean` indicates the reward model is
producing diverse signal; a declining `std_mean` is an early warning of
reward saturation before `zero_std_ratio` spikes.

**actor/pg_clipfrac_higher** and **actor/pg_clipfrac_lower** — these break
down PPO clipping by direction. `pg_clipfrac_higher` is the fraction of
`(image, denoising-timestep)` pairs where the probability ratio
`π_new / π_old` exceeded `1 + clip_ratio`, meaning the policy is trying
to increase the probability of high-advantage images more than the clip
allows. `pg_clipfrac_lower` is the fraction where the ratio fell below
`1 - clip_ratio`, meaning the policy is suppressing low-advantage images
more aggressively than allowed. A large asymmetry between the two (e.g.
`higher` >> `lower`) indicates the dominant learning direction and can
guide tuning of `clip_ratio` or the learning rate.

**timing_per_image_ms/{stage}** — per-image latency in milliseconds for each
core compute stage: `gen` (rollout), `ref` (reference log-prob), `old_log_prob`,
`adv` (advantage computation), and `update_actor`. Use these to pinpoint which stage
dominates step time and where to focus optimisation effort.

**perf/throughput** — images processed per second per GPU, computed as
`(train_batch_size × rollout.n) / (time_per_step × n_gpus)`.

## Further reading

For the algorithm background, detailed configuration notes, async reward, and rule-based
reward training (e.g. JPEG incompressibility), see:

- {doc}`../algo/flowgrpo`
