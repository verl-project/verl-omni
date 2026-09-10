# MiniCPM-o 4.5 simplex thinker OPD

Last updated: 09/09/2026.

This recipe implements the first, thinker-stage part of [RFC #565](https://github.com/verl-project/verl-omni/issues/565), under [#345](https://github.com/verl-project/verl-omni/issues/345). The student generates a complete text response; a frozen teacher scores the same tokens and the existing verl reverse-KL policy-gradient loss updates the student.

## Supported scope

- MiniCPM-o **4.5**, text output from the thinker stage.
- Text, images, and at most one audio clip per prompt; images and audio can be combined.
- FSDP2, padded per-sample actor sequences, LLM LoRA, and merged-weight rollout synchronization.
- Selected-token reverse-KL OPD with a separate teacher resource pool.

Video, multiple audio clips per prompt, talker/codec-policy training, and streaming duplex are not supported by this recipe. Unsupported inputs/stages raise errors rather than falling back to text-only training. There is no generated speech in the thinker-only topology. Speech-policy OPD is a separate RFC milestone; streaming duplex is tracked by [#566](https://github.com/verl-project/verl-omni/issues/566).

## Weights and dependencies

Use the repository's [installation instructions](../../../docs/start/install.md) and pinned vLLM-Omni revision. The actor uses the released MiniCPM remote model code with a targeted Transformers 5 initialization/Whisper compatibility adapter. Review the remote code before enabling `trust_remote_code`.

```bash
hf download openbmb/MiniCPM-o-4_5 \
  --revision 503e754207c94da6bb26850b4469f367c9ea3582 \
  --local-dir "$HOME/models/MiniCPM-o-4_5"
```

Use the original checkpoint as the frozen teacher. To reproduce the noise-perturbed-student setup from [Qwen3-Omni OPD PR #375](https://github.com/verl-project/verl-omni/pull/375) with 20% noise, create a separate student checkpoint:

```bash
python examples/opd_trainer/minicpm_o/prepare_noised_student.py \
  --source-model "$HOME/models/MiniCPM-o-4_5" \
  --output-model "$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
  --noise-ratio 0.20 \
  --seed 42
```

The script copies the complete checkpoint but perturbs only nonzero floating-point `llm.*` tensors. For every perturbed tensor it samples a Gaussian direction and rescales it so that `||ΔW||₂ / ||W||₂ = 0.20`, within the checkpoint dtype's rounding precision. The frozen vision/audio encoders, processor, tokenizer, and other model components remain identical to the teacher. The output contains `noise_manifest.json` recording the scope, ratio, and seed. The destination must not already exist.

Teacher and student must retain the same tokenizer and special-token mapping; vocabulary compatibility is checked before worker allocation. Using an identical unperturbed checkpoint on both sides is useful for a consistency smoke test, **not evidence of useful distillation or quality improvement**.

Do not downgrade the repository's Transformers version to match the model card's older reference environment. Use Python 3.12 with FlashInfer 0.6.16.post3: its communication module evaluates `array.array[int]`, which fails during worker initialization on Python 3.11.

## Data

Create `train.jsonl` and `test.jsonl` in an input directory. Text-only rows need only a `prompt` string:

```json
{"prompt":"Explain why the sky appears blue in two sentences."}
```

Optional image/audio paths are relative to the JSONL file:

```json
{"prompt":"Answer the spoken question about this scene.","images":["scene.png"],"audios":["question.wav"]}
```

A prompt may also be a message list with string content. In that case, include one `<image>` or `<audio>` placeholder for each media item, in the intended order. Audio files are decoded to mono and resampled to 16 kHz. Images are embedded in Parquet; audio paths are resolved to absolute paths, so the referenced audio must remain accessible on each worker node.

```bash
python examples/opd_trainer/minicpm_o/prepare_data.py \
  --input-dir "$HOME/data/minicpm_source" \
  --output-dir "$HOME/data/minicpm_simplex"
```

The converter writes `train.parquet` and `test.parquet`; it does not generate teacher answers or preference pairs. Keep held-out examples separate, and check source-media rights and speaker consent.

### MMK12 image-math data

The Qwen3-Omni OPD example uses [MMK12](https://huggingface.co/datasets/FanqingM/MMK12), an image-to-text K12 math dataset. MiniCPM reuses the same canonical converter and reward contract; it does not need a model-specific copy of either one. After downloading raw `train-*.parquet` and `test-*.parquet` shards, run:

```bash
bash examples/opd_trainer/minicpm_o/prepare_mmk12_data.sh \
  /path/to/raw/mmk12 \
  "$HOME/data/mmk12"
```

The output embeds one image per row and includes the ground-truth answer, parsed choice options, and an explicit `<think>...<answer>\boxed{...}</answer>` response instruction. Install the rule-based scorer dependency with `uv pip install math-verify`; no learned reward-model checkpoint is needed.

## Training

```bash
STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
TEACHER_MODEL="$HOME/models/MiniCPM-o-4_5" \
DATA_DIR="$HOME/data/minicpm_simplex" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora.sh
```

Defaults use four student GPUs and four teacher GPUs, TP=2 for each rollout/teacher replica, prompt/response budgets 1024/512, rank-32 LoRA, temperature 1, and two samples per prompt. Override `STUDENT_GPUS`, `TEACHER_GPUS`, `ROLLOUT_TP`, `TEACHER_TP`, `PROMPT_LENGTH`, or `RESPONSE_LENGTH` through the environment. Extra CLI arguments override the script's defaults.

For MMK12, use the dedicated wrapper:

```bash
STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
TEACHER_MODEL="$HOME/models/MiniCPM-o-4_5" \
DATA_DIR="$HOME/data/mmk12" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora_mmk12.sh
```

This wrapper changes the base objective from teacher-only OPD to **task reward plus OPD** by enabling `distillation.distillation_loss.use_task_rewards`. It selects `mmk12_reward.py`, whose normalized score combines answer correctness from `math_verify` with progressive `<answer>` and `\boxed{}` format credit. `REWARD_FUNCTION_PATH` can override the scorer path. Validation runs every ten steps by default, without validation before training.

Important settings:

| Setting | Requirement |
| --- | --- |
| `model.model_stage` | `thinker`; native serving stage 0 is named `llm` |
| `model.use_remove_padding` | `false`, preserving per-sample image/audio bounds |
| `model.lora.merge` | `true`; no native MiniCPM LoRA-manager patch is installed |
| `rollout.agent.default_agent_loop` | `minicpm_simplex_agent` |
| `rollout.agent.agent_loop_manager_class` | `verl_omni.pipelines.minicpm.agent_loop.MiniCPMAgentLoopManager` |
| `engine_kwargs.vllm_omni.async_chunk` | `false`; collect the complete bounded response |
| `distillation.distillation_loss` | `loss_mode=kl`, `use_policy_gradient=true`; the base recipe uses no task reward, while the MMK12 wrapper enables it |
| Teacher context | Full student prompt + response + one scoring token; configured by verl |

The recipe freezes vision/audio encoders and their projection modules and restricts LoRA to the LLM. `actor.freeze_vision_tower=false` avoids a Qwen-specific engine freeze path; the MiniCPM adapter owns encoder freezing.

In the base recipe, `use_task_rewards=false` skips training-time task reward while retaining teacher scoring. The MMK12 wrapper enables both signals and configures its rule-based scorer.

Both `data.train_batch_size × rollout.n` and `actor.ppo_mini_batch_size × rollout.n` must be divisible by the actor data-parallel size. Use microbatch 1 for a small smoke test; for four actor ranks and `rollout.n=2`, a PPO minibatch of 2 gives one sample per rank. The base recipe disables validation-before-training and periodic task-reward validation; the MMK12 wrapper enables periodic validation every ten steps.

## Replay and verification

The agent snapshots native processor outputs before rollout. Serving receives the source prompt token IDs; actor replay receives the matching media-expanded IDs. Teacher requests append the student's response IDs without decode/re-tokenize. The adapter checks the actual rollout prefix and the teacher's next-token IDs, including verl's final dummy scoring row.

Frozen encoder embeddings remain buffers under their original checkpoint names, avoiding conditional FSDP collectives and direct reads of sharded Whisper positional weights. Vision insertion is out of place so PEFT input-gradient hooks remain valid. Teacher fields are resized in synthetic zero-loss padding samples.

CPU contracts are in:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q --asyncio-mode=auto \
  tests/pipelines/test_minicpm_simplex_on_cpu.py \
  tests/trainer/omni/test_omni_distillation_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_omni_teacher_on_cpu.py
```

For real-weight validation, check complete rollout → teacher scoring → actor update → weight sync → fresh rollout over multiple steps, and repeat with image/audio prompts. Token equality alone does not establish probability parity or generation quality. Monitor distillation loss, gradients, teacher coverage, and post-sync behavior; missing/misaligned teacher or replay fields must fail the sample.

The shared omni teacher plumbing follows [#375](https://github.com/verl-project/verl-omni/pull/375), and the MiniCPM loading/Whisper compatibility overlaps the model foundation in [#550](https://github.com/verl-project/verl-omni/pull/550); neither requires a separate trainer or a forked distillation objective.
