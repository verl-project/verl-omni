# Lance FlowGRPO training

Last updated: 10/24/2026.

[Lance](https://huggingface.co/bytedance-research/Lance) is a 3B unified
multimodal model that covers image and video understanding, generation and
editing in one autoregressive + diffusion model.  It is BAGEL-lineage:
The integration reuses BAGEL's MoT training module and adapts Lance's
checkpoint layout, latent geometry, sigma schedule and rotary positions.

M1 covers text-to-image LoRA training on the generation expert. M2 adds
text-to-video. Conditioning is a separate draft milestone of
[RFC #222](https://github.com/verl-project/verl-omni/issues/222).

## What differs from BAGEL

| Aspect | BAGEL-7B-MoT | Lance_3B |
|---|---|---|
| Transformer | Qwen2-MoT | identical (inherited) |
| Latent | BAGEL AE, 16 channels, 8x downsample, 2x2 latent patch | Wan2.2 VAE, 48 channels, 16x downsample, no latent patch |
| `vae2llm` | `(hidden, 64)` | `(2048, 48)` |
| `latent_pos_embed` | `(max_latent_size**2, hidden)`, 32 grid | `(4096, 2048)`, 64 grid |
| Sigma shift | 3.0 | **3.5** |
| Config | `config.json` with `llm_config` / `vae_config` | `Lance_3B/llm_config.json` plus constants |
| Checkpoint | `ema.safetensors` | `Lance_3B/model.safetensors` |
| RoPE | 1-D, one position for the whole image block | mRoPE, per-token `(t, h, w)` over the latent grid |

The last row is the one that reaches furthest.  BAGEL gives the image block a
single position and lets `latent_pos_embed` carry the layout; Lance keeps
`rope_scaling` on the language model and `LanceBagel.prepare_vae_latent` gives
latent `(hi, wi)` the position `(P + 1, P + 1 + hi, P + 1 + wi)`.  The trainer
has to build the same positions and the same rotary basis, or the log-probs it
recomputes are taken under a different basis than the rollout used and the
importance ratio is not 1 even before the first update.
`tests/pipelines/test_lance_flowgrpo_on_cpu.py` compares both against the
rollout's own `_per_token_mrope_for_vae_latent` and `BagelRotaryEmbedding`.

## Prerequisites

- Install VeRL-Omni (see the [installation guide](../../../docs/start/install.md)).

- 4 GPUs. Run commands from the repository root.

  "3B" counts one expert: `llm_config.json` gives 36 layers at hidden 2048 with
  an 11008 MLP, and the MoT carries a text and a generation copy of every
  layer, so the trainer holds roughly 5.9B parameters (about 12 GB in bf16)
  and the rollout holds its own copy.

- Download the checkpoint bundle:

  ```bash
  huggingface-cli download bytedance-research/Lance --local-dir ~/models/bytedance-research/Lance
  ```

  Point `model.path` at the bundle root; the adapters resolve the `Lance_3B`
  subdirectory themselves.  `model.tokenizer_path` is the one path that does
  not get resolved for you, and the released tokenizer needs two repairs: it
  lives inside `Lance_3B/` rather than at the bundle root (a run aimed at the
  root fails on a partial tokenizer `AutoTokenizer` cannot build), and it
  designates neither `eos_token` nor `pad_token`, so the agent loop fails the
  first time it pads a batch of prompts.  The data prep below writes a
  repaired copy next to the parquet and the recipe points there; the released
  checkpoint is left untouched.

## PickScore training

PickScore scores image-text alignment with a
[CLIP-based model](https://huggingface.co/yuvalkirstain/PickScore_v1).  The
reward function lives in `verl_omni/utils/reward_score/pickscore_reward.py`,
so there is no separate reward deployment and the GPU is shared between the
actor and the reward.

### Prepare the dataset

The raw PickScore dataset (`train.txt` / `test.txt`) comes from the
[flow_grpo repository](https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore).

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/lance_pickscore.py \
  --model_path ~/models/bytedance-research/Lance \
  --input_dir ~/data/pickscore \
  --output_dir $WORKSPACE/data/pickscore/lance
```

This produces `$WORKSPACE/data/pickscore/lance/train.parquet` and
`test.parquet`.

The run script filters captions whose complete chat-formatted prompt exceeds
256 tokens, using the same template as the rollout. This prevents the actor's
precomputed token prefix from being paired with a left-truncated rollout
prompt. Existing parquet files can be reused with the default 256-token data
prep setting. If changing the limit, regenerate the data with the same
`--max_prompt_length` and update `data.max_prompt_length` in the run command.
Keep the template's literal newlines: a literal backslash followed by `n`
becomes prompt text and also turns the blank negative prompt into a nonempty
condition.

### Run LoRA training

```bash
bash examples/flowgrpo_trainer/lance/run_lance_pickscore_lora.sh
```

The recipe mirrors
[`run_bagel_pickscore_lora.sh`](../bagel/run_bagel_pickscore_lora.sh): LoRA
rank 64 on the `*_moe_gen` projections, `noise_level=1.3`, SDE window `size=2`
over `[0, 7]`, and 15 denoising steps for rollouts.  Validation runs at 30
steps rather than the BAGEL recipe's 50: 50 is what BAGEL's pipeline defaults
to, and Lance's own default is 30 (`LANCE_DEFAULTS.num_timesteps`), so
validation samples the model the way it is meant to be sampled.  The rollout
count stays at 15, which keeps the SDE window over the same fraction of the
trajectory as the BAGEL recipe puts it over.  Otherwise only the checkpoint,
the deploy config and
`+actor_rollout_ref.model.architecture=OmniLanceForConditionalGeneration`
change.

### Checking rollout and trainer agree

FlowGRPO replays the rollout's recorded transitions through the trainer, so
the trainer's log-prob of each transition should match the one the rollout's
SDE sampler produced when it drew that transition.  `old_log_probs` is the
trainer's own recompute, so `actor/ratio_mean` being 1.0 on the first update
only says the update path is deterministic; it does not compare against the
rollout. The recipe enables the comparison with

    actor_rollout_ref.rollout.calculate_log_probs=True

and read `rollout_corr/logprob_abs_diff_mean`, `rollout_corr/logprob_abs_diff_max`
and the per-timestep `rollout_corr/logprob_abs_diff/ts_*` on the first step.
Both sides run the same weights (LoRA is merged before the sync) and skip the
Gaussian normalizer. Numerical differences can still come from precision
and attention backends; do not assume a discrepancy is bf16 noise without
checking the inputs and replay settings. Two differences from BAGEL are the
sigma schedule (Lance shifts by 3.5 and samples one more point) and the
rotary positions (Lance gives the latents per-token `(h, w)`). Each sample's
image positions must start after its own valid text prefix, including in
mixed-length batches. The CPU tests compare batched outputs and gradients
against individual forwards, with and without gradient checkpointing. This
does not replace a GPU rollout/trainer comparison with the released weights.
The per-timestep split shows where discrepancies occur along the grid.

A wider SDE window (for example `sde_window_size=15 sde_window_range='[0,15]'`
at 15 steps) puts every timestep into the comparison.

## Not covered here

- `x2t_image` / `x2t_video`: autoregressive token generation, so they belong to
  the AR/VLM RL path rather than the diffusion FlowGRPO adapter.

## Text-to-video (M2)

Select `Lance_3B_Video` explicitly, then run
`run_lance_t2v_imagebind_lora.sh` or `run_lance_t2v_imagebind_smoke.sh`.
This adds temporal position embeddings and three-axis video RoPE. All decoded
frames and the native flattened latents use the current named-artifact contract.
Video trajectory recording requires [vllm-omni#7907](https://github.com/vllm-project/vllm-omni/pull/7907).
The upstream dependency pin is unchanged. This stage remains a draft until that
dependency and current-version GPU replay validation are available.

Long training, convergence and throughput benchmarks require maintainer resources;
the smoke recipe covers two updates and does not establish convergence.

## Conditioning (M3 draft)

The image-edit, image-to-video and video-edit recipes use a Lance agent loop
that carries reference media beside the text prompt. The adapter maps a video
request's image to `first_frame` and preserves exported conditioning rows and
positions for replay. Shared workers and their request schema are unchanged.

Prepare each dataset with `data_process/lance_{image_edit,i2v,video_edit}.py`,
then use the corresponding `run_lance_*_lora.sh` or two-update smoke script.
Input JSONL fields are documented in those preparation scripts.

M3 depends on M2 and vllm-omni#7907. Its conditioning interface still needs
maintainer agreement. The i2v replay applies timestep zero to reference-frame
tokens. After the full forward and CFG, only unfixed tokens enter the scheduler,
so log-probs and KL means exclude the reference frame. This requires the companion
rollout's active-token SDE sampling fix; the earlier replay metric does not prove
a fixed first frame or validate this corrected objective. i2v remains experimental.
Full training, convergence and current-version GPU validation are not claimed.
Regenerate trajectories and old log-probs with the matching rollout sampler;
recordings from its earlier all-token SDE path are incompatible with this objective.

Each batch must use the same latent shape and number of distinct fixed tokens;
their positions may differ between samples. The active tokens retain their original
order. All-fixed trajectories are rejected because they have no stochastic action.

The edit replay contract assumes the reference VAE block and the generated block
share the same latent grid, as vllm-omni#7907 constructs them. ViT rows attend only
to the prefix and ViT block; VAE rows can additionally attend to the VAE block.
The CPU regression compares those segments with an independent dense attention mask.

Video editing samples the reference at 12 fps in vllm-omni. Its recipe declares
12 fps for emitted video artifacts; text-to-video and i2v retain the configured
frame rate.

Reference media stays raw through the agent loop. Qwen's dataset video decoder
returns tensors plus metadata and can resample frames; Lance instead requires
the original path or decoded PIL frames so its own sampler sees the source clip.
