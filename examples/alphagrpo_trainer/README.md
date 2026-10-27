# BAGEL AlphaGRPO

The experimental single-turn [AlphaGRPO](https://github.com/huangrh99/AlphaGRPO)
GPU recipe trains BAGEL thinking and image generation together. It combines a clipped
sequence-ratio text objective with image FlowGRPO, DVReward and the thinking-tag
format reward. A separate experimental recipe adds the reference two-stage
self-reflective refinement workflow. Its image-conditioned path has CPU checks
only; it has not undergone GPU integration or quality validation.

Earlier **single-turn** native BAGEL smoke runs verified FSDP2 joint LoRA updates and merged weight sync.
Four short runs with four H100 GPUs and the official 30B judge completed at 512px
(two training seeds, eight updates per run). A 16-prompt heldout PickScore comparison
did not establish a stable quality improvement. Inference speed was not measured.

## Training

Use the [DVReward data preparation instructions](../flowgrpo_trainer/bagel/README.md#data-and-training),
then run from the repository root:

```bash
DATA_DIR=/path/to/dvreward \
  bash examples/alphagrpo_trainer/bagel/run_bagel_alphagrpo_lora.sh \
  actor_rollout_ref.model.path=/path/to/BAGEL-7B-MoT \
  actor_rollout_ref.model.tokenizer_path=/path/to/BAGEL-7B-MoT
```

The wrapper trains both understanding and generation LoRA projections with FSDP2.

## Self-reflective refinement

The implementation follows the base `alphagrpo_reflect` workflow at
[AlphaGRPO commit 116d5d5](https://github.com/huangrh99/AlphaGRPO/tree/116d5d52823b8d53e1cf6c86854fb349b63e02f0):
ODE initial generation without thinking, lowest-reward initial-image selection per
prompt, image-conditioned reflection and regeneration, then failure penalty reward
(FPR). A non-improving sample receives its group's minimum **initial** score,
matching `alpha_grpo/tasks/reflect.py`. An entirely non-improving batch, or one with
zero group advantages, skips the actor optimizer update. Text and image use shared
advantages, as in the reference base configuration; reflection-format judge,
isolated text rewards, mixed T2I training and other ablations are not enabled.

```bash
DATA_DIR=/path/to/dvreward REWARD_GPUS=<dedicated-judge-gpu-count> \
  bash examples/alphagrpo_trainer/bagel/run_bagel_alphagrpo_reflect_lora.sh \
  actor_rollout_ref.model.path=/path/to/BAGEL-7B-MoT \
  actor_rollout_ref.model.tokenizer_path=/path/to/BAGEL-7B-MoT
```

This script launches training; it is not a CPU check. `REWARD_GPUS` is required
explicitly and is **additional** to the actor/rollout allocation. The reward model
retains the base DVReward recipe's 30B-A3B judge; selecting a smaller judge changes
the experiment. No GPU experiments have been run for this refinement extension.

The recipe selects 14 refinements per prompt, seven initial candidates, 40
sampling points, five training transitions, LoRA rank 32/alpha 64, learning rate
`5e-5`, global advantage normalization and text-loss weight `0.2`. Refinement uses
the reference text/image CFG scales `4/2`, interval `[0, 1]`, and `text_channel`
renormalization. Initial ODE generation uses `[0.4, 1]` and global renormalization.
The candidate count must divide across agent workers; reduce `agent.num_workers`
when using small batches. The original captions/questions remain the reward targets.

This is an integration of the reference method, not a numerically identical paper
reproduction: it retains the existing full-distribution thinking sampler
(`think_top_p=1`), the existing FlowGRPO sample-standard-deviation normalization,
and the pinned vLLM-Omni image preprocessing. The reference uses top-p 0.8,
population standard deviation, and its own image transforms. The text budget here
is 512 and the total replay budget is 8192. Earlier single-turn GPU checks do not
validate this path.

The native SigLIP processor resizes to 980×980, yielding 4900 visual patches
before image boundaries. The replay budget must also include VAE tokens, the
reflection prompt and thinking tokens, even for a small generated image.

## Refinement replay constraints

- Use the synchronous V0 recipe (`trainer.use_v1=false`). The TQ/V1 scheduler does
  not implement this batch-level selection barrier.
- Reward workers must remain resident during rollout on their own resource pool
  (`reward.reward_model.rollout.free_cache_engine=false`).
  The colocated reward-after-rollout path cannot score the initial stage midway.
- The rollout returns frozen VAE/ViT input embeddings, image-block lengths and
  native text IDs. Training replays both image blocks through the trainable MoT
  layers, using block-causal attention and compressed image RoPE positions.
  It does not reuse detached transformer KV caches as training context.
- Token embeddings, visual encoders/connectors, VAE projection, and time/position
  embeddings stay frozen. LoRA targets remain the two transformer pathways.
  Image features consume extra host/actor memory; no memory or throughput claim
  has been validated on GPUs. Fixed-size initial and refined canvases are required.
- Three CFG branches are replayed separately: image + text + thinking, image
  without the user text, and text without the image. The planning system prompt
  is retained in all three branches.

## Gotchas

- Single-turn AlphaGRPO uses the reference planning system prompt. Its three image contexts
  are system + prompt + thinking, system only, and system + prompt. Replay uses
  exact native token IDs; decoding is used only for reward scoring and logging.
- Thinking requires `think_top_p=1`: sampling and replay use the same
  temperature-scaled distribution. This differs from the reference's truncated
  sampling with full-vocabulary likelihoods. Sequence ratios stay paired with
  each sample's own advantage.
- Rollout correction is unsupported for this joint objective and is rejected.
  Image-step correction weights cannot be reused as text importance weights.
- Only sampled thinking actions enter the text loss. A forced end token after
  reaching the thinking budget conditions the image but is excluded from the
  text loss. The text objective runs once per micro-batch, with image-step-count
  compensation before the shared gradient-accumulation division.
- The single-turn reward adds `1` when thinking starts with `<think>` and ends
  with `</think>`, matching the reference T2I task. `dvreward`, component scores
  and `thinking_format_score` are logged separately.
- Rollout currently requires TP=CFG=SP=1 and full-request execution. Actor
  sequence parallelism and timestep staging are unsupported. Merged LoRA sync
  routes the language head separately from the transformer body.
- `pipeline.max_sequence_length` must hold the system prompt, native prompt and
  `max_think_tokens + 2` framing budget. The default thinking budget is 128;
  this recipe uses 512 with a total length of 2048. Overflow is rejected. Negative prompts
  are unsupported because text CFG uses the planning system context.
- Image-only DVReward uses the original FlowGRPO adapter; enabling its upstream
  `think=true` path would not provide joint text training or matching replay. Use
  `model.algorithm=alphagrpo` and the AlphaGRPO recipe for thinking. This is an
  integration, not evidence of a stable quality or speed improvement.
