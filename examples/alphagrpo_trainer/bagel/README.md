# BAGEL DVReward Training

[DVReward](https://github.com/huangrh99/AlphaGRPO) scores generated images against
precomputed semantic and quality questions using a frozen vision-language judge.
The BAGEL recipe uses image FlowGRPO and Qwen3-VL-30B-A3B-Instruct.

## Data and training

Use the official `alphagrpo20k` train/test JSONL files or your own question-decomposed
prompts. Each record contains `prompt`, `semantic_questions`, and `quality_questions`.
Questions can be strings or dictionaries with a `question` field. Obtain the
official files with Git LFS; LFS pointer files are not datasets.

From the repository root:

```bash
python examples/flowgrpo_trainer/data_process/bagel_dvreward.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --input_dir /path/to/alphagrpo20k \
  --output_dir /path/to/dvreward

DATA_DIR=/path/to/dvreward \
  bash examples/alphagrpo_trainer/bagel/run_bagel_dvreward_lora.sh \
  actor_rollout_ref.model.path=/path/to/BAGEL-7B-MoT \
  actor_rollout_ref.model.tokenizer_path=/path/to/BAGEL-7B-MoT
```

A smaller judge can be selected with `reward.reward_model.model_path=...`;
this changes the reward model from the official setup.

For each question, the scorer uses first-token top-5 log-probabilities to compute
`P(yes) / (P(yes) + P(no))`. Semantic and quality questions are averaged separately,
then combined as `sqrt(semantic_score * quality_score)`. Both component scores are
returned as metrics. HTTP failures propagate; they are not converted into zero rewards.

## Gotchas

- The recipe requests `image_preview`; DVReward selects the named decoded image
  using `preview_artifact`, even when the primary output is a latent. Missing,
  latent or non-image previews fail before scoring. Direct tensor calls require
  `extra_info["media_kind"] = "image"`.
- Both question groups must be non-empty. This avoids assigning perfect scores
  to absent questions. Question records are scored in order, without filtering
  `is_valid`, matching the official scorer's behavior.
- Prompts exceeding the token limit are rejected rather than truncated, so the
  image condition still matches the questions used to score it.
- Unlike the reference's JPEG transport, this scorer uses the repository's PNG
  data-URI helper. When neither yes nor no appears in top-5 tokens, a literal
  yes/no answer is used; any other answer is rejected rather than treated as no.
- Native BAGEL records log-probs only for noisy SDE steps. Window slicing applies
  to the full latent and timestep trajectories; slicing log-probs again drops
  training steps when the window starts after zero.
- This recipe uses the existing image-only FlowGRPO adapter. It does not train
  thinking actions or implement AlphaGRPO's joint objective.

## AlphaGRPO

For joint thinking/image training and self-reflective refinement, see
[BAGEL AlphaGRPO](../README.md).
