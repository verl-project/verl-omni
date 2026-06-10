# Diffusion-DPO

Last updated: 06/05/2026.

Diffusion-DPO
([paper](https://arxiv.org/abs/2311.12908),
[code](https://github.com/SalesforceAIResearch/DiffusionDPO))
adapts Direct Preference Optimization (DPO) to text-to-image diffusion models.
It aligns a diffusion policy to pairwise preferences by comparing how well the
current model and a frozen reference model explain chosen and rejected images
under a forward noising process.

VeRL-Omni supports Diffusion-DPO as a direct-preference algorithm. The default
recipe is **online DPO**: the trainer samples multiple images for each prompt,
scores them with a reward model or reward function, converts the best and worst
samples into a chosen/rejected pair, and updates the actor with the DPO loss.
Offline preference pairs are also supported through the same loss and engine
contract.

## Algorithm

For a prompt $c$, online Diffusion-DPO first samples $K$ images from the
current rollout policy:

$$
x_0^{1:K} \sim \pi_\theta(\cdot \mid c).
$$

A reward function assigns scalar scores $r(x_0^k, c)$. VeRL-Omni builds one
preference pair per prompt by selecting the highest-scoring sample as the
chosen image $x_0^w$ and the lowest-scoring sample as the rejected image
$x_0^l$:

$$
x_0^w = \arg\max_{x_0^k} r(x_0^k, c),
\qquad
x_0^l = \arg\min_{x_0^k} r(x_0^k, c).
$$

The pair is then noised with the same noise $\epsilon$ and timestep $t$:

$$
x_t = (1-\sigma_t)x_0 + \sigma_t \epsilon.
$$

For flow-matching models, the target velocity is:

$$
u(x_0, \epsilon) = \epsilon - x_0.
$$

Diffusion-DPO compares the current model's prediction error against the
reference model's prediction error. Let

$$
\Delta_\theta(x_0)
= \left\|v_\theta(x_t,c,t)-u(x_0,\epsilon)\right\|_2^2
- \left\|v_{\mathrm{ref}}(x_t,c,t)-u(x_0,\epsilon)\right\|_2^2.
$$

The pairwise objective is:

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
= -\mathbb{E}_{(c,x_0^w,x_0^l)}
\log \sigma\left(
-\frac{\beta}{2}
\left[
\Delta_\theta(x_0^w)-\Delta_\theta(x_0^l)
\right]
\right).
$$

Here $\beta$ is the DPO inverse temperature. Larger values make the update more
sensitive to the current-vs-reference error margin between the chosen and
rejected samples.

## How VeRL-Omni Implements Diffusion-DPO

VeRL-Omni runs Diffusion-DPO through the direct-preference trainer.

| Layer | What it does | Code |
|---|---|---|
| Trainer | Runs online rollout, reward scoring, best/worst pair selection, reference prediction, and actor update. | `verl_omni/trainer/diffusion/ray_diffusion_trainer.py` |
| Actor loss | Selects online pairs and computes the pairwise DPO objective from model and reference prediction errors. | `verl_omni/trainer/diffusion/diffusion_algos.py` |
| FSDP engine | Re-noises clean latents with shared pairwise noise/timesteps and performs a one-shot flow-matching forward pass. | `verl_omni/workers/engine/fsdp/diffusers_impl.py` |
| Pairwise utilities | Samples and validates shared noise/timesteps for adjacent chosen/rejected pairs. | `verl_omni/pipelines/utils.py` |
| Qwen-Image adapter | Builds Qwen-Image transformer inputs for DPO training and optional True-CFG inference. | `verl_omni/pipelines/qwen_image_dpo/` |

The online batch layout is important. After rollout and reward scoring,
`DPOLoss.prepare_actor_batch(...)` groups samples by prompt `uid`, sorts each
group by reward, and keeps `[chosen, rejected]` adjacent in the actor batch.
`DPODiffusersFSDPEngine` then samples one shared noise tensor and one shared
timestep per pair, repeats both across the chosen and rejected samples, and
returns:

- `noise_pred`: current actor prediction.
- `noise`: shared pairwise flow noise.
- `latent`: clean latent for the generated image.
- `timesteps`: shared pairwise training timestep.

The trainer computes `ref_noise_pred` with the reference policy before actor
update. The DPO loss consumes `noise_pred`, `ref_noise_pred`, `noise`, `latent`,
and `sample_level_rewards`; it also checks that adjacent pairs share the same
prompt `uid` and that the chosen reward is not lower than the rejected reward.

## Configuration

The reference online Qwen-Image OCR recipe selects Diffusion-DPO with:

```bash
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
algorithm.paired_preference=true
actor_rollout_ref.model.algorithm=dpo
actor_rollout_ref.model.model_type=diffusion_dpo_model
actor_rollout_ref.model.external_lib=verl_omni.pipelines.qwen_image_dpo
actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo
actor_rollout_ref.actor.diffusion_loss.dpo_beta=100.0
actor_rollout_ref.rollout.calculate_log_probs=false
```

### Core Parameters

- `algorithm.trainer_type`: must be `direct_preference` for Diffusion-DPO.
- `algorithm.sample_source`: use `online` for live rollout and reward scoring.
  Use `offline` only when the dataset already contains preference pairs and
  scores.
- `algorithm.paired_preference`: must be `true`; DPO trains on adjacent
  chosen/rejected pairs.
- `actor_rollout_ref.rollout.n`: number of images sampled per prompt before
  online pair selection. It must be at least `2`; the example uses `16`.
- `actor_rollout_ref.actor.diffusion_loss.dpo_beta`: $\beta$ in the pairwise
  DPO objective. The default config value is `2000.0`, while the online
  Qwen-Image OCR recipe uses `100.0`.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: must be an even
  number greater than or equal to `2` when `paired_preference=true`, so a
  chosen/rejected pair is not split across micro batches.
- `actor_rollout_ref.actor.shuffle`: pair-preserving DPO updates require
  unshuffled actor batches; the trainer disables shuffling if needed.
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: controls the
  reference forward micro-batch size used to compute `ref_noise_pred`.
- `actor_rollout_ref.rollout.calculate_log_probs`: should be `false`; DPO does
  not train from reverse-process log probabilities.

## Reference Example

The ready-to-run online DPO example post-trains `Qwen/Qwen-Image` with an OCR
reward model (`Qwen/Qwen3-VL-8B-Instruct`) using `vllm_omni` rollout. It is
configured for one node with 4 GPUs, LoRA rank `64`, rollout group size `16`,
35 inference steps during training rollout, and 300 actor update steps.

First install the OCR reward dependency after setting up the base VeRL-Omni
environment:

```bash
pip install Levenshtein
```

Obtain the raw OCR dataset from the original Flow-GRPO repository
([dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr))
and place it under `$WORKSPACE/data/ocr`, where `WORKSPACE` defaults to
`$HOME` if unset. Then preprocess it into the Qwen-Image parquet files consumed
by the DPO script:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/qwen_image
```

The command writes:

- `$WORKSPACE/data/ocr/qwen_image/train.parquet`
- `$WORKSPACE/data/ocr/qwen_image/test.parquet`

Launch online DPO training from the repository root:

```bash
bash examples/dpo_trainer/run_qwen_image_online_dpo_lora.sh
```

You can override any Hydra option at launch time. For example, to reduce the
rollout group size for a quick smoke run:

```bash
bash examples/dpo_trainer/run_qwen_image_online_dpo_lora.sh \
  data.train_batch_size=4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  trainer.total_training_steps=2
```

## References

- B. Wallace *et al.*, *Diffusion Model Alignment Using Direct Preference
  Optimization*, CVPR 2024.
- Diffusion-DPO official repository:
  <https://github.com/SalesforceAIResearch/DiffusionDPO>.
- FlowGRPO official repository DPO loss implementation:
  <https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_dpo.py>.

## Citation

```bibtex
@inproceedings{Wallace_2024_CVPR,
  author = {Wallace, Bram and Dang, Meihua and Rafailov, Rafael and Zhou, Linqi and Lou, Aaron and Purushwalkam, Senthil and Ermon, Stefano and Xiong, Caiming and Joty, Shafiq and Naik, Nikhil},
  title = {Diffusion Model Alignment Using Direct Preference Optimization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
  pages = {8228--8238}
}
```
