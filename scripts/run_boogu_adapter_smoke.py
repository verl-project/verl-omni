import argparse
import csv
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch

from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import BooguImageFlowGRPO
from verl_omni.pipelines.utils import prepare_model_inputs


def build_model_config(true_cfg_scale: float = 2.0):
    cfg = SimpleNamespace()
    cfg.algorithm = "flow_grpo"
    cfg.architecture = "BooguImagePipeline"
    cfg.external_lib = None
    cfg.pipeline = SimpleNamespace(height=64, width=64, num_inference_steps=4, true_cfg_scale=true_cfg_scale)
    cfg.algo = SimpleNamespace(noise_level=0.0, sde_type="sde")
    cfg.local_path = "DUMMY/BOOGU"
    return cfg


class TinyBooguTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(axes_dim_rope=(4, 4, 4), axes_lens=(2, 2, 2))
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(8, 4, 3, padding=1),
        )

    def forward(
        self,
        hidden_states,
        timestep,
        instruction_hidden_states,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states=None,
        return_dict=False,
    ):
        del freqs_cis, instruction_attention_mask, ref_image_hidden_states, return_dict
        scale = instruction_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        t = timestep.view(-1, 1, 1, 1).to(hidden_states.dtype)
        return self.net(hidden_states) + 0.05 * scale + 0.01 * t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="assets/pr_results/boogu_smoke")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    module = TinyBooguTransformer()
    model_config = build_model_config(true_cfg_scale=2.0)

    latents = torch.randn(1, 2, 4, 8, 8)
    timesteps = torch.tensor([[900.0, 500.0]])
    prompt_embeds = torch.randn(1, 2, 16)
    prompt_mask = torch.ones(1, 2, dtype=torch.long)
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_prompt_mask = torch.ones_like(prompt_mask)
    micro_batch = {"condition_image_latents": torch.randn(1, 1, 4, 8, 8)}

    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        model_inputs, negative_model_inputs = prepare_model_inputs(
            module=module,
            model_config=model_config,
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_mask,
            micro_batch=micro_batch,
            step=0,
        )
        prediction = BooguImageFlowGRPO.forward(module, model_config, model_inputs, negative_model_inputs)
        loss = (prediction**2).mean()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 10 == 0 or step == 0:
            print(f"iter={step + 1} loss={losses[-1]:.6f}")

    csv_path = os.path.join(args.out_dir, "loss.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "loss"])
        writer.writerows([[idx + 1, value] for idx, value in enumerate(losses)])

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2)
    plt.title("BOOGU adapter smoke run (synthetic model)")
    plt.xlabel("iteration")
    plt.ylabel("MSE loss (pred^2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_png = os.path.join(args.out_dir, "loss_curve.png")
    plt.savefig(output_png, dpi=180)
    print(f"saved: {output_png}")


if __name__ == "__main__":
    main()
