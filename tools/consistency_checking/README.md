# Diffusion consistency checking

Last updated: 11/20/2026

These tools compare saved training and rollout evidence. They separate numerical
log-probability differences from a declared scheduler-input contract violation;
module/operator localization uses the existing [#620 workflow](../../docs/perf/diffusion_consistency.md#escalate-to-module-level-localization).

| Entry | Purpose |
| --- | --- |
| `check_diffusion_consistency.py` | Validate pinned metadata and MSProbe dumps, then compare saved log probabilities on CPU. The existing `scripts/check_diffusion_consistency.py` command forwards here. |
| `replay_diffusion_trainer_capture.py` | Read shared trainer capture events and compare every saved driver row/timestep; diagnose recognized timestep/sigma contracts in saved engine probes. |
| [SD3.5 recipe](sd3/README.md) | Collect a fixed real rollout/actor first-forward pair, retain independent pins and replay it on CPU. |

For the normal V1 PPO/SDE trainer, enable the capture library through the existing
Ray `runtime_env.worker_process_setup_hook`; the [trainer capture guide](../../docs/perf/diffusion_consistency.md#capture-on-the-real-trainer-path)
lists the Hydra overrides and artifact contract. No nightly CI modification or
training-default change is required.

```bash
python3 tools/consistency_checking/replay_diffusion_trainer_capture.py \
  --root /absolute/capture-events \
  --output /absolute/fresh-report.json
```

Keep the report outside the capture root. An admitted report covers the saved
pairs/probes; it does not establish complete engine coverage, model-weight
identity or an automatic cross-backend module mapping. For fused/split modules,
retain paired MSProbe dumps and use the shared localization workflow rather than
adding a second model-specific mapping registry here.
