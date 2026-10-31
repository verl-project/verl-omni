# DMD2 computation core

`DMD2Program` implements continuous-flow distribution matching using the registered
`DMDLoss` and public `verl_omni.workers.utils.losses.diffusion_loss` wrapper.
The objective excludes adversarial loss; model construction and training lifecycle
remain caller-owned.

The caller supplies a velocity callback:

```python
def predict(role, latents, sigmas, negative):
    # Return flow velocities with the same modality keys and shapes as latents.
    ...
```

`role` is `student`, `fake_score`, or `teacher`; `negative=True` selects unconditional
teacher inputs for CFG. Latents are floating `(B, ...)` tensors sharing a batch size
and device. The callback owns models, conditions, precision, and forward contexts.

Teacher `cfg_norm="layer_norm"` with guidance other than 1 requires packed
`(B, tokens, features)` latents and velocities. It rescales last-axis vectors by
their L2 norms, not `torch.nn.LayerNorm`. The program checks rank, while the caller
must ensure the feature axis is last. CFG on unpacked latents requires `"none"` or
`"scalar"` normalization. Fake-score updates and inference have no CFG layout restriction.

A minimal configuration for packed image latents:

```python
from verl_omni.pipelines.dmd2.program import DMD2Program
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig

config = DiffusionDMDConfig(
    score_discrete_steps=0,
    modality_shifts={"image": 3.0},
    modality_weights={"image": 1.0},
)
program = DMD2Program(config, predict, seed=42)
```

The shared YAML defaults to `score_discrete_steps=1000`, so callers must override it
to `0` and supply both modality dictionaries. Hydra needs `+` for new dictionary
keys: `+dmd.modality_shifts.image=3.0` and `+dmd.modality_weights.image=1.0`.

| Fields | Meaning in this program |
| --- | --- |
| `num_inference_steps` | Positive integer number of generation steps. |
| `modality_shifts` | Per-modality rational shifts for generation and score sampling. |
| `modality_weights` | Per-modality loss weights; keys match the shifts. |
| `score_time_min/max` | Bounds on uniform base time **before** the score shifts. |
| `score_discrete_steps` | Must be `0`; discrete score sampling is unsupported here. |
| `rollout_timestep_shift`, `score_timestep_shift` | Upstream scalar-shift fields, unused here; keep their defaults. |
| `score_sigma_min/max` | Upstream sigma-bound fields, unused here; keep their defaults. |

Changing an unused field raises an error pointing to the supported setting.

Each stage samples one CPU FP32 `t ~ U(score_time_min, score_time_max)`, shared
across the microbatch and all modalities. Modality `m` uses
`sigma_m = shift_m * t / (1 + (shift_m - 1) * t)`. Default time bounds `[0.02, 0.98]`
with shift 3 give sigma endpoints `[0.05769, 0.99324]` and nonuniform sigma sampling.
Student-stage fake-score and teacher predictions share the same noisy samples and sigmas.

Supply fresh standard Gaussian initial latents for each stage. Generation predicts
clean samples and re-noises between steps; only the selected student-stage exit
retains a gradient graph. Fake-score updates detach generated samples; student
updates detach teacher and fake-score predictions. CFG applies only to the teacher.

`compute_loss(initial, stage, data, loss_function, dp_group=...)` returns weighted
loss, per-modality metrics, and a nonfinite flag. Bind the public loss wrapper with
`loss_mode="dmd2"`; supply positive integer `gradient_accumulation_steps` and
`sp_size` in the `TensorDict`. The wrapper applies these scales once. Use `sp_size=1`
when gathering whole-sample outputs already scales backward.

`training_stages()` returns `fake_update_ratio` fake-score attempts followed by one
student attempt. The caller executes backward and optimizer updates, owns clipping,
schedulers, EMA and checkpoints, and handles reduction and skipped updates across ranks.
For resume, each rank saves these RNG states at the same training boundary:

| State | Owner and use |
| --- | --- |
| `program.state_dict()` | Private CPU generator for exit selection only. |
| Global CPU RNG | `torch.get_rng_state()` / `torch.set_rng_state()`; score time, plus any CPU tensor noise. |
| Used accelerator RNGs | Device backend's `get_rng_state` / `set_rng_state` (or `*_all`); initial noise, re-noising, and model randomness. |

`seed` and `load_state_dict()` affect only exit selection. Restore caller-owned CPU,
device, data-loader and other RNG states after model loading, before sampling the
next initial latents. Coordinate streams within model-parallel groups and use
independent streams across data-parallel replicas. RNG restoration does not ensure
deterministic backend kernels or reductions.

`sample(initial)` uses the final exit without a gradient graph or advancing the
exit-selection RNG. It still consumes the caller's noise RNG between steps.

With the repository's normal dependencies installed, run the CPU tests with:

```bash
pytest -q tests/pipelines/test_dmd2_program_on_cpu.py \
  tests/trainer/diffusion/test_dmd_config_on_cpu.py \
  tests/trainer/diffusion/test_dmd_loss_on_cpu.py \
  tests/trainer/diffusion/test_distillation_utils_on_cpu.py
```

CPU CI collects these files via the `_on_cpu.py` suffix. They test the computation
core without production models, engines, or distributed execution.
