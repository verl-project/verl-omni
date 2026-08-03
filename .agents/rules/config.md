---
paths:
  - verl_omni/trainer/config/**
  - verl_omni/workers/config/**
---

# Config Rules

Dataclasses are the source of truth; YAML is composed by Hydra on top of them.
There are 16 config dataclasses: 2 in `verl_omni/trainer/config/algorithm.py`
(everything else in that directory is YAML), 14 in
`verl_omni/workers/config/{diffusion,omni}/`.

## Dataclass shape

```python
@dataclass
class DiffusionAlgoConfig(BaseConfig):
    """Diffusion-specific algorithm config."""

    trainer_type: str = "policy_gradient"
    adv_mode: str = "continuous"
    # True for pair-based algorithms (e.g. DPO)
    paired_preference: bool = False
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
```

- **Always inherit `BaseConfig`** (`from verl.base_config import BaseConfig`) or an
  existing subclass — 16/16 do. Never a bare `@dataclass`.
- **Nothing is `frozen=True`.** Mutability is controlled by `BaseConfig`'s
  `_mutable_fields` allowlist; extend it rather than replace it:
  ```python
  _mutable_fields = BaseConfig._mutable_fields | {"strategy", "algorithm"}
  ```
  Fields outside the allowlist reject assignment after construction. This is also
  why tests build configs with `object.__setattr__` ([testing.md](testing.md)).
- **Every field has a default.** "Required" is expressed with `MISSING`
  (`from omegaconf import MISSING`) and asserted in `__post_init__`:
  ```python
  path: str = MISSING
  ...
  assert self.strategy != MISSING
  ```
- **Document fields with a `#` comment above them**, not an `Attributes:` block —
  there are zero `Attributes:` sections in either config directory, and
  `check-docstrings` does not cover config files at all.

## Validation lives in `__post_init__`

All 27 `raise ValueError` sites follow one shape — set membership, listing the
valid values:

```python
def __post_init__(self):
    super().__post_init__()   # mandatory in subclasses; skipping it drops parent validation silently
    valid_adv_modes = {"continuous", "positive_only"}
    if self.adv_mode not in valid_adv_modes:
        raise ValueError(f"Invalid adv_mode: {self.adv_mode}. Must be one of {sorted(valid_adv_modes)}")
```

Don't reach for `Literal[...]` — it appears nowhere in these directories; plain
`str` plus a `valid_*` set is the established pattern.

`__post_init__` also does real I/O in places (tokenizer loading, path resolution,
`import kernels` in `workers/config/diffusion/model.py`), so it is not free to call.

## Wiring a new dataclass

Three steps, all required — miss one and the config is unreachable from YAML:

1. Add it to its module's `__all__`; the package `__init__.py` files re-export via
   `from .x import *` + `__all__ = list(x.__all__)`.
2. Reference it from YAML with `_target_`, e.g.
   `_target_: verl_omni.trainer.config.DiffusionAlgoConfig`.
3. Instantiate through `omega_conf_to_dataclass` (`verl.utils.config`) at the
   consuming site — not `hydra.utils.instantiate`, which this repo reserves for the
   agent-loop object.

Entry points are `@hydra.main` in `verl_omni/trainer/main_diffusion.py` and
`main_omni.py`, composing `diffusion_trainer.yaml` / `omni_trainer.yaml` through a
`defaults:` list.

## Generated YAML

Three files, all headed `# Do not modify this file directly.`:

| File                                       | Generated from                                       |
| ------------------------------------------ | ---------------------------------------------------- |
| `_generated_diffusion_trainer.yaml`        | `diffusion_trainer.yaml`                             |
| `_generated_diffusion_veomni_trainer.yaml` | same, with `diffusion/model_engine=veomni_diffusion` |
| `_generated_omni_trainer.yaml`             | `omni_trainer.yaml`                                  |

After touching any config dataclass:

```bash
scripts/generate_trainer_config.sh
```

The script regenerates **and verifies** (`git diff --exit-code` per target); the
`autogen-trainer-cfg` pre-commit hook runs it. Two traps:

- A VeOmni-only field changes **only** the `veomni` file — don't assume one diff.
- Without verl installed the script prints
  `Skipping omni_trainer: verl is not installed` and still exits 0, so a green
  local run can leave `_generated_omni_trainer.yaml` stale. CI catches it.

## Hand-written YAML has an enforced doc format

`tests/special_sanity/test_config_docs.py` gates a 9-file allowlist —
`diffusion_trainer.yaml` and the
`diffusion/{actor,ref,rollout,model,engine,model_engine}` groups (notably **not**
`omni_trainer.yaml`). The rules: a comment above every field, a blank line between
fields, no inline comments.

## Backward compatibility

Adding a field with a default is safe. Renaming or removing one breaks existing
user YAML and requires a `[BREAKING]` PR title
([commit-and-pr](../skills/commit-and-pr/SKILL.md)); deprecate before removing.
