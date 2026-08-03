---
paths:
  - "**/*"
---

# Code Style Rules

Rules beyond the automated pre-commit hooks (`ruff --fix`, `ruff-format`, `mypy`).
Formatting is handled by the tools below; this file covers conventions they do
**not** enforce.

## Automated gates (run before every commit)

`pre-commit install` wires these up. What each hook **actually** enforces — read off
`.pre-commit-config.yaml` and `tests/special_sanity/`, not off the hook names:

| Hook                       | What it actually enforces                                                  |
| -------------------------- | -------------------------------------------------------------------------- |
| `ruff` / `ruff-format`     | Lint + format, `line-length = 120` (scope below)                           |
| `mypy`                     | Static typing                                                              |
| `check-license`            | Apache header on every git-tracked `.py`; accepts `Copyright 2024/2025/2026 Bytedance …` |
| `autogen-trainer-cfg`      | `verl_omni/trainer/config/_generated_*.yaml` matches the dataclasses       |
| `check-docstrings`         | Presence (not style) of docstrings, in a hardcoded file list — see below     |
| `check-naming-conventions` | **Spelling only** — two project names, see below                          |
| `validate-structure`       | Test files must live in `tests/<module>/` (see [testing.md](testing.md))    |
| `check-device-api-usage`   | No `.cuda` / `"cuda"` / `"nccl"` under `verl_omni/`; per-file whitelist in the script |
| `check-dataproto-usage`    | No `DataProto` under `verl_omni/workers/engine/` — use `TensorDict`         |
| `compileall`               | Every `.py` byte-compiles with `PYTHONWARNINGS=error`                      |
| `check-docs-time-info`     | `Last updated` info in docs                                                |

Never hand-edit `_generated_*.yaml` — regenerate via
`scripts/generate_trainer_config.sh` (see [config.md](config.md)).

## Spelling (enforced, repo-wide grep)

`check-naming-conventions` greps the **whole worktree** — code, comments, docs,
markdown — and fails the commit on a misspelling of either project name: write
**`verl`** (all lower case) and **`SGLang`** or **`sglang`** (no other casing).

Read the exact patterns off the hook rather than restating them; a file that quotes
a rejected spelling in order to document it fails the hook too:

```bash
grep -A6 'id: check-naming-conventions' .pre-commit-config.yaml
```

Its `--exclude-dir` list covers `.git`, `.github`, `.specstory`, `.venv`, and
`__pycache__` — nothing else, so this directory is in scope.

## License Header (mandatory)

Every git-tracked `.py` file must start with the Apache 2.0 header. The checker
accepts 2024/2025/2026; every file under `verl_omni/` uses 2026, so use `2026` for
new files:

```python
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Docstrings

**What the gate actually checks**: `check-docstrings` only verifies that public
(non-`_`) top-level functions, classes, and methods **have some docstring**, and only
in a hardcoded file list — read it off the hook rather than guessing:

```bash
grep -n 'verl_omni/' tests/special_sanity/check_docstrings.py
```

It does **not** check docstring style, and does **not** cover the rest of the
repo. The following are therefore **conventions** (follow them; they are what
reviewers ask for), not automated gates:

- Public functions/classes get a docstring.
- Google-style `Args:` / `Returns:` / `Raises:` sections.
- Document tensor shapes and dtypes explicitly, e.g. `(C, H, W)` or `(N, C, H, W)`.
- Credit upstream in the module docstring when code is adapted
  (e.g. `It is adapted from https://github.com/kvablack/ddpo-pytorch.`).
- Recent review history favors **terse** docstrings — verbose explanatory blocks
  have been trimmed on request (see #311).

## Comments

This tree comments **sparsely** — about 4% of non-blank lines under `verl_omni/`,
license headers excluded. Measure it rather than trusting this number:

```bash
find verl_omni -name '*.py' | xargs awk 'FNR>15 && /^[[:space:]]*#/ {c++} FNR>15 && NF {t++} END {printf "%.1f%%\n", 100*c/t}'
```

A patch that arrives at 15–20% comment density does not look like this codebase, and
reviewers say so.

Roughly one comment in eight is a **trailing** comment rather than one on its own line:

```bash
find verl_omni -name '*.py' | xargs awk 'FNR>15 && /^[[:space:]]*#/ {o++} FNR>15 && /[^[:space:]]  #/ {r++} END {printf "%d own-line, %d trailing\n", o, r}'
```

What the existing comments actually do:

- **Explain why**, where the code cannot — `# Free cached GPU memory so colocated
  vLLM processes can see it via cudaMemGetInfo`, `# to spare GPU memory for reward
  model`, `# Fallback for CPU-only environments where vLLM-Omni
  current_omni_platform.device_type is empty.`
- **Label a step** in a long procedure, in three or four words — `# dump
  generations`, `# gather output`, `# Encode through T5 text encoder`.
- **Annotate one line, at the end of that line** — a tensor shape
  (`sequence_reward = sample_level_rewards.mean(dim=1)  # [B]`), what a default
  accepts (`target_modules: Optional[Any] = "all-linear"  # allow both "all-linear"
  and ["q_proj", "k_proj"]`), a caveat on a call
  (`find_latest_ckpt_path(...)  # None if no latest`), or why one entry of a literal
  is there. Two spaces before the `#`.
- **Flag known debt** — `# TODO: <what>` or `# TODO (name): <what>`, both forms in
  use. Name the condition for removal when there is one:
  `# TODO (mike): drop this once it is fixed in upstream diffusers.`

What to delete before sending a patch:

- Comments that restate the line below (`# increment the counter`,
  `# return the result`). If the code says it, the comment is noise.
- A block comment above a list, dict, or argument group whose note really belongs to
  individual entries. Annotate the entries; a header covering some-but-not-all of
  them makes the reader count lines to match reason to entry. Repeating the same
  trailing comment on two adjacent entries is fine and already in the tree.
- Narration of your own edit (`# Added handling for the new config field`,
  `# Changed to use the batched path`). Git records that; the file should read as
  if it always looked this way.
- Section banners around every few lines. The `# ---------` rule appears only where
  it separates top-level groups in the tree's longest modules; that is the ceiling,
  not a template.
- Restating a docstring in a comment directly above or below it.

Long-form explanation belongs in the docstring or in `docs/`, not in a comment
block above the function.

## Ruff lint scope

`[tool.ruff.lint]` selects `E, F, UP, B, I, G` and **ignores**:
`F403`, `F405` (star imports — **allowed**), `E731`, `B007`, `UP032`, `G004`
(f-strings in `.log()` are fine), `UP045`, `UP035`. `line-length = 120`.
`isort` treats `verl_omni` as first-party.

## Imports

- Group: stdlib, third-party, `verl_omni` (ruff `isort` handles ordering).
- `from x import *` is **not linted against** here (F403/F405 ignored) and is
  used in some `__init__`/config modules; prefer explicit imports for clarity in
  new code, but it is not a hard rule.
- Prefer **lazy imports inside functions** for heavy optional deps (diffusers,
  vllm_omni, flash-attn). This is the established pattern in the pipeline
  adapters and the reward dispatcher, and it is what keeps CPU import paths
  cheap:
  ```python
  if data_source == "jpeg_compressibility":
      from verl_omni.utils.reward_score import jpeg_compressibility
  ```
  Note the individual reward scorers do **not** follow it — `pickscore_reward.py`
  and `hpsv3_reward.py` import transformers models at module level. Those files
  are only imported when the reward is configured, so treat lazy importing as
  required on paths a CPU test touches and optional elsewhere.

## Naming Conventions

Not automated (`check-naming-conventions` only checks spelling — see above), but
consistently followed:

| Type                 | Pattern                | Example                              |
| -------------------- | ---------------------- | ------------------------------------ |
| Config dataclass     | `XxxConfig`            | `DiffusionModelConfig`, `OmniAlgoConfig` |
| Loss class           | `XxxLoss`              | `OmniDPOLoss`                        |
| Training adapter     | `<Model><Algo>`        | `QwenImageDPO`, `StableDiffusion3FlowGRPO` |
| Rollout adapter      | `XxxWithLogProb` for policy-gradient algos, else `<Model><Algo>Pipeline` | `QwenImagePipelineWithLogProb`, `QwenImageDPOPipeline` |
| Reward scorer fn     | `compute_score` or `compute_score_<name>` | `compute_score_hpsv3` |
| CPU test file        | `test_*_on_cpu.py`     | `test_qwen_image_dpo_adapter_on_cpu.py` |

CPU-test naming is load-bearing, not cosmetic — CI selects tests by that suffix
([testing.md](testing.md)). The handful of non-`_on_cpu` test files are GPU/NPU tests,
`tests/special_sanity/` checks, and a few older `tests/workers/` files.

## Device handling

**Avoid hardcoded `.cuda` / `"cuda"` / `"nccl"`; use the device API from
`verl.utils.device`** — `get_device_name`, `is_cuda_available`, `is_npu_available`
are the ones already imported across this tree (upstream verl, not a local module).

`check-device-api-usage` gates this over all of `verl_omni/`, but it is a plain
substring grep on file contents, so it also flags docstrings, comments, and
`EngineRegistry.register(device=["cuda", "npu"])` declarations. Those are exempted
per file in `tests/special_sanity/check_device_api_usage.py`; a handful of genuinely
hardcoded sites are exempted there too, with a `# TODO`. Prefer fixing the call over
adding an entry, and give a reason when you do add one.

## Reuse over duplication

Sharing between sibling implementations is the established pattern here, not an
aspiration. Four mechanisms are in active use — reach for one before copying a
block:

1. **A sibling's `common.py`.** A few pipeline packages
   (`qwen_image_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo`) export shared
   helpers, and other packages import across directories rather than re-implement:
   ```python
   from verl_omni.pipelines.qwen_image_flow_grpo.common import apply_true_cfg, build_img_shapes
   ```
2. **Subclass the closest existing class.** `QwenImageEditPlusFlowGRPO` and the NFT
   adapter both subclass `QwenImage` from `qwen_image_flow_grpo`; every loss class
   subclasses `DiffusionLossFn`; `MultiVisualRewardManager` subclasses
   `VisualRewardManager`; `PolicyGradient`/`DirectPreference` trainers share
   `BaseRayDiffusionTrainer`.
3. **A mixin, when the shared part cuts across the hierarchy.**
   `QwenImageTokenIdPromptMixin` (defined in `qwen_image_flow_grpo/common.py`) is
   mixed into two unrelated rollout pipelines; `NPUColocateWorkerMixin` was
   extracted the same way (#82).
4. **A `refactor:` commit.** It is a first-class type in the title convention and
   is used often — #287 folded Qwen-Image step execution into both FlowGRPO and
   MixGRPO, #96 unified Diffusion-DPO with DiffusionNFT, #56 dropped duplicated
   reward-loop patches. Consolidating on the way past is welcome; a *pure*
   mechanical cleanup on its own is not (`AGENTS.md` §1).

Practical threshold: if you are about to copy more than a few lines from a sibling
directory, import them instead — and if the sibling's version needs a parameter to
serve both callers, add the parameter there rather than forking the function.

Two caveats, so this is not applied blindly:

- **Don't invent a shared abstraction for a single caller.** Most pipeline packages
  have no `common.py` — they had nothing worth sharing yet.
- **Don't merge across a registry boundary.** Each `(architecture, algorithm)` pair
  registers its own adapter ([pipelines.md](pipelines.md)); collapsing two of them
  into one class with an `if algorithm == ...` switch defeats the dispatch.

## Performance Patterns

Conventions, not gated:

- Avoid needless GPU→CPU syncs (`.item()`, `.tolist()`, `print(tensor)`) in hot
  paths.
- Prefer batched tensor ops over Python loops over elements — recent review
  feedback explicitly asked for whole-tensor `permute/sanitize/quantize` instead
  of per-frame loops (#311).
- Be explicit about `dtype`/`device`; do not rely on implicit promotion.

## Modules

Changes are labeled by module in commit and PR titles. The authoritative list and
the path→module mapping live in the
[commit-and-pr](../skills/commit-and-pr/SKILL.md) skill.
