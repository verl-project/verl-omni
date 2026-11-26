# Contributing to verl-omni

Thank you for considering a contribution to verl-omni! We welcome contributions of any kind - bug fixes, enhancements, documentation improvements, or even just feedback. Whether you're an experienced developer or this is your first open-source project, your help is invaluable.

Your support can take many forms:

- Report issues or unexpected behaviors.
- Suggest or implement new features.
- Improve or expand documentation.
- Review pull requests and assist other contributors.
- Spread the word: share verl-omni in blog posts, social media, or give the repo a ⭐.

## Finding Issues to Contribute

Looking for ways to dive in? Check out these issues:

- [Good first issues](https://github.com/verl-project/verl-omni/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Call for contribution](https://github.com/verl-project/verl-omni/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)

Furthermore, you can learn the development plan and roadmap via the [RFC: Multi-modal Generation RL 2026Q2 Roadmap](https://github.com/verl-project/verl/issues/5755) and the [RFC: Q3 Road Map](https://github.com/verl-project/verl-omni/issues/97).

## Developing

For the full dependency setup, see the [installation doc](docs/start/install.md).

## Code Linting and Formatting

We rely on pre-commit to keep our code consistent. To set it up:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
# run a specific hook with pre-commit
# pre-commit run --all-files --show-diff-on-failure --color=always <hook-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

The lightweight `lint` job runs on every pull request, without a CI label. It
runs Ruff lint and format over the checkout and the `check-pr-lint` hook for
new AST and Ruff findings relative to the PR merge base. To run just the new
checker locally, stage your changes and run `uvx pre-commit run check-pr-lint`.
Labeled GPU and NPU PR runs check lint before using accelerator runners.

The AST rules follow module responsibilities rather than treating all Python
files alike ([RFC #942](https://github.com/verl-project/verl-omni/issues/942)):

- Shared trainers, workers, agent loops, and pipeline contracts use the registry
  or shared interfaces, not concrete pipeline imports (`VO001`).
- Rollout adapters and workers use shared contracts instead of training engine internals (`VO002`).
- Reuse public APIs across directories and sibling reward scorers. Keep private
  helpers within their owning package; `reward_utils` remains a shared scorer helper (`VO003`).
- Use an explicit sampling seed, not a request ID hash, when seeding pipeline RNGs (`VO004`).
- For Qwen-Image RoPE lengths, use `txt_seq_lens_from_embeds()` rather than summing the attention mask (`VO005`).
- Do not silently discard broad exceptions with `pass` or `continue` in production code ([RFC #388](https://github.com/verl-project/verl-omni/issues/388)).
- Test scripts that call `pytest.main()` must check or propagate its exit status (`VO006`, [PR #814](https://github.com/verl-project/verl-omni/pull/814#discussion_r4452763469)). Name L1 CPU tests `test_*_on_cpu.py`, not `test_*_cpu.py` ([PR #409](https://github.com/verl-project/verl-omni/pull/409#discussion_r3965752859)).
- CPU-marked tests also need the `_on_cpu.py` suffix (`VO007`). GPU tests use
  topology helpers instead of bare `torch.cuda.is_available()` skip conditions (`VO013`).
- Shared trainers, workers and agent loops must not dispatch on known concrete
  model-name literals (`VO008`). Training adapters keep stateless class/static
  hooks (`VO009`); pipeline packages must be imported at the pipeline entrypoint (`VO010`).
- Config dataclasses need a parent config, non-frozen mutability, field defaults
  (or `omegaconf.MISSING`, never `dataclasses.MISSING`), no `Literal[...]`, subclass
  `super().__post_init__()`, and `__all__` export (`VO011`). Reward managers leave
  HTTP client imports to scorers (`VO012`) and propagate `ArtifactContractError` (`VO016`).
- Public API contracts (`__all__` exports, shared pipeline contract modules,
  and `compute_score*` entrypoints) require concise docstrings (`VO015`).
- Pass personal model/data paths through configuration or temporary fixtures
  instead of `/home/`, `/Users/`, `/user/`, `/nfs/`, `/mnt/` or `/root/` literals (`VO014`).

The scripts use the `check_<subject>.py` naming convention also used by
[SGLang-Omni's checks](https://github.com/sgl-project/sglang-omni/tree/main/scripts):
`check_pr_lint.py` handles Git snapshots, incremental findings and Ruff;
`check_ast_rules.py` holds Python AST contracts; `check_example_recipes.py`
holds path and recipe checks; `check_comments.py` checks comment tokens. Each
rule module has its own `test_<subject>.py` regression suite under
`tests/special_sanity/`. Production Ruff additionally
checks mutable defaults, duplicate fields and invalid exports, bare exceptions,
control flow that suppresses exceptions, missing subprocess `check=`, discarded
async tasks, resource handling, debugging calls, `eval` and `shell=True`.
New `global` statements are rejected outside reward scorers, whose existing
contract permits module-level caches. Tests run only the additional `T100` and
`PT015` rules (use `pytest.fail()` for an intentional test failure), alongside the
repository's existing full Ruff checks. These rules run with isolated settings
and ignore `noqa`; PR-local lint configuration cannot disable them.

Keep necessary public API contracts in concise docstrings. Default to no
implementation comments; retain only non-obvious algorithm/layout/concurrency
constraints or verified upstream workarounds. Do not add line-by-line narration,
edit history, numbered steps, decorative banners or commented-out code.
`CM001`–`CM005` reject recognizable edit/step/chatty narration, decorative
banners, echo comments, PR-template/filler/signature-echo docstrings, and
`PR #...` comments; isolated production Ruff `ERA001` rejects commented-out
code. License/attribution, tooling directives, and required config documentation
remain intact. In addition, the PR `Lint` workflow runs Ruff plus
`check-license`, `validate-structure`, `check-device-api-usage`,
`check-dataproto-usage`, `check-naming-conventions`, and `compileall`.

Example shell changes are parsed with `bash -n` and checked for trailing spaces
after `\` continuations (`EX001`). `run_*.sh` scripts must use
`#!/usr/bin/env bash` with `set -e` and `set -o pipefail` (`EX005`) and forward
caller overrides with trailing double-quoted `"$@"` (`EX002`; comments, printed
text and heredocs do not count). Shell recipes also catch orphaned or duplicate
Hydra overrides (`EX007`), hardcoded tokens/device lists (`CUDA`/`ASCEND_RT`/
`MLU`/`HIP`/`ROCR`)/Ascend `source` paths/private IPs/`cd /...`/host-wide `pkill`
or `pip install` (`EX008`), disallow committed example media (`EX003`),
`README_*.md` (`EX004`), or `dataprocess/`, duplicate folders, and
`.venv/bin/python` (`EX006`), and require new `_npu`/`_mlu`/`_amd`/`_rocm`
variants with an existing GPU sibling recipe to delegate via `exec bash` / `bash` / `source`
instead of duplicating it (`EX009`).

The checker reads staged Git objects or the PR snapshots without importing
candidate Python or executing recipes. Historical findings are compared by
diagnostic and enclosing function/class; new violations still fail. Known model
names and literal patterns are heuristic checks: dynamic dispatch, cross-module
inheritance and runtime field consumption still require tests and review.
Run the checker regressions with:

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml --with ruff==0.12.2 pytest -q \
  tests/special_sanity/test_ci_policy.py tests/special_sanity/test_pr_lint.py \
  tests/special_sanity/test_ast_rules.py tests/special_sanity/test_comments.py \
  tests/special_sanity/test_example_recipes.py
```

Use static checks for known code patterns. For behavior changes, establish the
contract at the owning boundary and test the contract across its consumers.
For example, L1 should exercise a registered adapter with a new TQ field through
the real projection path, and compare serial and packed rollouts for seed or
padding changes. Use diff-selected L2 GPU smoke for numerical parity between the
actor and rollout, not just a successful launch.

## Testing

Our test suites run on GitHub Actions. Check these [workflows](https://github.com/verl-project/verl-omni/tree/main/.github/workflows) for details.

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. CPU unit tests: add `tests/<module>/test_*_on_cpu.py` (auto-collected by
   the CPU CI job) or a check under `tests/special_sanity/` — no workflow edit
   needed.
2. GPU/NPU smoke: register the script in `tests/gpu_smoke/run_gpu_smoke_*.sh`
   (GPU) or `tests/npu_smoke/run_npu_smoke_{core,e2e}.sh` (NPU), and extend the
   GPU group patterns in
   `tests/gpu_smoke/select_gpu_smoke_groups.py` if needed
   (see [GPU smoke tests](docs/contributing/gpu_smoke_tests.md) and
   [NPU smoke tests](docs/contributing/npu_smoke_tests.md)). Do not edit
   `.github/workflows/` — CI infrastructure changes need maintainer approval.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

## Building the Docs

```bash
pip install -e .
pip install -r docs/requirements-docs.txt

cd docs
make clean
make html

# Preview locally
python -m http.server -d _build/html/
```

Open your browser at http://localhost:8000 to explore the docs.

If autodoc fails on missing `vllm` / `vllm_omni` imports, install the runtime stack from [docs/start/install.md](docs/start/install.md) first.

## Model & Algorithm Integrations

To integrate a new diffusion model for an existing PPO-like algorithm (new
`DiffusionModelBase` + `VllmOmniPipelineBase` pair), follow:

- [How to Integrate a New Diffusion Model for FlowGRPO Training](docs/contributing/integrating_a_diffusion_model.md)

To integrate a new policy-gradient diffusion algorithm (new advantage
estimator, loss, and SDE step scheduler), follow:

- [How to Integrate a New Policy-Gradient Algorithm for Diffusion Model](docs/contributing/integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md)

To integrate a new direct-preference diffusion algorithm (offline DPO-style
pairs or online DiffusionNFT-style forward-process training), follow:

- [How to Integrate a New Direct-Preference Algorithm for Diffusion Model](docs/contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md)

Additional guides:

- [How to Integrate a Reward Scorer](docs/contributing/integrating_a_reward_scorer.md)
- [How to Integrate an Image-to-Image Diffusion Model](docs/contributing/integrating_an_i2i_diffusion_model.md)
- [How to Integrate a Non-Diffusers Model for FlowGRPO Training](docs/contributing/integrating_a_non_diffusers_model.md)
- [How to Add a New Omni Model](docs/contributing/integrating_an_omni_model.md)
- [How to Add Continuous Batching (Step-Execution) Support for a Diffusion Model](docs/contributing/integrating_a_stepwise_continuous_batching_model.md)
- [Testing Guide](docs/contributing/testing_guide.md)
- [GPU Smoke Tests](docs/contributing/gpu_smoke_tests.md)
- [NPU Smoke Tests](docs/contributing/npu_smoke_tests.md)
- [CI/CD Layers](docs/contributing/ci_cd.md)
- [Common Pitfalls](docs/contributing/common_pitfalls.md)
- [Integrating the Diffusion Prompt-Embedding Cache](docs/contributing/integrating_prompt_embedding_cache.md)

## Governance

Project roles, module ownership, and the committer nomination process are documented in [`docs/community/governance.md`](docs/community/governance.md). Path-based reviewer routing lives in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## Pull Requests & Code Reviews

Thanks for submitting a PR! To streamline reviews:

- Follow our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) for title format and checklist.
- Format the PR title as `[{modules}] {type}: {description}`. The
  [PR-title CI check](tests/special_sanity/check_pr_title.py) is the source of
  truth for accepted modules and types; the pull request template mirrors the
  current values.
- Adhere to our pre-commit lint rules and ensure all checks pass.
- Update docs for any user-facing changes.
- Add or update tests in the CI workflows, or explain why tests aren't applicable.

## AI-Assisted Contributions

You are the author of everything you post. If an agent drafted your PR
description, commit messages, replies or code comments, read them properly and
revise until they are concise and say what you would write yourself.

See

- [`AGENTS.md`](AGENTS.md) for rules that all AI coding agents must follow
- [`editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md) for guidelines on editing agent instructions.

## License

See the [LICENSE](https://github.com/verl-project/verl-omni/blob/main/LICENSE) file for full details.

## Thank You

We appreciate your contributions to verl-omni. Your efforts help make the project stronger and more user-friendly. Happy coding!
