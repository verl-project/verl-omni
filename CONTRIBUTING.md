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

Furthermore, you can learn the development plan and roadmap via the [RFC: Multi-modal Generation RL 2026Q2 Roadmap](https://github.com/verl-project/verl/issues/5755).

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

## Testing

Our test suites run on GitHub Actions. Check these [workflows](https://github.com/verl-project/verl-omni/tree/main/.github/workflows) for details.

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file under `.github/workflows/`.
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

## Building the Docs

```bash
# Ensure verl-omni is on your PYTHONPATH, e.g.:
pip install -e .

# Install documentation dependencies
cd docs
pip install -r requirements-docs.txt

# Generate HTML docs
make clean
make html

# Preview locally
python -m http.server -d _build/html/
```

Open your browser at http://localhost:8000 to explore the docs.

## Model & Algorithm Integrations

To integrate a new diffusion model for an existing PPO-like algorithm (new
`DiffusionModelBase` + `VllmOmniPipelineBase` pair), follow:

- [How to Integrate a New Diffusion Model for FlowGRPO Training](docs/contributing/integrating_a_diffusion_model.md)

To integrate a new PPO-like RL algorithm (new advantage estimator, loss, and
SDE step scheduler), follow:

- [How to Integrate a New PPO-like Algorithm for Diffusion Model](docs/contributing/integrating_a_new_algorithm_for_diffusion_model.md)

## Pull Requests & Code Reviews

Thanks for submitting a PR! To streamline reviews:

- Follow our [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) for title format and checklist.
- Format the PR title as `[{modules}] {type}: {description}` — valid modules include `vllm_omni`, `diffusion`, `omni`, `rollout`, `trainer`, `reward`, `model`, `algo`, `fsdp`, `ray`, `worker`, `data`, `cfg`, `ckpt`, `doc`, `ci`, `tests`, `docker`, `misc`.
- Adhere to our pre-commit lint rules and ensure all checks pass.
- Update docs for any user-facing changes.
- Add or update tests in the CI workflows, or explain why tests aren't applicable.

## AI-Assisted Contributions

See

- [`AGENTS.md`](AGENTS.md) for rules that all AI coding agents must follow
- [`editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md) for guidelines on editing agent instructions.

## License

See the [LICENSE](https://github.com/verl-project/verl-omni/blob/main/LICENSE) file for full details.

## Thank You

We appreciate your contributions to verl-omni. Your efforts help make the project stronger and more user-friendly. Happy coding!
