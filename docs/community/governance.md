# Governance

Last updated: 09/09/2026

VeRL-Omni is an open-source project. Committer status is earned through contribution, maintenance, and stewardship — not purchased or assigned by company affiliation.

## Values

VeRL-Omni aims to be an easy, fast, and stable RL training framework for diffusion and omni-modality models.

### Design Values

1. **Top performance**: Training and rollout throughput are first-class. We monitor overheads, overlap reward with generation, and publish recipes and benchmarks. We never leave performance on the table.
2. **Ease of use**: VeRL-Omni must be simple to install, configure, and run. We provide clear documentation, Hydra configs, example recipes, helpful error messages, and reproducible end-to-end training paths. Many users fork our code or study it deeply, so we keep it readable and modular.
3. **Wide coverage**: VeRL-Omni supports frontier diffusion, unified multimodal, and omni-modality models, plus high-performance accelerators. We make it easy to add new models, algorithms, rewards, and hardware backends.
4. **Production ready**: VeRL-Omni is used for long-running RL jobs. Training must be stable, observable, and operable — with clear logs, metrics, and checkpointing.
5. **Extensibility**: VeRL-Omni cannot cover every use case in-tree. We design pipelines, trainers, workers, and reward loops so they can be forked, composed, and customized.

### Collaboration Values

1. **Tightly Knit and Fast-Moving**: Our maintainer team is aligned on vision, philosophy, and roadmap. We work closely to unblock each other and move quickly.
2. **Individual Merit**: No one buys their way into governance. Committer status belongs to individuals, not companies. We reward contribution, maintenance, and project stewardship.

## Project Maintainers

### Lead Maintainers

Lead maintainers are responsible for the overall direction and strategy of the project:

- [@samithuang](https://github.com/samithuang) (Yongxiang Huang)
- [@wuxibin89](https://github.com/wuxibin89) (Xibin Wu)

### Active Committers

Committers have write access and merge rights. They typically have deep expertise in specific areas of this project and shepherd the community contributions:

- [@AndyZhou952](https://github.com/AndyZhou952) (Jingan Zhou): Trainer and algorithm core; Diffusion Models
- [@chenyingshu](https://github.com/chenyingshu) (Susan, Yingshu Chen): Trainer and algorithm core; Reward system; Pipeline and model adaptation; Datasets and examples
- [@cr-gao](https://github.com/cr-gao) (Chenrui Gao): Workers and training engines
- [@knlnguyen1802](https://github.com/knlnguyen1802) (Long Nguyen): Rollout and agent loop; Pipeline and model adaptation
- [@NancyFyong](https://github.com/NancyFyong) (Zhiyong Feng): Trainer and algorithm core; Rollout and agent loop; Pipeline and model adaptation; Datasets and examples
- [@ruihanglix](https://github.com/ruihanglix) (Ruihang Li): Reward system
- [@Sky-Trigger](https://github.com/Sky-Trigger) (Mengbo Wang): Rollout and agent loop; Reward system
- [@WenzheWang](https://github.com/WenzheWang) (Wenzhe Wang): Trainer and algorithm core
- [@wtomin](https://github.com/wtomin) (Didan DENG): Tests and CI; Packaging and environment; Documentation
- [@ZihaoW123](https://github.com/ZihaoW123) (Zihao Wang): Rollout and agent loop; Reward system; Pipeline and model adaptation
- [@zhtmike](https://github.com/zhtmike) (Cheung Ka Wai): Workers and training engines

### Path Ownership

Directory rules cover the whole tree unless a more specific path below overrides them.

| Path | Committers |
| --- | --- |
| `verl_omni/trainer/` | [@chenyingshu](https://github.com/chenyingshu), [@NancyFyong](https://github.com/NancyFyong), [@WenzheWang](https://github.com/WenzheWang) |
| `verl_omni/workers/` | [@cr-gao](https://github.com/cr-gao), [@zhtmike](https://github.com/zhtmike) |
| `verl_omni/agent_loop/` <br> `verl_omni/workers/rollout/` <br> `verl_omni/utils/vllm_omni/` | [@knlnguyen1802](https://github.com/knlnguyen1802), [@Sky-Trigger](https://github.com/Sky-Trigger), [@NancyFyong](https://github.com/NancyFyong), [@ZihaoW123](https://github.com/ZihaoW123) |
| `verl_omni/reward_loop/` <br> `verl_omni/utils/reward_score/` | [@ruihanglix](https://github.com/ruihanglix), [@chenyingshu](https://github.com/chenyingshu), [@Sky-Trigger](https://github.com/Sky-Trigger), [@ZihaoW123](https://github.com/ZihaoW123) |
| `verl_omni/pipelines/` | [@chenyingshu](https://github.com/chenyingshu), [@NancyFyong](https://github.com/NancyFyong), [@ZihaoW123](https://github.com/ZihaoW123), [@knlnguyen1802](https://github.com/knlnguyen1802) |
| `verl_omni/utils/dataset/` <br> `examples/` | [@chenyingshu](https://github.com/chenyingshu), [@NancyFyong](https://github.com/NancyFyong) |
| `tests/` <br> `.github/` <br> `scripts/` <br> `docker/` <br> `docs/` <br> `.agents/` | [@wtomin](https://github.com/wtomin) |

`verl_omni/workers/rollout/` overrides `verl_omni/workers/` for reviewer routing. Unlisted paths fall back to the default owner in [`.github/CODEOWNERS`](https://github.com/verl-project/verl-omni/blob/main/.github/CODEOWNERS).

## Reviewer Routing

For path-based reviewer routing, consult [`.github/CODEOWNERS`](https://github.com/verl-project/verl-omni/blob/main/.github/CODEOWNERS). The path table above is the source of truth for subsystem boundaries.

`CODEOWNERS` is operational routing metadata, not a governance or approval policy. Being listed does not grant committer status or merge rights, and it does not mean every listed reviewer must approve a change. The maintainer roster above is authoritative for project roles and merge rights.

## Committer Nomination Process

Every month, any active committer can nominate new committer(s) to the project. Up to **two new committers** will be admitted per month based on the quality and impact of their contributions.
