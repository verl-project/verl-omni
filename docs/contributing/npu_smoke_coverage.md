# GPU-to-NPU Smoke Coverage

Last updated: 10/21/2026.

This inventory uses upstream commit `9958831` and the three GPU smoke runners
as its baseline. A migration is only validated after its pytest cases pass on
the target NPU stack. Shared assertions, collection, or a green workflow with
skipped jobs do not establish NPU runtime coverage.

## This migration

| GPU source | NPU entry | Assertions retained | Runtime status |
|---|---|---|---|
| `test_diffusion_agent_loop.py::test_single_turn` (GPU core ID 0) | NPU ID 4, `test_diffusion_single_turn_npu` | Both request/step modes; batch expansion, trajectory/log-probability keys, two-turn contract | Passed on eight-NPU A3 at `5926e66` |
| `test_diffusion_rollout_seed_gpu.py::test_rollout_without_seed_produces_different_initial_latents` (GPU core ID 4) | NPU ID 5, `test_rollout_without_seed_npu` | Two agent workers; different initial latents across reruns and within a rollout batch | Passed on eight-NPU A3 at `5926e66` |
| `test_diffusion_rollout_seed_gpu.py::test_rollout_seeds_unique_across_agent_loop_workers` (GPU core ID 4) | NPU ID 6, `test_rollout_seed_reproducibility_npu` | Stable global sample indices; bit-equal same-seed reruns; distinct latents for distinct rollout indices | Passed on eight-NPU A3 at `5926e66` |

The four new pytest cases passed (`4 passed, 22 warnings`, exit 0). Core
entries 0/4/5/6 passed, and the combined smoke runner exited 0 on that
revision. These runs preceded the merge of NPU entry 3 in PR #770; the
seven-entry combined suite needs a fresh integrated run, and the workflow's
A2 runner has not yet validated these new entries.

The shared execution helpers leave the GPU fixtures, Ray setup, topology,
attention defaults, and cleanup in the GPU entry points. The NPU entry point
supplies local assets, native NPU attention, TP=2, two agent workers, and NPU
Ray environment settings. Its seed cases use 256x256 images to keep the smoke
workload small; their assertions are unchanged. This checks reproducibility
within NPU execution, not GPU/NPU numerical parity or model convergence.

## GPU core inventory

| GPU core entry | NPU relationship at this baseline |
|---|---|
| 0: diffusion agent loop | NPU ID 4; request and step modes passed on A3, A2 CI pending |
| 1: vLLM-Omni rollout | NPU ID 0 covers basic generation and sleep/wake; request-level batching and algorithm-specific step contracts still need a case-level audit |
| 2: diffusers FSDP engine | NPU ID 3 was merged in PR #770 for FSDP with SP=1; FSDP2 and SP coverage are still separate gaps |
| 3: diffusers VeOmni engine | Not migrated |
| 4: diffusion rollout seed multi-worker | NPU IDs 5 and 6 passed on A3, A2 CI pending |
| 5: visual reward manager | Not migrated |
| 6: diffusion composite agent loop | Not migrated |
| 7: OmniAgentLoopWorker image-generation tool | Not migrated |
| 8: diffusers Ulysses SP | Not migrated; includes forward, backward, FSDP, MiniMax-H3, and SP=2/4 variants |
| 8: trainable FSDP2 snapshot restore | Not migrated; a separate source file from Ulysses despite the repeated runner ID |

Paths and pytest node IDs are the identities for coverage tracking. Runner
numbers alone are insufficient, including the repeated GPU core ID 8 above.

## GPU trainer inventory

| GPU group / entry | NPU relationship at this baseline |
|---|---|
| diffusion 0: Qwen-Image-Edit FlowGRPO v1 sync | Not migrated |
| diffusion 1: Qwen-Image FlowGRPO v1 sync | Related NPU ID 1 exists; trainer/configuration parity still needs auditing |
| diffusion 2: Qwen-Image online DPO v1 sync | Not migrated |
| diffusion 3: DiffusionNFT v1 sync | Not migrated |
| diffusion 4: FlowGRPO v1 separate-async | Not migrated |
| diffusion 5: Bagel PickScore LoRA FlowGRPO | Not migrated |
| diffusion 6: FlowGRPO synchronous separate trainer | Not migrated |
| diffusion 7: MiniMax-H3 T2VA FSDP-SP | Not migrated |
| diffusion 8: diffusion OPD colocated teachers | Not migrated |
| diffusion 9: diffusion OPD separate-async teachers | Not migrated |
| diffusion 10: Boogu-Image DiffusionNFT | Not migrated |
| diffusion 11: named image previews and MiniMax-H3 offload placeholders | Not migrated |
| omni 0: Qwen3-Omni Thinker GSPO LoRA v1 sync | Not migrated |
| omni 1: Qwen3-Omni multimodal offline MLLM DPO LoRA | Not migrated |
| omni 2: Qwen3-Omni Thinker GSPO LoRA separate-async | Not migrated |
| omni 3: Qwen3-TTS Talker full-parameter GRPO | Not migrated |
| omni 4: MiniCPM-o Thinker GSPO LoRA v1 sync | Not migrated |

NPU ID 2 adds Wan2.2 DanceGRPO coverage, which is not an entry in these GPU
runners at this baseline. It does not substitute for another model's GPU case.

For a complete parity claim, expand every entry into its collected pytest
node IDs and parameter combinations, including algorithms, strategies, SP,
request/step execution, reward modes, and offload variants. Record source SHA,
dependency pins, hardware, exit status, and logs per migrated case. Unsupported
backend cases must remain explicit gaps rather than being counted as passed
through skips or weakened assertions. Update this inventory when the GPU suite
changes; this migration does not claim full GPU-suite coverage.
