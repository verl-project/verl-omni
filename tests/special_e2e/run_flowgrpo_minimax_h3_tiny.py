#!/usr/bin/env python
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
"""MiniMax-H3 T2VA, FL2VA, and Ref2VA FlowGRPO GPU smoke runner.

Assembles one minimal ``verl_omni.trainer.main_diffusion`` invocation per task
with a self-contained tiny random-weight MiniMax-H3 checkpoint, synthetic
parquet data, a deterministic test-local joint video/audio reward, CPS
reverse-SDE rollout transitions and log-probabilities, and a one-step
policy-gradient actor update. FL2VA uses an embedded PNG first-frame condition
and ``frame_indices=[0]`` to exercise its fixed-row replay mask. Ref2VA uses a
single embedded PNG reference image and ``pipeline.task=ref2va`` to exercise the
reference-block layout, condition-anchor replay, and target-only scoring path.

Usage:
    python tests/special_e2e/run_flowgrpo_minimax_h3_tiny.py \
        --num-gpus 2 --total-steps 2

Env overrides:
    MODEL_PATH  Tiny H3 checkpoint (default ``~/models/tiny-random/minimax-h3``)
    DATA_DIR    Dummy task parquet root (default ``~/data/dummy_h3_flowgrpo``)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

# Allow execution from the repository root without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tests.special_e2e.build_minimax_h3_tiny_random import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as _DEFAULT_TINY_MODEL_DIR,
)
from tests.special_e2e.build_minimax_h3_tiny_random import ensure_tiny_minimax_h3_checkpoint  # noqa: E402
from tests.special_e2e.create_dummy_h3_fl2va_data import build_dummy_h3_fl2va_data  # noqa: E402
from tests.special_e2e.create_dummy_h3_ref2va_data import build_dummy_h3_ref2va_data  # noqa: E402
from tests.special_e2e.create_dummy_h3_t2av_data import build_dummy_h3_data  # noqa: E402

_DEFAULT_DATA_DIR = os.path.expanduser("~/data/dummy_h3_flowgrpo")


def _require_minimax_h3_diffusers() -> None:
    """Fail before Ray startup when the actor's Diffusers class is unavailable."""
    import diffusers

    if not hasattr(diffusers, "MiniMaxH3Transformer3DModel"):
        raise RuntimeError(
            "MiniMax H3 FlowGRPO requires Diffusers revision "
            "245d78fb48f1c87dfb560a94bea6e191c9f9f1c0; see "
            "examples/flowgrpo_trainer/minimax_h3/README.md."
        )


def _fixup_ld_library_path() -> None:
    """Prefer the venv cuDNN over the incompatible host CUDA cuDNN."""
    venv_site = sysconfig.get_paths()["purelib"]
    cudnn_dir = os.path.join(venv_site, "nvidia", "cudnn", "lib")
    if not os.path.isdir(cudnn_dir):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(":") if part and part != "/usr/local/cuda/lib64"]
    os.environ["LD_LIBRARY_PATH"] = ":".join([cudnn_dir, *parts])


def _hydra_overrides(
    *,
    tiny_model_dir: str,
    train_parquet: str,
    val_parquet: str,
    reward_stub_path: str,
    output_dir: str,
    task: str,
    num_gpus: int,
    rollout_tp: int,
    text_encoder_tp: int,
    total_training_steps: int,
    ray_num_cpus: int,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
) -> list[str]:
    """Build a minimal task-specific MiniMax H3 FlowGRPO Hydra invocation."""
    micro_bsz_per_gpu = 1
    # MiniMax H3 FlowGRPO supports one output per request, so Ref2VA (which
    # runs through the same rollout adapter) uses a single prompt response.
    n_resp_per_prompt = 1 if task == "ref2va" else 2
    mini_bsz = max(1, num_gpus * micro_bsz_per_gpu)
    train_batch_size = mini_bsz * n_resp_per_prompt
    partition_dir = "Ref2VA" if task == "ref2va" else "FL2VA"
    fl2va = f"{tiny_model_dir}/{partition_dir}"
    actor_transformer = f"{tiny_model_dir}/transformer"
    h3_lora_targets = "['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

    overrides = [
        # data
        f"data.train_files={train_parquet}",
        f"data.val_files={val_parquet}",
        f"data.train_batch_size={train_batch_size}",
        "data.val_max_samples=2",
        "data.max_prompt_length=256",
        "data.truncation=error",
        "data.seed=42",
        # FlowGRPO policy-gradient dispatch.
        "algorithm.trainer_type=policy_gradient",
        "algorithm.sample_source=online",
        "algorithm.adv_estimator=flow_grpo",
        "algorithm.global_std=True",
        # model
        f"actor_rollout_ref.model.path={fl2va}",
        f"actor_rollout_ref.model.config_path={actor_transformer}",
        "+actor_rollout_ref.model.architecture=MiniMaxH3Pipeline",
        "actor_rollout_ref.model.algorithm=flow_grpo",
        "actor_rollout_ref.model.transformer_subfolder=transformer",
        "actor_rollout_ref.model.attn_backend=native",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.lora_rank=8",
        "actor_rollout_ref.model.lora_alpha=16",
        f"actor_rollout_ref.model.target_modules={h3_lora_targets}",
        "actor_rollout_ref.model.fsdp_layer_prefixes=['transformer_blocks.','token_refiner.refiner_blocks.']",
        (
            "+actor_rollout_ref.actor.fsdp_config.wrap_policy."
            "transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]"
        ),
        # actor
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.actor.optim.weight_decay=1e-4",
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]",
        "actor_rollout_ref.actor.optim.override_optimizer_config={eps: 1e-8}",
        "actor_rollout_ref.actor.optim.clip_grad=1.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_bsz}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bsz_per_gpu}",
        "actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1",
        # rollout
        "actor_rollout_ref.rollout.name=vllm_omni",
        "actor_rollout_ref.rollout.max_num_seqs=1",
        "actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.n={n_resp_per_prompt}",
        "actor_rollout_ref.rollout.seed=42",
        f"actor_rollout_ref.rollout.agent.num_workers={max(1, num_gpus // rollout_tp)}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=True",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_bsz_per_gpu}",
        f"actor_rollout_ref.rollout.pipeline.task={task}",
        f"actor_rollout_ref.rollout.pipeline.height={height}",
        f"actor_rollout_ref.rollout.pipeline.width={width}",
        f"actor_rollout_ref.rollout.pipeline.num_frames={num_frames}",
        "actor_rollout_ref.rollout.pipeline.aspect_ratio=16:9",
        "actor_rollout_ref.rollout.pipeline.frame_rate=24.0",
        f"actor_rollout_ref.rollout.pipeline.num_inference_steps={num_inference_steps}",
        "actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0",
        "actor_rollout_ref.rollout.pipeline.max_sequence_length=256",
        "actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0",
        "+actor_rollout_ref.rollout.pipeline.output_type=np",
        "actor_rollout_ref.rollout.algo.noise_level=0.8",
        "actor_rollout_ref.rollout.algo.sde_type=cps",
        "actor_rollout_ref.rollout.algo.sde_window_size=2",
        "actor_rollout_ref.rollout.algo.sde_window_range=[0,3]",
        "actor_rollout_ref.rollout.algo.sde_contiguous=True",
        "actor_rollout_ref.rollout.algo.sde_window_seed=42",
        # val kwargs are retained for config completeness; validation is disabled.
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.task={task}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.height={height}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.width={width}",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames={num_frames}",
        "actor_rollout_ref.rollout.val_kwargs.pipeline.aspect_ratio=16:9",
        "actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0",
        f"actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps={num_inference_steps}",
        "actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt",
        "actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0",
        # vLLM-Omni engine kwargs: the field is top-level, not parallel_config.
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.text_encoder_tp_size={text_encoder_tp}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_bsz_per_gpu}",
        # reward: local stub only; no CLAP, ImageBind, or reward-model weights.
        "reward.num_workers=1",
        "reward.reward_model.enable=False",
        f"reward.custom_reward_function.path={reward_stub_path}",
        "reward.custom_reward_function.name=compute_score",
        "reward.aggregation=weighted_sum",
        # trainer
        "trainer.logger=console",
        "trainer.project_name=verl-test",
        f"trainer.experiment_name=flowgrpo-minimax-h3-tiny-{task}",
        f"trainer.default_local_dir={output_dir}/checkpoints",
        f"trainer.validation_data_dir={output_dir}/validation_data",
        f"trainer.rollout_data_dir={output_dir}/rollout_data",
        "trainer.log_val_generations=0",
        "trainer.video_fps=24",
        "trainer.val_before_train=False",
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_training_steps}",
        f"ray_kwargs.ray_init.num_cpus={ray_num_cpus}",
    ]
    if task == "fl2va":
        overrides.extend(
            [
                "actor_rollout_ref.rollout.pipeline.frame_indices=[0]",
                "actor_rollout_ref.rollout.val_kwargs.pipeline.frame_indices=[0]",
            ]
        )
    elif task == "ref2va":
        overrides.append("actor_rollout_ref.rollout.pipeline.reference_image_short_edge=256")
    return overrides


def run_smoke(
    *,
    task: str,
    tiny_model_dir: str,
    data_dir: str,
    output_dir: str,
    num_gpus: int,
    rollout_tp: int,
    text_encoder_tp: int,
    total_training_steps: int,
    ray_num_cpus: int,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    force_rebuild: bool,
) -> int:
    """Run one task-specific MiniMax H3 FlowGRPO training step."""
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"unsupported MiniMax H3 FlowGRPO smoke task: {task!r}")

    tiny_model_dir = os.path.expanduser(tiny_model_dir)
    data_dir = os.path.expanduser(data_dir)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] ensuring tiny MiniMax-H3 checkpoint at {tiny_model_dir}", flush=True)
    ensure_tiny_minimax_h3_checkpoint(tiny_model_dir, skip_if_exists=not force_rebuild)

    micro_bsz_per_gpu = 1
    # Match the rollout n in ``_hydra_overrides``: Ref2VA supports one output per
    # request, while T2VA and FL2VA sample two responses per prompt.
    n_resp_per_prompt = 1 if task == "ref2va" else 2
    train_batch_size = max(1, num_gpus * micro_bsz_per_gpu) * n_resp_per_prompt
    print(f"[2/3] ensuring dummy {task} parquet at {data_dir}", flush=True)
    if task == "t2va":
        train_parquet, val_parquet = build_dummy_h3_data(
            data_dir,
            train_size=train_batch_size,
            val_size=2,
        )
    elif task == "fl2va":
        train_parquet, val_parquet = build_dummy_h3_fl2va_data(
            data_dir,
            train_size=train_batch_size,
            val_size=2,
            # vLLM-Omni requires each source condition-image side to be >=256.
            image_width=max(width, 256),
            image_height=max(height, 256),
        )
    else:
        # Ref2VA reference images must be >=256 per side and have a short edge
        # of 256, the smallest value vLLM-Omni accepts.
        train_parquet, val_parquet = build_dummy_h3_ref2va_data(
            data_dir,
            train_size=train_batch_size,
            val_size=2,
            image_width=max(width, 256),
            image_height=max(height, 256),
        )

    reward_stub_path = str(_REPO_ROOT / "tests" / "special_e2e" / "minimax_h3_dummy_reward.py")
    assert os.path.isfile(reward_stub_path), reward_stub_path

    _fixup_ld_library_path()
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    overrides = _hydra_overrides(
        tiny_model_dir=tiny_model_dir,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        reward_stub_path=reward_stub_path,
        output_dir=output_dir,
        task=task,
        num_gpus=num_gpus,
        rollout_tp=rollout_tp,
        text_encoder_tp=text_encoder_tp,
        total_training_steps=total_training_steps,
        ray_num_cpus=ray_num_cpus,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    )
    cmd = [sys.executable, "-m", "verl_omni.trainer.main_diffusion", *overrides]
    print(
        f"[3/3] launching FlowGRPO {task.upper()} main_diffusion (num_gpus={num_gpus}, tp={rollout_tp}, "
        f"te_tp={text_encoder_tp}, steps={total_training_steps})",
        flush=True,
    )
    print("  cmd:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny MiniMax-H3 T2VA, FL2VA, and Ref2VA FlowGRPO GPU smoke coverage."
    )
    parser.add_argument("--tiny-model-dir", default=os.environ.get("MODEL_PATH", _DEFAULT_TINY_MODEL_DIR))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR))
    parser.add_argument(
        "--task",
        choices=("all", "t2va", "fl2va", "ref2va"),
        default=os.environ.get("TASK", "all"),
        help="Run all coverage paths (default) or one task while debugging.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "outputs" / "run_flowgrpo_minimax_h3_tiny"),
    )
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", 2)))
    parser.add_argument("--rollout-tp", type=int, default=int(os.environ.get("ROLLOUT_TP", 1)))
    parser.add_argument("--text-encoder-tp", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=int(os.environ.get("TOTAL_TRAINING_STEPS", 2)))
    parser.add_argument("--ray-num-cpus", type=int, default=int(os.environ.get("RAY_NUM_CPUS", 16)))
    # H3 requires 4-15 output seconds; 97 frames at 24 fps is 4.04 seconds.
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require_minimax_h3_diffusers()
    tasks = ("t2va", "fl2va", "ref2va") if args.task == "all" else (args.task,)
    for index, task in enumerate(tasks, start=1):
        print(f"===== MiniMax-H3 FlowGRPO {task.upper()} smoke ({index}/{len(tasks)}) =====", flush=True)
        rc = run_smoke(
            task=task,
            tiny_model_dir=args.tiny_model_dir,
            data_dir=os.path.join(args.data_dir, task),
            output_dir=os.path.join(args.output_dir, task),
            num_gpus=args.num_gpus,
            rollout_tp=args.rollout_tp,
            text_encoder_tp=args.text_encoder_tp,
            total_training_steps=args.total_steps,
            ray_num_cpus=args.ray_num_cpus,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            force_rebuild=args.force_rebuild and index == 1,
        )
        if rc != 0:
            sys.exit(rc)
    print("MiniMax-H3 tiny FlowGRPO T2VA + FL2VA + Ref2VA smoke PASSED.")


if __name__ == "__main__":
    main()
