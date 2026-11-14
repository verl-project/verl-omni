#!/usr/bin/env bash
# Real SD3.5 Medium FlowGRPO smoke: generation, rule reward, actor updates,
# weight sync, validation, checkpoint save. Re-run with resume_mode=auto to resume.
# Set CUDA_VISIBLE_DEVICES and MODEL_PATH before running; one GPU is sufficient.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}
DATA_DIR=${DATA_DIR:-/tmp/flowgrpo_sd35_colocate_async_data}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/flowgrpo_sd35_colocate_async}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "$DATA_DIR" --train_size 32 --val_size 2

bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_colocate_async.sh \
    "data.train_files=$DATA_DIR/train.parquet" \
    "data.val_files=$DATA_DIR/test.parquet" \
    data.train_batch_size=2 \
    data.val_max_samples=2 \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.checkpoint.save_lora_only=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=6 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=6 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.reward_model.enable=false \
    reward.reward_model.enable_resource_pool=false \
    reward.models=null \
    reward.reward_functions=null \
    reward.reward_manager.name=VisualRewardManager \
    reward.custom_reward_function.path=null \
    reward.num_workers=1 \
    trainer.n_gpus_per_node=1 \
    trainer.logger='[console]' \
    trainer.val_before_train=true \
    trainer.test_freq=1 \
    trainer.save_freq=1 \
    trainer.log_val_generations=0 \
    "trainer.total_training_steps=$TOTAL_TRAIN_STEPS" \
    "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
    "trainer.validation_data_dir=$OUTPUT_DIR/validation" \
    "trainer.rollout_data_dir=$OUTPUT_DIR/rollouts" \
    trainer.resume_mode=disable \
    trainer.v1.colocate_async.num_warmup_batches=2 \
    trainer.v1.sampler.max_off_policy_threshold=2 \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    ray_kwargs.ray_init.num_cpus=24 \
    ++ray_kwargs.ray_init.object_store_memory=2147483648 \
    ++ray_kwargs.ray_init.include_dashboard=false \
    "$@"

# Check saved state as well as process exit: an early trainer return is not a pass.
python3 - "$OUTPUT_DIR" "$TOTAL_TRAIN_STEPS" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

output, expected_step = Path(sys.argv[1]), int(sys.argv[2])
root = output / "checkpoints"
assert int((root / "latest_checkpointed_iteration.txt").read_text()) == expected_step
actor = root / f"global_step_{expected_step}" / "actor"

def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor

def load_model(step):
    state = torch.load(root / f"global_step_{step}" / "actor/model_world_size_1_rank_0.pt",
                       map_location="cpu", weights_only=False)
    return {k: local(v) for k, v in state.items()}

model = load_model(expected_step)
assert model and all(torch.isfinite(v).all() for v in model.values())
assert any(torch.count_nonzero(v) for k, v in model.items() if "lora_B" in k)
optimizer = torch.load(actor / "optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
active_states = [v for v in optimizer["state"].values() if "step" in v]
steps = {int(local(v["step"])) for v in active_states}
assert steps == {2 * expected_step}
assert all(torch.isfinite(local(v[key])).all() for v in active_states
           for key in ("exp_avg", "exp_avg_sq"))
changed = None
if expected_step > 1:
    previous = load_model(expected_step - 1)
    changed = sum(not torch.equal(v, previous[k]) for k, v in model.items())
    assert changed > 0
assert (root / f"global_step_{expected_step}" / "transfer_queue/metadata.json").exists()
for step in range(1, expected_step + 1):
    rows = [json.loads(line) for line in (output / f"rollouts/{step}.jsonl").read_text().splitlines()]
    groups = Counter(row["uid"].rsplit("_", 2)[0] for row in rows)
    assert len(rows) == 4 and len(groups) == 2 and set(groups.values()) == {2}
    validation = [json.loads(line) for line in (output / f"validation/{step}.jsonl").read_text().splitlines()]
    assert len(validation) == 2
    for row in [*rows, *validation]:
        assert math.isfinite(row["score"])
        with Image.open(row["output"]) as image:
            assert image.size == (384, 384)
            assert any(low < high for low, high in image.getextrema())
(output / "smoke_verification.json").write_text(json.dumps({
    "step": expected_step, "optimizer_steps": sorted(steps), "finite_lora_tensors": len(model),
    "changed_tensors_from_previous_step": changed,
    "complete_groups_per_step": 2, "images_per_step": 4, "image_size": [384, 384],
}, indent=2) + "\n")
PY
