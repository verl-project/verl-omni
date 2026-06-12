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
"""
Unit tests for Ulysses Sequence Parallel (SP) support in Diffusers models.

Run the distributed tests with ≥ 4 GPUs:

    torchrun --nproc_per_node=4 --local-ranks-filter=0 tests/workers/test_diffusers_ulysses.py
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from verl_omni.models.diffusers.qwen_image import apply_qwen_image_ulysses_mask_fix

apply_qwen_image_ulysses_mask_fix()


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_device():
    return getattr(torch, get_device_name(), torch.cuda)


def initialize_global_process_group(timeout_second: int = 36000):
    backend = "hccl" if get_device_name() == "npu" else "nccl"
    torch.distributed.init_process_group(
        backend,
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)


_DEFAULT_MODEL_PATH = os.path.expanduser("~/models/tiny-random/Qwen-Image")


def sync_model_parameters_global(module):
    for p in module.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)


def _load_config_for_sp(sp_size: int) -> dict:
    from diffusers import AutoModel

    if not os.path.isdir(_DEFAULT_MODEL_PATH):
        pytest.skip(
            f"Tiny Qwen-Image model not found at {_DEFAULT_MODEL_PATH!r}. "
            "Provide the model or adjust _DEFAULT_MODEL_PATH."
        )

    cfg = AutoModel.load_config(_DEFAULT_MODEL_PATH, subfolder="transformer")

    orig_heads = cfg.get("num_attention_heads", 1)
    if orig_heads % sp_size != 0:
        # Round up to the nearest multiple of sp_size.
        new_heads = ((orig_heads + sp_size - 1) // sp_size) * sp_size
        cfg["num_attention_heads"] = new_heads
        if "num_key_value_heads" in cfg:
            cfg["num_key_value_heads"] = new_heads

    return cfg


# =============================================================================
# 1.  SP forward equivalence test  (distributed, ≥ sp_size GPUs required)
# =============================================================================


@pytest.mark.parametrize("sp_size", [2, 4])
@pytest.mark.parametrize("backend", ["native"])
def test_diffusers_ulysses_fwd(sp_size, backend):
    """
    Ulysses SP forward must produce numerically equivalent output to a plain
    (non-SP) forward pass on the same weights and inputs.
    """
    if not torch.distributed.is_initialized():
        initialize_global_process_group()

    world_size = torch.distributed.get_world_size()
    if world_size < sp_size:
        pytest.skip(f"Requires ≥ {sp_size} GPUs, found {world_size}")

    dp_size = world_size // sp_size
    _diffusers_ulysses_fwd(sp_size=sp_size, dp_size=dp_size, backend=backend)


def _diffusers_ulysses_fwd(sp_size: int, dp_size: int, backend: str):
    """Load a tiny diffusers model and check SP ≡ non-SP forward numerically."""
    try:
        from diffusers import AutoModel, ContextParallelConfig
    except ImportError:
        pytest.skip("diffusers package is not installed")

    assert get_torch_device().device_count() >= sp_size, f"Need at least {sp_size} GPUs for sp_size={sp_size}"

    device = get_device_name()
    rank = torch.distributed.get_rank()

    ulysses_device_mesh = init_device_mesh(
        device_type=device,
        mesh_shape=(dp_size, 1, sp_size),
        mesh_dim_names=("dp", "ring", "ulysses"),
    )

    cfg = _load_config_for_sp(sp_size)
    latent_dim = cfg.get("in_channels", 64)
    encoder_hidden_dim = cfg.get("cross_attention_dim", cfg.get("encoder_hidden_size", 32))

    module_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_sp.enable_parallelism(config=ContextParallelConfig(ulysses_degree=sp_size, mesh=ulysses_device_mesh))
    module_sp.set_attention_backend(backend)
    module_sp = module_sp.to(device=device, dtype=torch.bfloat16)
    sync_model_parameters_global(module_sp)

    module_no_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_no_sp.set_attention_backend(backend)
    module_no_sp = module_no_sp.to(device=device, dtype=torch.bfloat16)
    module_no_sp.load_state_dict(
        {k: v.detach().clone() for k, v in module_sp.state_dict().items()},
        strict=False,
    )

    batch_size = 2
    latent_h = latent_w = max(sp_size * 2, 4)
    latent_seq_len = latent_h * latent_w
    text_seq_len = sp_size * 4

    hidden_states = torch.zeros(batch_size, latent_seq_len, latent_dim, dtype=torch.bfloat16, device=device)
    encoder_hidden_states = torch.zeros(
        batch_size, text_seq_len, encoder_hidden_dim, dtype=torch.bfloat16, device=device
    )
    if rank == 0:
        torch.manual_seed(42)
        hidden_states.normal_()
        encoder_hidden_states.normal_()
    torch.distributed.broadcast(hidden_states, src=0)
    torch.distributed.broadcast(encoder_hidden_states, src=0)

    valid_text_lens = [sp_size * 1, sp_size * 3]  # both < text_seq_len
    encoder_hidden_states_mask = torch.zeros(batch_size, text_seq_len, dtype=torch.bool, device=device)
    for i, n in enumerate(valid_text_lens):
        encoder_hidden_states_mask[i, :n] = True

    timestep = torch.full([batch_size], 0.5, dtype=torch.float32, device=device)
    img_shapes = [[(1, latent_h, latent_w)]] * batch_size

    model_inputs = dict(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )

    # 1. SP forward
    module_sp.eval()
    with torch.no_grad():
        output_sp = module_sp(**model_inputs)[0]

    # 2. plain (non-SP) forward
    module_no_sp.eval()
    with torch.no_grad():
        output_no_sp = module_no_sp(**model_inputs)[0]

    assert output_sp.shape == output_no_sp.shape, f"Shape mismatch: SP {output_sp.shape} vs non-SP {output_no_sp.shape}"

    # we need a strict tolerance here to show _patch is working
    torch.testing.assert_close(output_sp.float(), output_no_sp.float(), rtol=1e-2, atol=1e-2)

    if rank == 0:
        print(
            f"[sp_size={sp_size}]  mean(SP)={output_sp.float().mean().item():.6f}  "
            f"mean(no-SP)={output_no_sp.float().mean().item():.6f}  ✓"
        )


# =============================================================================
# 2.  SP forward + backward equivalence test
# =============================================================================


@pytest.mark.parametrize("sp_size", [2, 4])
@pytest.mark.parametrize("backend", ["native"])
def test_diffusers_ulysses_fwd_bwd(sp_size, backend):
    """
    Ulysses SP backward pass must produce equivalent gradients to a plain
    (non-SP) backward pass on the same weights and inputs.
    """
    if not torch.distributed.is_initialized():
        initialize_global_process_group()

    world_size = torch.distributed.get_world_size()
    if world_size < sp_size:
        pytest.skip(f"Requires ≥ {sp_size} GPUs, found {world_size}")

    dp_size = world_size // sp_size
    _diffusers_ulysses_fwd_bwd(sp_size=sp_size, dp_size=dp_size, backend=backend)


def _diffusers_ulysses_fwd_bwd(sp_size: int, dp_size: int, backend: str):
    """Load a tiny diffusers model and compare SP vs non-SP forward+backward."""
    try:
        from diffusers import AutoModel, ContextParallelConfig
    except ImportError:
        pytest.skip("diffusers package is not installed")

    assert get_torch_device().device_count() >= sp_size, f"Need at least {sp_size} GPUs for sp_size={sp_size}"

    device = get_device_name()
    rank = torch.distributed.get_rank()

    ulysses_device_mesh = init_device_mesh(
        device_type=device,
        mesh_shape=(dp_size, 1, sp_size),
        mesh_dim_names=("dp", "ring", "ulysses"),
    )
    sp_group = ulysses_device_mesh["ulysses"].get_group()

    cfg = _load_config_for_sp(sp_size)
    latent_dim = cfg.get("in_channels", 64)
    encoder_hidden_dim = cfg.get("cross_attention_dim", cfg.get("encoder_hidden_size", 32))

    module_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_sp.enable_parallelism(config=ContextParallelConfig(ulysses_degree=sp_size, mesh=ulysses_device_mesh))
    module_sp.set_attention_backend(backend)
    module_sp = module_sp.to(device=device, dtype=torch.bfloat16)
    sync_model_parameters_global(module_sp)

    module_no_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_no_sp.set_attention_backend(backend)
    module_no_sp = module_no_sp.to(device=device, dtype=torch.bfloat16)
    module_no_sp.load_state_dict(
        {k: v.detach().clone() for k, v in module_sp.state_dict().items()},
        strict=False,
    )

    batch_size = 2
    latent_h = latent_w = max(sp_size * 2, 4)
    latent_seq_len = latent_h * latent_w
    text_seq_len = sp_size * 4

    hidden_states = torch.zeros(batch_size, latent_seq_len, latent_dim, dtype=torch.bfloat16, device=device)
    encoder_hidden_states = torch.zeros(
        batch_size, text_seq_len, encoder_hidden_dim, dtype=torch.bfloat16, device=device
    )
    if rank == 0:
        torch.manual_seed(0)
        hidden_states.normal_()
        encoder_hidden_states.normal_()
    torch.distributed.broadcast(hidden_states, src=0)
    torch.distributed.broadcast(encoder_hidden_states, src=0)

    timestep = torch.full([batch_size], 0.5, dtype=torch.float32, device=device)
    encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool, device=device)
    img_shapes = [[(1, latent_h, latent_w)]] * batch_size

    model_inputs = dict(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )

    # 1. SP forward + backward
    module_sp.train()
    output_sp = module_sp(**model_inputs)[0]
    loss_sp = output_sp.float().mean()
    loss_sp.backward()

    # 2. plain (non-SP) forward + backward
    module_no_sp.train()
    output_no_sp = module_no_sp(**model_inputs)[0]
    loss_no_sp = output_no_sp.float().mean()
    loss_no_sp.backward()

    torch.testing.assert_close(loss_sp, loss_no_sp, rtol=1e-2, atol=3e-5)

    # Each SP rank holds a partial gradient (it processes only S_local sequence
    # positions). Sum across SP ranks to reconstruct the full gradient, equivalent
    # to what the non-SP model computes over the full sequence.
    for p_sp in module_sp.parameters():
        if p_sp.grad is not None:
            torch.distributed.all_reduce(p_sp.grad, op=torch.distributed.ReduceOp.SUM, group=sp_group)

    grad_sp_list, grad_no_sp_list = [], []
    for p_sp, p_no_sp in zip(module_sp.parameters(), module_no_sp.parameters(), strict=False):
        if p_sp.grad is not None and p_no_sp.grad is not None:
            grad_sp_list.append(p_sp.grad.detach().float())
            grad_no_sp_list.append(p_no_sp.grad.detach().float())

    assert len(grad_sp_list) > 0, "No gradients were computed for SP model"

    grad_sp_vec = torch.cat([g.flatten() for g in grad_sp_list])
    grad_no_sp_vec = torch.cat([g.flatten() for g in grad_no_sp_list])

    norm_sp = grad_sp_vec.norm()
    norm_no_sp = grad_no_sp_vec.norm()
    torch.testing.assert_close(norm_sp, norm_no_sp, rtol=1e-2, atol=1e-2)

    if rank == 0:
        print(
            f"[sp_size={sp_size}]  loss(SP)={loss_sp.item():.6f}"
            f"  loss(no-SP)={loss_no_sp.item():.6f}"
            f"  |grad|(SP)={norm_sp.item():.4f}"
            f"  |grad|(no-SP)={norm_no_sp.item():.4f}  ✓"
        )


# =============================================================================
# 3.  FSDP-wrapped SP forward + backward equivalence test
# =============================================================================


@pytest.mark.parametrize("sp_size", [2, 4])
@pytest.mark.parametrize("backend", ["native"])
def test_diffusers_ulysses_fwd_bwd_fsdp(sp_size, backend):
    """
    FSDP-wrapped Ulysses SP backward must produce equivalent gradients to a
    plain non-SP backward pass.

    Mirrors the production PPODiffusersFSDPEngine (via DiffusersFSDPEngine): FSDP's reduce-scatter across
    the SP process group automatically sums the partial gradients, so no
    explicit all-reduce is needed after backward.
    """
    if not torch.distributed.is_initialized():
        initialize_global_process_group()

    world_size = torch.distributed.get_world_size()
    if world_size < sp_size:
        pytest.skip(f"Requires ≥ {sp_size} GPUs, found {world_size}")

    dp_size = world_size // sp_size
    _diffusers_ulysses_fwd_bwd_fsdp(sp_size=sp_size, dp_size=dp_size, backend=backend)


def _diffusers_ulysses_fwd_bwd_fsdp(sp_size: int, dp_size: int, backend: str):
    """
    FSDP-wrapped SP forward+backward test.

    The SP model is wrapped with FSDP using a device_mesh that covers all SP
    ranks per DP group, matching DiffusersFSDPEngine._build_fsdp_module.
    FSDP's automatic reduce-scatter removes the need for a manual all-reduce
    when comparing gradients.
    """
    try:
        from diffusers import AutoModel, ContextParallelConfig
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError:
        pytest.skip("diffusers package is not installed")

    import functools

    assert get_torch_device().device_count() >= sp_size, f"Need at least {sp_size} GPUs for sp_size={sp_size}"

    device = get_device_name()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    ulysses_device_mesh = init_device_mesh(
        device_type=device,
        mesh_shape=(dp_size, 1, sp_size),
        mesh_dim_names=("dp", "ring", "ulysses"),
    )

    cfg = _load_config_for_sp(sp_size)
    latent_dim = cfg.get("in_channels", 64)
    encoder_hidden_dim = cfg.get("cross_attention_dim", cfg.get("encoder_hidden_size", 32))

    module_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_sp.enable_parallelism(config=ContextParallelConfig(ulysses_degree=sp_size, mesh=ulysses_device_mesh))
    module_sp.set_attention_backend(backend)
    module_sp = module_sp.to(device=device, dtype=torch.bfloat16)
    sync_model_parameters_global(module_sp)

    module_no_sp = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
    module_no_sp.set_attention_backend(backend)
    module_no_sp = module_no_sp.to(device=device, dtype=torch.bfloat16)
    module_no_sp.load_state_dict(
        {k: v.detach().clone() for k, v in module_sp.state_dict().items()},
        strict=False,
    )

    # Use a 1-D FSDP mesh covering all world_size ranks so that FSDP's
    # reduce-scatter sums gradients across SP ranks automatically.
    # The ulysses_device_mesh is kept separate and used only for SP all-to-all.
    fsdp_mesh = init_device_mesh(
        device_type=device,
        mesh_shape=(world_size,),
        mesh_dim_names=["fsdp"],
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={QwenImageTransformerBlock},
    )
    module_sp = FSDP(
        module_sp,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_mesh,
        use_orig_params=True,
    )

    batch_size = 2
    latent_h = latent_w = max(sp_size * 2, 4)
    latent_seq_len = latent_h * latent_w
    text_seq_len = sp_size * 4

    hidden_states = torch.zeros(batch_size, latent_seq_len, latent_dim, dtype=torch.bfloat16, device=device)
    encoder_hidden_states = torch.zeros(
        batch_size, text_seq_len, encoder_hidden_dim, dtype=torch.bfloat16, device=device
    )
    if rank == 0:
        torch.manual_seed(0)
        hidden_states.normal_()
        encoder_hidden_states.normal_()
    torch.distributed.broadcast(hidden_states, src=0)
    torch.distributed.broadcast(encoder_hidden_states, src=0)

    timestep = torch.full([batch_size], 0.5, dtype=torch.float32, device=device)
    encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool, device=device)
    img_shapes = [[(1, latent_h, latent_w)]] * batch_size

    model_inputs = dict(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        img_shapes=img_shapes,
        return_dict=False,
    )

    # 1. FSDP-wrapped SP forward + backward
    module_sp.train()
    output_sp = module_sp(**model_inputs)[0]
    loss_sp = output_sp.float().mean()
    loss_sp.backward()

    # 2. plain non-SP forward + backward
    module_no_sp.train()
    output_no_sp = module_no_sp(**model_inputs)[0]
    loss_no_sp = output_no_sp.float().mean()
    loss_no_sp.backward()

    torch.testing.assert_close(loss_sp, loss_no_sp, rtol=1e-2, atol=3e-5)

    # FSDP sums the sp_size partial gradients (one per SP rank) via reduce-scatter
    # then divides by world_size, leaving full_grad / sp_size on each rank. Use
    # summon_full_params to gather un-sharded gradients, then scale by sp_size to
    # recover the full gradient before comparing with the non-SP model.
    grad_sp_list, grad_no_sp_list = [], []
    with FSDP.summon_full_params(module_sp, with_grads=True):
        for p_sp, p_no_sp in zip(module_sp.parameters(), module_no_sp.parameters(), strict=False):
            if p_sp.grad is not None and p_no_sp.grad is not None:
                grad_sp_list.append(p_sp.grad.detach().float() * sp_size)
                grad_no_sp_list.append(p_no_sp.grad.detach().float())

    assert len(grad_sp_list) > 0, "No gradients were computed for FSDP SP model"

    grad_sp_vec = torch.cat([g.flatten() for g in grad_sp_list])
    grad_no_sp_vec = torch.cat([g.flatten() for g in grad_no_sp_list])

    norm_sp = grad_sp_vec.norm()
    norm_no_sp = grad_no_sp_vec.norm()
    torch.testing.assert_close(norm_sp, norm_no_sp, rtol=1e-2, atol=1e-2)

    if rank == 0:
        print(
            f"[fsdp sp_size={sp_size}]  loss(SP)={loss_sp.item():.6f}"
            f"  loss(no-SP)={loss_no_sp.item():.6f}"
            f"  |grad|(SP)={norm_sp.item():.4f}"
            f"  |grad|(no-SP)={norm_no_sp.item():.4f}  ✓"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
