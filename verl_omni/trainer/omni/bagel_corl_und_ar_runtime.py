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
"""UND AR replica runtime for ``bagel_corl_sync``: colocated launch, wake, and verified weight publish."""

from __future__ import annotations

import logging
import time
from typing import Any

from verl_omni.trainer.omni.bagel_corl_config import actor_merges_lora

logger = logging.getLogger(__name__)


def as_ray_actor_class(cls: Any) -> Any:
    """Return ``cls`` as a Ray actor class, tolerating both ``verl`` shapes (decorated or plain)."""
    if hasattr(cls, "remote"):
        return cls
    import ray

    return ray.remote(cls)


def flatten_rpc_results(value: Any) -> list[Any]:
    """Flatten ``AsyncOmni.collective_rpc``'s per-stage, per-worker nesting into one list."""
    if isinstance(value, list | tuple):
        return [leaf for item in value for leaf in flatten_rpc_results(item)]
    return [value]


def launch_und_ar_replica(und_cfg, *, actor_pool, replica_rank: int, und_n_gpus: int, und_deploy: str, actor_model):
    """Create one UND AR replica colocated on the first ``und_n_gpus``-wide slice of the actor pool."""
    from verl.single_controller.ray.base import split_resource_pool
    from verl.utils.ray_utils import auto_await
    from verl.workers.rollout.replica import get_rollout_replica_class

    und_rollout = und_cfg.actor_rollout_ref.rollout
    und_model = und_cfg.actor_rollout_ref.model
    split_pools = split_resource_pool(actor_pool, split_size=und_n_gpus)
    if not split_pools:
        raise RuntimeError("bagel_corl_sync: actor resource pool is empty; cannot colocate UND AR")
    replica_cls = get_rollout_replica_class(str(und_rollout.get("name") or "vllm_omni"))
    replica = replica_cls(
        replica_rank=replica_rank,
        config=und_rollout,
        model_config=und_model,
        gpus_per_node=int(und_rollout.get("n_gpus_per_node") or und_n_gpus),
        name_suffix="bagel_und_ar",
    )
    logger.info(
        "bagel_corl_sync colocating UND AR on actor pool start_rank=%s und_n_gpus=%s und_deploy=%s",
        replica_rank,
        und_n_gpus,
        und_deploy,
    )
    logger.info(
        "bagel_corl_sync UND AR engine lora_rank=%s actor_lora_merge=%s (0 => full-weight sync, no engine adapter)",
        int(und_model.get("lora_rank", 0) or 0),
        actor_merges_lora(actor_model),
    )

    @auto_await
    async def _init_und():
        await replica.init_colocated(split_pools[0])

    _init_und()
    return replica


def build_dual_role_client(config, und_cfg, und_replica, gen_manager):
    """Wrap the UND AR replica and the GEN hybrid behind one ``BagelDualRoleLLMServerClient``."""
    from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer, LLMServerClient

    from verl_omni.workers.rollout.bagel_dual_role_llm_server import BagelDualRoleLLMServerClient

    und_rollout = und_cfg.actor_rollout_ref.rollout
    und_lb = as_ray_actor_class(GlobalRequestLoadBalancer).remote(
        servers={und_replica.server_address: und_replica.server_handle},
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism=bool(getattr(und_rollout, "full_determinism", False)),
    )
    und_client = LLMServerClient(config=und_cfg, load_balancer_handle=und_lb)
    return BagelDualRoleLLMServerClient(config=config, und_client=und_client, gen_client=gen_manager.get_client())


def wake_replicas(replicas: list, *, label: str) -> None:
    """Wake every server of ``replicas`` with both tags, sequentially, then report the count."""
    if not replicas:
        return
    from verl.utils.ray_utils import auto_await

    @auto_await
    async def _wake_all() -> None:
        for replica in replicas:
            for server in replica.servers:
                await server.wake_up.remote(tags=["weights", "kv_cache"])

    _wake_all()
    logger.info(
        "bagel_corl_sync woke %s %s replica(s) (weights+kv_cache, sequentially) after sleep", len(replicas), label
    )


def _verify_pushed_fingerprints(
    server, replica_rank, checksums: dict, check_names, global_steps, fingerprints_match
) -> None:
    """Compare the actor-side checksum of each sentinel against what the engine reports."""
    import ray

    for name in check_names:
        if checksums.get(name) is None:
            raise RuntimeError(
                f"bagel_corl_sync: actor export carried no {name!r}; UND AR name routing no longer matches the Thinker"
            )
        engine_values = [
            v
            for v in flatten_rpc_results(
                ray.get(server.collective_rpc.remote("policy_param_checksum", kwargs={"name": name}))
            )
            if v is not None
        ]
        if not engine_values or not all(fingerprints_match(checksums[name], v) for v in engine_values):
            raise RuntimeError(
                f"bagel_corl_sync step={global_steps}: UND AR replica {replica_rank} did not take the pushed "
                f"weights; {name} fingerprint sent={checksums[name]!r} engine={engine_values!r}"
            )


def publish_und_ar_weights(replicas: list, *, actor_rollout_wg, receiver_ranks: list[int], check_names, global_steps):
    """Push the merged actor policy into each UND AR Thinker, verify the sentinels, and sleep the server.

    Returns ``(num_tensors, fingerprint, seconds)``.
    """
    import ray

    from verl_omni.workers.rollout.vllm_rollout.weight_fingerprint import fingerprints_match

    started = time.perf_counter()
    num_tensors = 0
    fingerprint = None
    for replica in replicas:
        if len(replica.servers) != 1:
            raise RuntimeError(
                f"bagel_corl_sync: UND AR replica {replica.replica_rank} has {len(replica.servers)} "
                "servers; the policy push assumes a single-node replica"
            )
        server = replica.servers[0]
        ray.get(server.wake_up.remote(tags=["weights", "kv_cache"]))
        try:
            results = ray.get(
                actor_rollout_wg.update_bagel_und_ar_weights(
                    server_handle=server,
                    und_replica_rank=replica.replica_rank,
                    receiver_ranks=receiver_ranks,
                    checksum_name=check_names[0],
                    checksum_names=list(check_names),
                    global_steps=global_steps,
                )
            )
            stats = next((r for r in flatten_rpc_results(results) if r is not None), None)
            checksums = {} if stats is None else stats.get("checksums", {})
            _verify_pushed_fingerprints(
                server, replica.replica_rank, checksums, check_names, global_steps, fingerprints_match
            )
        finally:
            ray.get(server.sleep.remote())
        num_tensors += int(stats["num_tensors"])
        fingerprint = stats["checksum"]
    return num_tensors, fingerprint, time.perf_counter() - started
