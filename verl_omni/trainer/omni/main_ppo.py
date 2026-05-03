"""
Entry point for PPO/GSPO training with verl-omni patches.

Importing verl_omni triggers all monkey-patches (Qwen3-Omni model support,
FSDP fixes, reward scoring, rollout replica registration, etc.) before
handing off to the upstream verl trainer.

Usage:
    python3 -m verl_omni.trainer.omni.main_ppo [hydra overrides...]
"""
import verl_omni  # noqa: F401  — triggers patches + replica registration
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
