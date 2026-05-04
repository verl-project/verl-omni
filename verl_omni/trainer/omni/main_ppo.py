"""
Entry point for PPO/GSPO training with verl-omni.

Importing verl_omni triggers rollout replica registration before
handing off to the upstream verl trainer.

Usage:
    python3 -m verl_omni.trainer.omni.main_ppo [hydra overrides...]
"""
import verl_omni  # noqa: F401  — triggers patches + replica registration
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
