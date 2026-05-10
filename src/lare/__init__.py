"""LaRe (Latent Reward) integration for LDRP.

Two systems are designed:
- System A (LaRe-Path): step-level proxy reward for path planning (IQL/QMIX).
- System B (LaRe-Task): assignment-level proxy reward for task assignment (PPO).

This package currently implements System A only. System B is left as future work.
"""
