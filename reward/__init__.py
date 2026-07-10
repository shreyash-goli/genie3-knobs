"""Reward-function reform (NEXT_STEPS.md §2).

Candidate terminal-reward designs + the no-GPU experiments that decide which one to adopt,
kept separate from the production `oracle/reward_oracle.py` so nothing here can perturb the
live training path until a design is chosen. See reward/README.md for the design rationale
and the experiment ladder (Tier B ranking-validity → Tier A form ablation → Tier C retrain).
"""
