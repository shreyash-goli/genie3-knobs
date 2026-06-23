# genie3-rl-knobs

**RL-over-search-knobs MVP for Genie 3 binder design** — AlQuraishi Lab.

A research scaffold to answer one cheap-to-test question before investing in the harder
follow-up:

> **Can a learned policy select among our existing search/conditioning knobs better than a
> fixed heuristic or a contextual bandit?**

This is *not* a policy that fine-tunes Genie 3 or perturbs its diffusion process — that is
the eventual goal and is deliberately deferred (see *Scope* below). A positive result here
does **not** validate diffusion-score-as-action; it only tells us whether the "three levers"
framing has enough signal to justify the bigger build. **The contextual bandit is the bar
that matters** — if PPO can't beat it, the RL framing isn't earning its complexity yet.

## The three levers (action space)

| lever | values | realised in data as |
|---|---|---|
| branch timestep | diffusion-clock branch points (e.g. 700–950) | `branch_t_<T>/` sweep dirs |
| binder length delta | {0, +30, +60} residues | base vs `_longbinder` problem variant |
| hotspot conditioning mode | {all, missed-only, ablate-competitors} | `_only_*` / `_ablate_*` variants |

State = target id (+ context: best ipTM/i_pAE seen, exploration so far). Reward = sparse,
terminal: a swappable weighted blend of success / ipTM / interface-pAE / hotspot coverage /
diversity (`oracle/reward_oracle.py::compute_reward`).

## Offline-first

`env.step()` does **not** run Genie 3. The Stage-0 ingester parses the **already-computed**
sweep outputs on `/pscratch` (4,050 evaluated children: `iptm`, `avg_interface_pae`,
`complex_success`, …) into an offline labelled dataset, and the env samples a real logged
child for the chosen lever cell. This makes the full fixed/random/bandit/PPO comparison
runnable with **zero GPU and no new generation**. The live oracle
(Genie3 → ProteinMPNN → ColabFold) is a documented stub (`oracle/reward_oracle.py::
LiveRewardModel`) to be wired later — it should *shell out* to the genie3 env, not import it.

## Staged plan (build/validate in order)

- **Stage 0** — logging infra: `instrumentation/trajectory_logger.py` (live hook + offline
  ingester). *Done; 4,050 records.*
- **Stage 1** — minimal demo: timestep-only action space; fixed/random/bandit/PPO.
  **Stopping rule:** if PPO can't beat the bandit, stop and diagnose before Stage 2.
- **Stage 2** — full 3-lever (timestep × length × hotspot) action space; same comparison.
- **Stage 3** — two-target contrast (BHRF1 where length helped vs InsulinR where it didn't):
  does the learned policy act *target-dependently*?
- **Stage 4 (OUT OF SCOPE — clean stubs only)** — frontier buffer
  (`buffer/frontier_buffer.py`, LatProtRL Alg 3 signatures), commitment-window-restricted
  fine-tuning, diffusion-score-as-action.

## Layout

```
config.py                     paths + action-space constants (override via RLKNOBS_* env vars)
instrumentation/              Stage-0 logger + /pscratch sweep ingester   (spec's logging/, renamed*)
oracle/reward_oracle.py           compute_reward() + OfflineRewardModel
oracle/live_oracle.py             LiveRewardModel: Genie3 → ProteinMPNN → ColabFold shell-out
envs/genie_branch_env.py          Gym env over the levers (1- or 3-lever, data-driven)
envs/commitment_window.py         Per-target commitment window detector + intervention MDP
baselines/                        fixed_heuristic, random_policy, contextual_bandit (the real bar)
policy/train_ppo.py               Stable-Baselines3 PPO over the discrete action space
policy/lora_finetune.py           LoRA adapter attachment + PPO fine-tuning loop for V1Denoiser
buffer/frontier_buffer.py         Diffusion x_T seed cache (LatProtRL Alg 3, lever-indexed)
experiments/compare_timestep_lever.py     4-way comparison on timestep-only action space (gate)
experiments/compare_full_levers.py        4-way comparison on full 3-lever action space
experiments/contrast_bhrf1_insulinr.py    Target-dependent behaviour check (BHRF1 vs InsulinR)
experiments/validate_live_oracle.py       Sanity-check live vs offline reward agreement
experiments/benchmark_frontier_seeding.py Cold-start vs buffer-seeded rollout comparison
experiments/train_intervention_policy.py  Commitment window detection + intervention PPO
experiments/finetune_genie3_lora.py       LoRA fine-tuning of V1Denoiser (GPU, genie3 env)
external/genie3               local checkout (symlink*); external/latprotrl_ref (reference)
tests/                        env shapes, compute_reward, bandit-learns-optimum, ingest parsing
```

\* **Two deliberate deviations from the spec tree**, both documented:
1. `logging/` → `instrumentation/` — a top-level `logging` package shadows Python's stdlib
   `logging` once on `sys.path` and breaks SB3/torch. 2. `external/genie3` is a **symlink** to
   the local `~/genie3` checkout, not a git submodule: this NERSC node has no GitHub network
   access. Run `setup_genie3_submodule.sh` on a networked machine to convert it to a pinned
   submodule (`external/genie3_pinned_commit.txt`).

## Environment

**NERSC GPU node setup** — before running any live oracle experiment, fix the cuDNN mismatch
(jaxlib 0.6.2 requires cuDNN 9.8.0 but the conda env ships 9.5.1):
```bash
pip install "nvidia-cudnn-cu12==9.8.0.87"
```
Torch pins 9.5.1 in its metadata but the two runtimes (PyTorch for Genie3, JAX for ColabFold)
don't share cuDNN at runtime so this is safe. Re-run on each new interactive session — pip
installs are not persisted across salloc allocations if your home is network-mounted read-only.

Use the **`genie3` conda env**. SB3 is pinned `<2.7` so it coexists with genie3's
`torch==2.7.1` (SB3 ≥2.7 demands `torch>=2.8` and would clobber that pin).

```bash
conda activate genie3
pip install -r requirements.txt          # numpy, pandas, gymnasium, stable-baselines3<2.7, pytest
```

## Run

```bash
conda activate genie3
python -m instrumentation.trajectory_logger      # Stage 0: build data/records.jsonl (+ sqlite)
#   add --coverage to also compute hotspot contact fractions from PDBs (slower)
python -m pytest -q                               # tests

python -m experiments.compare_timestep_lever       # gate: PPO vs bandit (1-D)
python -m experiments.compare_full_levers          # full 3-lever space
python -m experiments.contrast_bhrf1_insulinr      # target-dependent behaviour check

# Stages 4-7 (GPU node, genie3 conda env):
python -m experiments.validate_live_oracle         # sanity-check live oracle pipeline
python -m experiments.benchmark_frontier_seeding   # cold vs buffer-seeded (offline)
python -m experiments.train_intervention_policy    # commitment windows + intervention PPO
python -m experiments.finetune_genie3_lora         # LoRA fine-tuning (GPU required)
```

Results land in `data/experiment_logs/<name>/{summary.json,episodes.jsonl}` and are
summarised in [`experiments/RESULTS.md`](experiments/RESULTS.md).

## What's still open (do not pre-decide in code)

- Whether the frontier buffer is indexed jointly by the three levers or by reward alone.
- Whether per-target commitment windows (some lock at step 10, others past 30) need a
  per-target adaptive index.
- Whether diffusion-score-as-action is tractable, or needs a lower-dim fallback (x̂₀ nudge,
  conditioning-embedding perturbation).

## Background / load-bearing references

LatProtRL (Lee et al., 2024) — MDP + PPO + Frontier Buffer; Genie 2/3 — SE(3)-equivariant
diffusion backbone generation; Proteina-Complexa — which structure metrics correlate with
binding. See the project spec for the full context (commitment window, hotspot failure
modes, length-ablation findings).
