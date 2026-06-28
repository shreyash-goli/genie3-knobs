# genie3-rl-knobs

**RL-over-search-knobs for Genie3 binder design** — AlQuraishi Lab.

> **Can a learned policy select among Genie3's conditioning knobs better than a fixed
> heuristic or a contextual bandit?**

A positive result here validates the three-levers framing and justifies the harder
follow-up: steering the diffusion score directly. **The contextual bandit is the bar
that matters** — if PPO can't beat it, the RL framing isn't earning its complexity.

---

## Workflow Design

### Problem framing

Genie3 generates protein binders via DDIM diffusion over 1000 timesteps. Given a target
protein with known hotspot residues, Genie3 conditions its denoising process on those
residues and produces candidate binder sequences. The choices of *when* to branch, *how
long* to make the binder, and *which hotspot subset* to condition on are currently set by
hand or grid-searched. The goal is to learn a policy that makes these choices adaptively
per target.

### The three levers (action space)

| Lever | Values | What it controls |
|---|---|---|
| `branch_timestep` | {700, 750, 800, 850, 900, 950} | When in the diffusion trajectory to branch — earlier = more stochastic, later = more committed to a fold class |
| `length_delta` | {0, +60 AAs} | Binder length relative to default |
| `hotspot_mode` | `all` / `ablate_competitors` / `missed_only` | Which hotspot residues to condition on: `all` uses the full set, `ablate_competitors` suppresses off-target contacts, `missed_only` forces residues the model historically misses |

These are discrete pre-defined choices that map to dataset selection strings Genie3 reads
from a problem JSON. Conditioning is binary per residue — there is no continuous strength
dial.

### Commitment window

The **commitment window** is the range of diffusion timesteps during which structural
decisions are still being made. After this window the fold is committed and interventions
have little effect. It is detected per target from offline data by measuring reward
variance across children branched at each timestep: high variance = structure still
deciding, low variance = committed. For the current targets this spans approximately
t=700–950 (out of 1000 total diffusion steps).

### MDP formulation

```
Episode  = one protein binder design attempt for one target
Steps    = N_WINDOW_STEPS = 10, uniformly spaced across the commitment window
Action   = Discrete(3): which hotspot_mode to apply at this diffusion timestep
Length   = sampled once at reset() from {0, 60 AAs}, fixed for the whole episode
Reward   = intermediate_reward_scale × compute_reward(metrics) at steps 0–8
           compute_reward(metrics) at step 9 (terminal, unscaled)
```

`intermediate_reward_scale=0.0` gives the original sparse terminal behaviour.
`intermediate_reward_scale=0.1` provides per-step shaping signal while keeping
the terminal reward dominant.

### Observation space

At each step the policy receives a float32 vector of size `n_targets + 5`:

| Component | Description |
|---|---|
| Target one-hot | Which target this episode is for |
| `step_progress` | Steps completed / 10 |
| `t_norm` | Current diffusion timestep / 1000 |
| `length_delta_norm` | 0.0 or 1.0 (length delta drawn at reset) |
| `best_iptm` | Running max ipTM seen so far this episode |
| `best_neg_ipae` | Running max of (1 − pAE/30) seen so far this episode |

### Policy network

A small MLP actor-critic (no pretrained weights, built fresh per experiment):

```
Input (obs_dim)
  → Linear(obs_dim, 64) → Tanh
  → Linear(64, 64) → Tanh        [shared trunk]
  ├→ Linear(64, 3)                [policy head: logits over hotspot modes]
  └→ Linear(64, 1)                [value head: PPO baseline]
```

At each step: obs → shared trunk → Categorical(policy logits) → sample action.
During eval: argmax. No recurrence — all within-episode memory is in the obs vector
via the running `best_iptm` / `best_neg_ipae` fields.

### PPO training loop

Standard PPO with a rollout buffer:

1. `env.reset()` — sample target uniformly, sample `length_delta`, build timestep schedule
2. For each of 10 steps: obs → policy → sample action → `env.step()` → oracle → reward → buffer
3. Every `ppo_update_freq=32` episodes: compute GAE advantages, iterate minibatches,
   PPO clipped policy loss + value loss + entropy bonus, gradient step on MLP weights

The actor-critic MLP is standalone — Genie3 weights are not touched during this loop.
LoRA fine-tuning of the Genie3 denoiser is a separate experimental track.

### Oracle: offline vs live

Both backends implement the same interface:
`.sample(target, timestep, hotspot_mode, length_delta) → (metrics_dict, backoff_level)`

**Offline oracle** (`OfflineRewardModel`): samples a real logged record from
`data/records.jsonl`. If the exact (target, timestep, hotspot_mode, length_delta) cell is
empty, backs off through a documented hierarchy — drop length match → drop hotspot match →
nearest timestep → target-global. Backoff level is logged per step. ~4,050 records,
instant lookup (microseconds per step).

**Live oracle** (`LiveRewardModel`): shells out to the Genie3 conda env, runs
`trajectory_branching.py` then `eval.py`, parses `child_*_metrics.json`. ~10 min per call.
RL stack and Genie3 stay in separate conda envs — never imported in-process.

All PPO training uses the offline oracle. Live oracle is used only for final validation
and LoRA fine-tuning.

### Reward function

`compute_reward(metrics)` in `oracle/reward_oracle.py` — pure function, no I/O.
The reward is arithmetic on numbers ColabFold already produced; it adds negligible compute.

```
reward = Σ(weight_i × value_i) / Σ(weights_present)
```

| Term | Weight | Computation | Status |
|---|---|---|---|
| `success` | 1.0 | 1.0 if ipTM≥0.80 AND pAE≤10.0, else 0.0 | active |
| `interface_iptm` | 0.5 | ipTM clipped to [0, 1] | active |
| `interface_ipae` | 0.5 | 1 − (pAE / 30) | active |
| `hotspot_coverage` | 0.5 | fraction of specified hotspot residues contacted (5Å geometry) | **stubbed — always None** |
| `diversity` | 0.1 | 1 − max sequence identity to siblings in same cell | active but noisy offline |
| `ics` | 0.5 (planned) | avg predicted contact probability at interface contacts (Promera) | **not yet implemented** |

Terms whose inputs are `None` are dropped and the denominator renormalised. With
`hotspot_coverage` always None, the effective reward is:
`(1.0×success + 0.5×iptm + 0.5×(1−pAE/30)) / 2.0`

### Baselines

Each experiment compares PPO against:
- **Random**: uniform random hotspot mode each step
- **Fixed-all / fixed-ablate / fixed-missed**: always picks the same mode for all 10 steps;
  best of these three is the "optimal bandit" floor
- **UCB bandit** (earlier one-shot experiments): upper-confidence-bound over the action space

### Frontier buffer (Go-Explore, currently inactive)

`buffer/frontier_buffer.py` stores `x_T` — the initial Gaussian noise vector at t=1000
from which a denoising trajectory starts — from high-reward live episodes. Future episodes
can be seeded from this buffer rather than fresh noise, so Genie3 resumes from a
known-good starting point (lightweight Go-Explore). Seeding experiment showed no
improvement (−0.017 vs cold start); implemented but not currently load-bearing.

### LoRA fine-tuning (separate experimental track)

Attaches LoRA adapters (r=8, α=16) to Genie3's V1Denoiser IPA + LatentTransformer layers,
then uses PPO to fine-tune those adapter weights — so the diffusion model itself learns to
generate better binders rather than just selecting among pre-defined configurations.
Requires live oracle. Current result: +0.167 pre→post, n=3 eval episodes (not reportable).

### What is not yet load-bearing

| Component | Blocker |
|---|---|
| Real iCS metric | Needs raw ColabFold PAE matrices; no code in `oracle/` yet |
| Hotspot coverage | Needs PDB geometry check post-ColabFold; stubbed as None |
| PPO beating fixed in windowed MDP | Offline data too sparse for non-`all` modes (~40% backoff) |
| LoRA result | Needs SLURM re-run with n≥10 eval episodes |
| Window-start validation | Preliminary; re-run after lever grid is filled |

---

## Experiment log

See [`RESULTS.md`](RESULTS.md) for full results through the windowed MDP and window-start
sweep. Summary of key findings:

| Finding | Evidence |
|---|---|
| PPO learns better timestep selection than a bandit | Stage 1: +0.036, 3 seeds consistent |
| Adding length + hotspot levers amplifies PPO's advantage | Stage 2: +0.132 vs bandit |
| Warm-starting the bandit doesn't close the gap | Stages 1b, 2b: PPO still wins |
| PPO finds qualitatively different strategies per target | Stage 3: InsulinR → `ablate_competitors` at t=900; bandit missed this |
| Direction scale is not a useful lever | Intervention policy: +0.003 over fixed |
| Offline training generalises to live oracle | Live validation: BHRF1 +0.155 above offline mean |
| Sparse terminal reward fails in the 10-step windowed MDP | Windowed PPO: training curve flat, PPO −0.013 vs fixed |
| Intermediate reward shaping unblocks the signal | Training curve lifts from 0.000 to ~0.04–0.05 |
| Offline data sparsity is now the binding constraint | PPO still −0.015 vs fixed at 2000 episodes with shaping |
| window_start=850 gives best PPO mean reward in placement sweep | Window sweep: best fixed-baseline outcomes at 800–850 |

---

## Code layout

```
config.py                          paths + action-space constants
instrumentation/                   offline ingester + trajectory logger
oracle/reward_oracle.py            compute_reward() + OfflineRewardModel
oracle/live_oracle.py              LiveRewardModel: Genie3 → ProteinMPNN → ColabFold
envs/genie_branch_env.py           one-shot lever-selection env (Stages 1–3)
envs/commitment_window.py          commitment window detector + windowed MDP
baselines/                         fixed, random, UCB bandit
policy/lora_finetune.py            actor-critic MLP + PPO loop + LoRA attach
buffer/frontier_buffer.py          x_T seed cache (Go-Explore style)
experiments/
  compare_timestep_lever.py        Stage 1: PPO vs bandit, timestep only
  compare_full_levers.py           Stage 2: full 3-lever space
  contrast_bhrf1_insulinr.py       Stage 3: target-dependent behaviour check
  validate_live_oracle.py          live vs offline reward sanity check
  benchmark_frontier_seeding.py    cold vs buffer-seeded rollout comparison
  train_intervention_policy.py     commitment window detection + intervention PPO
  finetune_genie3_lora.py          LoRA fine-tuning of V1Denoiser (GPU)
  ppo_vs_bandit_offline.py         windowed MDP: PPO vs bandit with intermediate reward
  window_start_sweep.py            sweep window_start boundary per guidance-interval paper
tests/                             env shapes, compute_reward, bandit, ingest parsing
```

---

## Next steps

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for the prioritised queue. In order:

1. **Fill lever grid** — run live oracle to fill ~8 missing (hotspot_mode × length_delta)
   cells at branch_t=800 for BHRF1 and InsulinR; removes the ~40% backoff caveat
2. **LoRA re-run** — 500 offline train + 10 pre/post live eval episodes on SLURM (m5016, 4h)
3. **Add iCS** — implement `_compute_ics()` in `oracle/live_oracle.py`; recompute offline
   records; add as reward term (see NEXT_STEPS.md §4 for the 6-step implementation plan)
4. **Add hotspot coverage** — PDB geometry check in `_aggregate_children()`
5. **Re-run windowed PPO + window sweep** after grid fill and iCS

---

## Environment setup

**NERSC GPU node** — fix cuDNN mismatch before any live oracle run:
```bash
pip install "nvidia-cudnn-cu12==9.8.0.87"
```
Re-run each new interactive session (pip installs not persisted across salloc on
network-mounted home).

Use the **`genie3` conda env**. SB3 is pinned `<2.7` to coexist with `torch==2.7.1`.

```bash
conda activate genie3
pip install -r requirements.txt
python -m instrumentation.trajectory_logger   # build data/records.jsonl
python -m pytest -q
```

## Run

```bash
# Offline experiments (no GPU, seconds to minutes):
python -m experiments.compare_timestep_lever
python -m experiments.compare_full_levers
python -m experiments.contrast_bhrf1_insulinr
python -m experiments.ppo_vs_bandit_offline
python -m experiments.window_start_sweep

# Live oracle experiments (GPU node, genie3 env, hours):
python -m experiments.validate_live_oracle
python -m experiments.benchmark_frontier_seeding
python -m experiments.finetune_genie3_lora

# SLURM (4h, m5016):
sbatch --account=m5016 --time=04:00:00 --nodes=1 --gpus=1 \
       --constraint=gpu --qos=regular \
       --wrap="conda run -n genie3 python -m experiments.finetune_genie3_lora"
```

Results land in `data/experiment_logs/<name>/results.json`.
