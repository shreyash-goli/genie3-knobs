# genie3-rl-knobs

**RL-over-search-knobs for Genie3 binder design** 

> **Can a learned policy select among Genie3's conditioning knobs better than a fixed
> heuristic or a contextual bandit?**

A positive result here validates the three-levers framing and justifies the harder
follow-up: steering the diffusion score directly. **The contextual bandit is the bar
that matters** — if PPO can't beat it, the RL framing isn't earning its complexity.

**Current answer (2026-07, windowed MDP):** on the 10-step commitment-window MDP, **PPO does
not beat a per-target contextual bandit** (Δ ≈ −0.015, inside the ±0.024 seed noise floor).
PPO's only real margin is over *random* conditioning (+0.10). Ablations (`target_onehot_ablation`,
`window_start_ablation`) trace PPO's apparent "win" over a single fixed action entirely to
**cross-target specialization** — the same effect the earlier one-shot Stage 1–3 experiments
found — not to learning *when* to switch hotspot mode within an episode. The problem is
contextual-bandit-shaped, not sequential-RL-shaped. See `NEXT_STEPS.md` §3.1 for the full
evidence chain. This does not close the FK-Steering axis (§7) or the generalization-to-unseen-
targets question (§5), both still open.

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
Action   = Discrete(4): 3 hotspot_mode arms + a learned `commit` action (index 3)
Length   = sampled once at reset() from {0, 60 AAs}, fixed for the whole episode
Reward   = intermediate_reward_scale × compute_reward(metrics) at non-terminal steps
           compute_reward(metrics) at the terminal step (unscaled)
```

`intermediate_reward_scale=0.0` gives the original sparse terminal behaviour.
`intermediate_reward_scale=0.1` provides per-step shaping signal while keeping
the terminal reward dominant.

The **`commit` action** (index 3, *not* one of the 3 `HOTSPOT_MODES` — so the fixed-bandit
baselines are unaffected) ends the episode early and takes the *unscaled* terminal reward
from the current step; in live mode it fires the single per-episode oracle call at the commit
step. Otherwise the episode times out after 10 steps. `step()` info carries
`termination_reason` (`"commit"` / `"timeout"`). Note the commit action was built for
formulation completeness — after the live-oracle fix (one oracle call per episode regardless),
it no longer saves calls, and §3.1 found no within-episode value for it to exploit.

### Observation space

At each step the policy receives a float32 vector of size `n_targets + 5 + len(HOTSPOT_MODES)`
(= `n_targets + 8`):

| Component | Description |
|---|---|
| Target one-hot | Which target this episode is for (zeroed if `mask_target_onehot=True`, for ablations) |
| `step_progress` | Steps completed / 10 |
| `t_norm` | Current diffusion timestep / 1000 |
| `length_delta_norm` | 0.0 or 1.0 (length delta drawn at reset) |
| `best_iptm` | Running max ipTM seen so far this episode |
| `best_neg_ipae` | Running max of (1 − pAE/30) seen so far this episode |
| Action history (×3) | Per-hotspot-mode usage counts so far this episode, normalised by 10 |

### Policy network

A small MLP actor-critic (no pretrained weights, built fresh per experiment):

```
Input (obs_dim)
  → Linear(obs_dim, 64) → Tanh
  → Linear(64, 64) → Tanh        [shared trunk]
  ├→ Linear(64, n_actions)        [policy head: logits over 3 hotspot modes + commit]
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
nearest timestep → target-global. Backoff level is logged per step. ~4,075 records,
instant lookup (microseconds per step). The 25-cell lever-grid fill (2026-07-07) removed
most of the earlier backoff caveat (30 combined hotspot+length cells still lack a backing
problem JSON — see `NEXT_STEPS.md` §1.3).

**Live oracle** (`LiveRewardModel`): shells out to the Genie3 conda env, runs
`trajectory_branching.py` then `eval.py`, parses `child_*_metrics.json`. ~10 min per call.
RL stack and Genie3 stay in separate conda envs — never imported in-process.

All PPO training uses the offline oracle. Live oracle is used only for final validation
and LoRA fine-tuning.

### Reward function

`compute_reward(metrics)` in `oracle/reward_oracle.py` — pure function, no I/O. As of
2026-07-16 this is the **tiered** structure from `NEXT_STEPS.md` §2.2, not a flat weighted
average:

```
designable AND hotspot_coverage ≥ threshold:  reward = 1.0 + coverage×3.0 + iptm×1.0
designable only:                              reward = 0.5 − (1 − coverage_or_0.5)/5.0
not designable:                               reward = −complex_scrmsd / 10.0
```

`designable` reuses the existing `complex_success`/thresholded-iptm+ipae check. The nuance
term uses normalized `iptm` as a stand-in for the design doc's `iptm_energy` (a free-energy
quantity from pre-softmax PAE logits ColabFold doesn't expose through this pipeline yet —
future work, not approximated further than this). Reward is no longer bounded to `[0,1]`:
roughly `[-3, 5]`. Falls back to the old flat weighted average
(`_legacy_weighted_average`) only when a metrics dict has no success-relevant signal at all
(e.g. partial dicts). `diversity` is not part of the tiered formula (dropped per §2.2's spec,
not carried forward as an extra term).

| Term | Role | Computation | Status |
|---|---|---|---|
| `complex_success` (designable) | tier gate | ipTM≥0.80 AND pAE≤10.0, or ColabFold's own success flags | active |
| `hotspot_coverage` | tier gate + tier-1 scale | fraction of specified hotspot residues contacted (8Å Cβ geometry) | computed on live calls; being backfilled on existing offline records (grid re-fill job, 2026-07-16) |
| `iptm` | tier-1 nuance (stand-in for `iptm_energy`) | ipTM clipped to [0, 1] | active |
| `complex_scrmsd` | fail-tier scale | ColabFold scRMSD | active |
| `ics` | not yet wired in | avg predicted contact probability at interface contacts (Promera-style) | implemented in `live_oracle.py`; still not a `compute_reward` term — same backfill blocker as hotspot_coverage |

**iCS** (`_compute_ics`) and **hotspot coverage** (`_compute_hotspot_coverage`) both landed
in `oracle/live_oracle.py` (2026-07-07) and are validated live end-to-end. A live-oracle
re-fill job (42 BHRF1/InsulinR lever cells, `experiments/fill_lever_grid.py --no-skip-existing`)
was submitted 2026-07-16 to backfill both on existing offline records (the raw ColabFold PAE
matrix wasn't saved when those cells first ran).

### Baselines

Each experiment compares PPO against:
- **Random**: uniform random hotspot mode each step
- **Fixed-all / fixed-ablate / fixed-missed**: always picks the same mode for all 10 steps;
  best of these three is the "optimal single-action" floor
- **Per-target contextual bandit** (`baselines/contextual_bandit.py`): UCB over the 3 hotspot
  arms, conditioned on target identity, one constant arm replayed across the window. This is
  **the** baseline that decides whether PPO earns its complexity — it *can* specialize per
  target, which a single fixed action cannot. Wired into the windowed-MDP comparison
  (`ppo_vs_bandit_offline.py`) 2026-07-12; PPO does not beat it (§3.1).

### Frontier buffer (Go-Explore, currently inactive)

`buffer/frontier_buffer.py` stores `x_T` — the initial Gaussian noise vector at t=1000
from which a denoising trajectory starts — from high-reward live episodes. Future episodes
can be seeded from this buffer rather than fresh noise, so Genie3 resumes from a
known-good starting point (lightweight Go-Explore). Seeding experiment showed no
improvement (−0.017 vs cold start); implemented but not currently load-bearing.

### LoRA fine-tuning (separate experimental track) — currently a misnomer, see NEXT_STEPS.md §8

Attaches LoRA adapters (r=8, α=16) to Genie3's V1Denoiser IPA + LatentTransformer layers, but
**`train_lora_ppo` never actually trains them** (found 2026-07-16) — its optimizer only
touches the small lever-selection MLP; the diffusion model's gradients are structurally
blocked (subprocess generation, plus `torch.inference_mode()` in genie3's sampler). The
existing n=3 result (+0.167 pre→post) reflects the already-known lever-selection-transfer
result, not diffusion-model adaptation. A real fix (DDPO-style policy gradients through the
sampler's per-step Gaussian — confirmed feasible against genie3's actual code, genie3's
`DDIMSampler` is stochastic with `eta=1.0` by default) is scoped in `NEXT_STEPS.md` §8 but not
implemented — multi-day effort, needs its own session.

### What is not yet load-bearing

| Component | Status / blocker |
|---|---|
| iCS as a reward term | `_compute_ics` implemented & live-validated; not yet in `compute_reward`; grid re-fill job (2026-07-16, job 55986875) backfilling it on offline records |
| Hotspot coverage in reward | Now gates the tiered `compute_reward` (2026-07-16); still `None` on offline records not yet covered by the grid re-fill job |
| PPO beating the contextual bandit | Does not, on the windowed MDP (§3.1) — believed to be a genuine finding (contextual-bandit-shaped problem), not a data-sparsity artifact |
| LoRA result | Not just "needs a re-run" — `train_lora_ppo` doesn't train the LoRA adapter at all (found 2026-07-16, `NEXT_STEPS.md` §8). Held off pending either a relabeled actor_critic-only run or the real DDPO-style fix |
| FK-Steering live phase | Offline best-of-k cut built (`fk_steering.py`); true mid-trajectory kill/duplicate needs genie3 partial-state cloning + a usable intermediate potential (§7.1 found the cheap geometric ones too weak) |
| Structural policy encoder / generalization | Flat MLP can't transfer to unseen targets; §5 redesign not started |

---

## Experiment log

Summary of key findings:

| Finding | Evidence |
|---|---|
| PPO learns better timestep selection than a bandit (one-shot env) | Stage 1: +0.036, 3 seeds consistent |
| Adding length + hotspot levers amplifies PPO's advantage (one-shot) | Stage 2: +0.132 vs bandit |
| Warm-starting the bandit doesn't close the gap | Stages 1b, 2b: PPO still wins |
| PPO finds qualitatively different strategies per target | Stage 3: InsulinR → `ablate_competitors` at t=900; bandit missed this |
| Direction scale is not a useful lever | Intervention policy: +0.003 over fixed |
| Offline training generalises to live oracle | Live validation: BHRF1 +0.155 above offline mean |
| ~~Sparse terminal reward fails in the windowed MDP~~ **was a training-loop bug** | The training curve "flat at 0.000" was a loop that only ever stepped `step_count==0` and never reached the terminal reward; fixed 2026-07-06 (`NEXT_STEPS.md` §0.1) |
| After the fix, PPO beats fixed+random in `ppo_vs_bandit_offline` | 0.593 vs 0.579 (best fixed) vs 0.575 (random) — but this margin is cross-target specialization only |
| The windowed-MDP "win" is entirely cross-target specialization | `target_onehot_ablation`: Δ vs fixed flips +0.010 → −0.034 when the target one-hot is masked |
| Single-target PPO loses to a constant action at 11/12 configs | `window_start_ablation` (10 placements + 2 natural windows) |
| **PPO does not beat the per-target contextual bandit** | `ppo_vs_bandit_offline` (2026-07-12): Δ = −0.015, inside the ±0.024 noise floor; PPO's only real margin is +0.10 over random |
| Cheap geometric FK potentials (rg, nc_termini) too weak to steer on | `fk_correlation_test`: overall Pearson −0.23 (rg), +0.00 (nc_termini), n=125 |
| Branch-point `x_T` summaries barely predict terminal reward | `xt_state_probe`: Spearman ≈ 0.14–0.19, n=25 — soft no-go for richer within-episode state |
| Measurement noise floor for identical configs is ±0.024 | 5-seed sweep (§6); clip ε=0.1 the only consistently-positive cheap fix (+0.011) |

---

## Code layout

```
config.py                          paths + action-space constants
instrumentation/                   offline ingester + trajectory logger
oracle/reward_oracle.py            compute_reward() + OfflineRewardModel
oracle/live_oracle.py              LiveRewardModel + iCS / hotspot-coverage / property metrics
oracle/branching_wrapper.py        genie3 trajectory_branching + eval.py bridge
envs/genie_branch_env.py           one-shot lever-selection env (Stages 1–3)
envs/commitment_window.py          commitment window detector + windowed MDP (Discrete(4), commit)
baselines/                         fixed_heuristic, random_policy, contextual_bandit (UCB, per-target)
policy/lora_finetune.py            actor-critic MLP + PPO loop + LoRA attach + GAE buffer
policy/fk_steering.py              FK-Steering resample + rollout (offline best-of-k cut)
buffer/frontier_buffer.py          x_T seed cache (Go-Explore style)
experiments/
  compare_timestep_lever.py        Stage 1: PPO vs bandit, timestep only
  compare_full_levers.py           Stage 2: full 3-lever space
  contrast_bhrf1_insulinr.py       Stage 3: target-dependent behaviour check
  validate_live_oracle.py          live vs offline reward sanity check
  benchmark_frontier_seeding.py    cold vs buffer-seeded rollout comparison
  train_intervention_policy.py     commitment window detection + intervention PPO
  finetune_genie3_lora.py          LoRA fine-tuning of V1Denoiser (GPU)
  ppo_vs_bandit_offline.py         windowed MDP: PPO vs contextual bandit + fixed + random
  window_start_sweep.py            sweep window_start boundary per guidance-interval paper
  window_start_ablation.py         single-target rerun of the sweep (isolates specialization)
  target_onehot_ablation.py        mask target one-hot → confirms cross-target specialization
  fill_lever_grid.py               live-oracle lever-grid fill (multi-GPU)
  fk_correlation_test.py           intermediate-reward correlation gate (§7.1)
  fk_vs_ppo_offline.py             FK-Steering 2×2 comparison (§7.3)
  xt_state_probe.py                branch-point x_T vs terminal-reward probe (§3.3)
tests/                             env shapes, compute_reward, bandit, GAE, FK, ingest parsing
```

---

## Next steps

Done since the last README revision: lever grid filled (25/25 cells), iCS and hotspot
coverage implemented and live-validated, commit action + action-history obs added, FK-Steering
offline cut and the contextual-bandit windowed comparison wired in, full FK-vs-PPO offline 2×2
run at N_TRAIN=500/N_EVAL=200 (`4=0.637 > 3=0.510 > 2=0.487 > 1=0.415` — search and policy are
complementary, not redundant). As of 2026-07-16: reward reform implemented (tiered
`compute_reward`), live-oracle grid re-fill submitted to backfill hotspot_coverage/iCS (job
55986875), and the LoRA re-run held off after finding `train_lora_ppo` never actually trains
the LoRA adapter (`NEXT_STEPS.md` §8). Remaining:

1. **Merge the grid re-fill results** into `data/records.jsonl` once job 55986875 completes,
   confirm hotspot_coverage/iCS are populated, re-check the tiered reward against real data
2. **LoRA path decision** — either run `finetune_genie3_lora.py` as an honestly-relabeled
   actor_critic-transfers-to-live-oracle check (n≥10), or implement the real DDPO-style fix
   scoped in `NEXT_STEPS.md` §8 (multi-day effort: sampler training-path variant, diffusion
   log-prob, trajectory storage, new training loop)
3. **FK-Steering live phase** — mid-trajectory kill/duplicate needs genie3 partial-state cloning
   and a usable intermediate potential (the cheap geometric ones failed the §7.1 gate)
4. **Structural policy encoder** — the §5 redesign, the only path to generalizing beyond the
   current memorized targets
5. **Contact-geometry reward terms** (`iptm_energy` proper, `i_con`) — the tiered reward
   currently stands in `iptm` for `iptm_energy`; a real free-energy term needs raw pre-softmax
   PAE logits ColabFold doesn't expose through this pipeline yet

The detailed, cross-checked design log lives in [`NEXT_STEPS.md`](NEXT_STEPS.md).

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

# SLURM (4h, m4351):
sbatch --account=m4351 --time=04:00:00 --nodes=1 --gpus=1 \
       --constraint=gpu --qos=regular \
       --wrap="conda run -n genie3 python -m experiments.finetune_genie3_lora"
```

Results land in `data/experiment_logs/<name>/results.json`.
