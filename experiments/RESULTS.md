# Results — RL-over-search-knobs MVP (PI update)

*All numbers below are from this session's runs (genie3 conda env, offline oracle, CPU).
Reproduce with `python -m experiments.stage{1,2,3}_*`; raw logs in
`data/experiment_logs/<stage>/{summary.json,episodes.jsonl}`. Reward is the swappable blend
in `oracle/reward_oracle.py` (success-dominated); **absolute values are modest by
construction — read the rankings and the per-target policy, not the magnitudes**.*

## TL;DR for the PI

- We built the **validation infrastructure**, not the final architecture: a PPO policy over
  our existing knobs (timestep / length / hotspot), trained + evaluated entirely on the
  **already-computed sweep data** (4,050 evaluated children, **zero new GPU cost**).
- **PPO passes the Stage-1 gate** (beats the contextual bandit on every seed) — but at the
  1-D level the margin is **small and sensitive to how hard we tune the bandit**. The case
  for a learned policy gets stronger as the action space grows (Stage 2) and is **clearest
  qualitatively in Stage 3**: PPO learns the *target-dependent* length behaviour that matches
  our prior ablations; the bandit does not.
- Two honesty flags up front: (1) Stage 2/3 lean ~40–46% on the offline simulator's
  documented back-off because the hotspot/length levers were only ever logged at one
  timestep; (2) a warm-started bandit nearly closes the Stage-1 gap. Both shape what the
  live-oracle / Stage-4 work should do first.

---

## Stage 0 — logging infrastructure ✅

Ingested **4,050** evaluated children from `/pscratch` sweeps (`trajectory_branching_v2`,
`trajectory_branching`, `hotspot_ablation`) → `data/records.jsonl` + SQLite, each row
carrying the three levers, oracle metrics (`iptm`, `avg_interface_pae`, `complex_success`,
…), a sequence-diversity proxy, and the backbone PDB path (logged now for future surrogate
use). **Lever coverage is asymmetric**: the timestep lever is dense (6 timesteps × 10 targets
× 40 children in v2); the hotspot/length levers were varied only at `branch_t_800`
(`hotspot_ablation`). This asymmetry is the single biggest caveat on Stages 2–3.

## Stage 1 — timestep-only action space (the gate)

|A| = 6 timesteps · 10 targets · 3 seeds · 200 eval/target. **Data grounding: 100% exact
cells** (no back-off).

| policy | mean reward |
|---|---|
| random | 0.2960 ± 0.0060 |
| fixed (best single timestep) | 0.3171 ± 0.0044 |
| bandit — cold (UCB) | 0.3005 ± 0.0133 |
| bandit — warm-started from lookup table | 0.3166 ± 0.0115 |
| **PPO** | **0.3361 ± 0.0016** |

**Verdict: PPO clears the gate** (beats both bandit variants on every seed) → Stage 2 is
warranted. **But read it cautiously:** the *cold* bandit under-performs even the fixed
heuristic, and warm-starting it (the "lookup table you already have") lifts it to ≈ fixed and
**halves PPO's margin (Δ 0.036 → 0.019)**. On the 1-D action space, RL does *not* convincingly
earn its complexity — its edge is small and bandit-tuning-sensitive.

PPO did, however, already learn a **target-dependent timestep split** (targets 01–05 → branch
t=700; targets 06–10 → t=950), i.e. the optimal branch point is target-dependent — direct
evidence for a per-target (not fixed) commitment window.

## Stage 2 — full three-lever action space

Levers = timestep × length × hotspot, on the 7 targets that actually have lever variation in
the data; 3 seeds · 200 eval/target. **Data grounding: 60.5% exact, 30.5% `drop_hotspot`,
9.0% `drop_length`** (back-off, per `episodes.jsonl`).

| policy | mean reward |
|---|---|
| random | 0.2843 ± 0.0057 |
| fixed | 0.3773 ± 0.0041 |
| bandit (UCB) | 0.3052 ± 0.0258 |
| **PPO** | **0.4375 ± 0.0063** |

PPO beats the bandit by **+0.132** and the fixed heuristic by +0.060. The gap over the bandit
*widens* with the larger action space — consistent with PPO's parametric policy generalising
across cells while the bandit must estimate each (target × lever) cell independently under a
fixed budget. Learned policy: extend length (+60) @ t=800 for bhrf1/sc2rbd/pdl1/h1; no
extension @ t=950 for insulinr/vegfa/il17a. **Caveat:** ~40% of these evaluations sit on
backed-off (interpolated) cells, so the full-lever margin is partly a property of the
simulator's back-off, not purely measured data.

## Stage 3 — two-target contrast (the real question)

BHRF1 (length *helped* recover missed hotspots) vs InsulinR (length *did not* generalise).
3 seeds · 300 eval/target. **Data grounding: 53.6% exact, 31.5% `drop_hotspot`, 14.9%
`drop_length`.**

| policy | mean reward |
|---|---|
| random | 0.3906 ± 0.0054 |
| fixed | 0.4372 ± 0.0059 |
| bandit (UCB) | 0.4848 ± 0.0520 |
| **PPO** | **0.5794 ± 0.0097** |

**The point of a learned policy is target-dependent action selection, and PPO shows it:**

| target | PPO chooses | bandit chooses |
|---|---|---|
| BHRF1 (length helped) | t=800, **length +60**, all-hotspots | length +60 |
| InsulinR (length didn't) | t=900, **length +0**, ablate-competitors | length +60 |

PPO **extends the binder on BHRF1 but not on InsulinR** — exactly the contrast our earlier
length ablation found by hand — and additionally picks a different hotspot-conditioning mode
for InsulinR. The bandit extends length on **both** targets, i.e. it fails to differentiate.
This is the strongest qualitative evidence in the MVP that the lever framing carries
target-specific signal a single fixed heuristic cannot capture.

---

## Does PPO earn its complexity? (the honest answer)

- **Yes at the level of "is there exploitable, target-dependent structure here": ** PPO
  recovers known target-specific behaviour (Stage 3) and beats every baseline at every stage.
- **Not yet conclusively at the level of "do we need RL specifically":** on the clean,
  fully-measured 1-D problem its edge over a *well-tuned* bandit is small (Δ ≈ 0.02); its big
  wins (Stages 2–3) ride partly on interpolated cells. The framing has real signal and is
  worth extending — but the next dollar should go to removing the two caveats below, not to
  jumping to diffusion-score-as-action.

## What Stage 4 should do differently as a result

1. **Generate the missing real cells first.** The hotspot/length levers exist in data at only
   one timestep, so ~40–46% of Stage 2/3 evaluation is interpolation. Before trusting the
   full-lever / contrast conclusions, run the live oracle (Genie3 → ProteinMPNN → ColabFold,
   via `oracle/reward_oracle.py::LiveRewardModel`) to fill the
   (timestep × length × hotspot) grid for at least the BHRF1/InsulinR pair.
2. **Harden the baseline.** The cold bandit under-performed `fixed`; a warm-started bandit
   nearly caught PPO at 1-D. Stage-4 must keep beating the *strongest* lookup/bandit, or the
   RL win is a tuning artefact.
3. **Frontier buffer should index by levers, not reward alone.** Stage 3 shows the best lever
   *combination* is target-dependent; a reward-only buffer index would discard that. Seed the
   buffer design from the BHRF1/InsulinR contrast.
4. **Per-target commitment window, not a fixed one.** PPO split targets across branch
   timesteps unprompted (t=700 vs t=950 in Stage 1) — evidence the commit point is
   target-dependent, supporting per-target window detection before scaling.
5. **Only then** revisit the eventual goal (x̂₀ / conditioning-embedding nudge → full
   diffusion-score-as-action), now with a working PPO loop and a measured-data simulator to
   regression-test against.
