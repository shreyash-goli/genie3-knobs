# Reward-function reform (NEXT_STEPS.md §2)

Candidate terminal-reward designs and the **no-GPU experiments** that decide which one to
adopt — kept out of `oracle/` so evaluating them cannot perturb the live training path. A
winner gets promoted into `oracle/reward_oracle.py::compute_reward` only once the evidence
here justifies it.

```
reward/
  reward_designs.py        the candidate reward functions + registry
  developability.py        sequence-only developability terms (GRAVY/charge/pI/instability)
  retro_metrics.py         attach hotspot_coverage/iCS to kept scratch designs (no GPU)
  tier_b_ranking_validity.py   THE decisive experiment: is a reward VALID (ranks correctly)?
  tier_a_form_ablation.py      at-scale form/safety + developability check on all records
  test_reward_designs.py       unit tests
```

Run everything in the `genie3` env (numpy + BioPython + the repo deps live there):

```bash
conda activate genie3
python -m reward.tier_b_ranking_validity      # kept scratch, ~1 s
python -m reward.tier_a_form_ablation         # all 4075 records, ~3 s
python -m pytest reward/test_reward_designs.py -q
```

---

## 1. The current reward and its gap

`oracle/reward_oracle.py::compute_reward` is a **flat, renormalised weighted average**:

```
reward = (1.0·success + 0.5·iptm + 0.5·(1−pae/30) + 0.5·hotspot_cov + 0.1·diversity)
         / (sum of weights whose inputs are non-None)
```

Two problems, both confirmed against the data (4075 records):

- **Geometry-blind in practice.** `hotspot_coverage` carries weight 0.5 but is `None` in
  **0 / 4075** records, so it is always dropped. `ics` is computed in `live_oracle.py` but
  `compute_reward` never reads it. The only geometry-adjacent term that fires is the *global*
  `avg_interface_pae`, which cannot see *which* residues are contacted — the InsulinR failure
  mode (good global pAE, misses hotspots).
- **No explicit failure penalty.** Failure just gets a low positive score, not a negative, so
  "failed" is not cleanly distinguished from "succeeded but mediocre."

---

## 2. Reward function designs

All are pure `metrics_dict → float` callables with the same signature as `compute_reward`.
Registry: `reward_designs.REWARD_DESIGNS`.

| design | idea | reward range |
|---|---|---|
| **`current`** | today's production baseline, geometry forced blind (coverage/ics = None) | ~[0, 1] |
| **`current+coverage`** | *minimal* reform: same flat form, just let the existing coverage term fire | ~[0, 1] |
| **`gated`** ⭐ | signed, efficiency-scaled, coverage-gated (see below) | ~[−1, 1.5] |
| **`gated+dev`** | `gated` + sequence-only developability penalty (GRAVY/charge) | ~[−1.3, 1.5] |

**`gated` (recommended)** delivers the two behaviours a *tiered* scheme is really after —
without literal tiers and without any unavailable term:

```
quality = 0.5·iptm + 0.5·geometry              # geometry = iCS if present, else (1 − pae/30)
SUCCESS (designable):  reward = coverage · (0.5 + 1.0·quality)       ≥ 0
FAILURE (not desig.):  reward = −(0.7 + 0.3·(1 − quality))           ∈ [−1.0, −0.7]
```

1. **Failure is explicitly negative** (property (1)): a strong flat **floor** (−0.7) makes every
   failure clearly bad — all failures land in [−1.0, −0.7], strictly below any success — plus a
   small **slope** so a worse fold is a bit more negative, keeping a learning gradient in the
   failure region (the §6 value-starvation concern). The floor/slope split covers all modes:
   hybrid (default), flat (`failure_slope=0`), graded (`failure_floor=0`).
2. **Success magnitude scales with efficiency** (property (2)): the positive grows with
   `quality` (iptm + interface geometry) and is **gated by hotspot coverage**, so an
   epitope-missing impostor (designable, coverage→0) collapses to ~0 — sitting *between* real
   failure (<0) and genuine on-target success (up to 1.5).

Uses **no unavailable term.** `coverage_missing` pins the §2.1 decision: `neutral` (gate
disabled when coverage absent — the failure/efficiency behaviour still works on legacy records)
vs `strict` (missing coverage = 0, the right rule once coverage is populated everywhere).

> The literal three-tier block from NEXT_STEPS.md §2.2 was **removed**. It was a copy of
> Proteina-Complexa's *beam-search* scoring function — a different policy for a different
> problem — and it required `iptm_energy` (raw pre-softmax PAE logits, persisted nowhere).
> `gated` reproduces the two behaviours that mattered with terms we have.

### Developability (`developability.py`) — sequence-only, no GPU
GRAVY, net charge, pI, and instability index, all from `binder_seq` (present on all 4075
records). `WithDevelopability` composes onto any design. Defaults penalise **GRAVY and charge
only**; pI/instability weights default to 0 (see finding below).

---

## 3. Experiment ladder — validity first, GPU last

The reform's success criterion is **NOT** "PPO beats the bandit by more" — §3.1/§6 showed that
margin lives inside the ±0.024 noise floor and is just cross-target specialisation. The
reward's job is to be **valid**: rank genuine binders above geometry-blind impostors. Validity
is a ranking property you test by *re-scoring*, no training required.

| tier | question | data | cost |
|---|---|---|---|
| **B** | is the reward valid (ranks correctly)? | kept scratch + retro coverage | free, no GPU |
| **A** | does the form move the argmax cell? does developability add signal? | all 4075 records | free, no GPU |
| **C** | retrain PPO against the chosen reward | needs coverage/iCS populated everywhere | GPU |

---

## 4. Findings so far

### Tier B — the failure mode is **absent from the data we have**
On the 125 kept-scratch designs (hotspot_coverage computed retroactively — no GPU):

- **0 impostors.** Every one of the 18 *designable* designs has coverage ≥ 0.67 (median 1.0);
  **zero** "good pAE (≤10) but low coverage (≤0.5)" cases.
- Coverage **co-varies with quality**: Pearson(pae, coverage) = −0.44, Pearson(iptm, cov) =
  +0.46. So the geometry-blind `current` reward is **not being mis-led here**.

**Read:** this data cannot yet *validate* the gate, because it lacks the failure case. That is
the decisive result — before any GPU retrain, we need data containing designable-but-low-coverage
impostors (the broad `all`-conditioning corpus, re-run with the PAE-persisting eval.py so
coverage/iCS land, §1.7). Adoption does not have to wait on this; validation does.

### Tier A — form safety at scale + developability
On all 4075 records (no geometry available, so a form/safety check):

- **`current+coverage` and `gated` never move the argmax lever cell** on any of the 10 targets
  (0 shifts) — safe to land; offline PPO would pick the same best cell. `gated` reshapes the
  *landscape* (cell-rank ρ 0.13–0.91) but agrees on the winner.
- **Developability:** 21.4% of designs fail the GRAVY>0 / |charge|>10 hard filter — and they
  score *higher* on the current reward (0.241 vs 0.216). The structural reward actively prefers
  undevelopable designs; a GRAVY/charge penalty is worth adding, free from `binder_seq`.
- **pI/instability bands are miscalibrated for this design class:** 95.0% of binders fall
  outside pI∈[6,9] and 89.0% exceed instability 40 — those thresholds are antibody-calibrated,
  so on short Genie3 mini-binders they are a near-constant offset, not a discriminator. They are
  **computed but not penalised by default**; recalibrate the bands to the binder distribution
  before enabling.

---

## 5. Recommendation

1. **Adopt `gated` as the production reward now.** It delivers both behaviours you wanted
   (harsh failure penalty + efficiency-scaled success), needs no unavailable term, and is
   argmax-stable on all 10 targets (safe to land). Wire `hotspot_coverage`/`ics` into the
   metrics path; run `coverage_missing="neutral"` on legacy records and switch to `"strict"`
   once coverage is populated everywhere.
2. **Add developability as `gated+dev` with GRAVY/charge only.** 21% of designs are
   undevelopable yet currently over-rewarded. Keep pI/instability computed-but-unweighted until
   their bands are recalibrated to this binder distribution.
3. **`tiered` is removed** — it was a foreign beam-search function needing an unpersisted term.
4. **The remaining blocker is data, not design.** The decisive *validity* test needs a corpus
   containing the failure mode. When GPU is available, re-run the broad `all`-conditioning cells
   with the PAE-persisting eval.py so `hotspot_coverage`/`iCS` populate, then re-run Tier B and,
   if the gate separates impostors, a Tier-C retrain.
