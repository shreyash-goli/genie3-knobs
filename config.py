"""Central configuration: filesystem paths and the discrete action-space definition.

Everything that is environment-specific (where genie3 lives, where the /pscratch sweep
outputs are) is funnelled through here so the rest of the codebase has no hard-coded
absolute paths. Override any of these with the matching ``RLKNOBS_*`` environment variable.

This is a research scaffold (AlQuraishi Lab) -- prefer legibility over cleverness.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

#: Local genie3 checkout. Vendored under external/genie3 as a symlink to the working
#: copy (a proper git submodule could not be added from this NERSC node -- no GitHub
#: network access; see external/genie3_remote_url.txt + setup_genie3_submodule.sh).
GENIE3_ROOT = Path(os.environ.get("RLKNOBS_GENIE3_ROOT", REPO_ROOT / "external" / "genie3"))

#: Problem definitions (hotspots, binder length bounds, target metadata) used by the
#: hotspot-ablation sweeps. One JSON per problem variant.
DATASET_DIR = Path(
    os.environ.get(
        "RLKNOBS_DATASET_DIR",
        GENIE3_ROOT / "branching" / "hotspot_ablation" / "dataset",
    )
)

#: Root holding all the already-computed sweep outputs (child_*_metrics.json, *.pdb).
#: This is the raw material the Stage-0 ingester turns into an offline labelled dataset.
PSCRATCH_GENIE3 = Path(
    os.environ.get("RLKNOBS_PSCRATCH_GENIE3", "/pscratch/sd/s/shreyash/genie3")
)

#: Where the ingested offline dataset + experiment logs are written (git-ignored).
DATA_DIR = Path(os.environ.get("RLKNOBS_DATA_DIR", REPO_ROOT / "data"))
DATASET_JSONL = DATA_DIR / "records.jsonl"
DATASET_SQLITE = DATA_DIR / "records.sqlite"
EXPERIMENTS_LOG_DIR = DATA_DIR / "experiment_logs"

# --------------------------------------------------------------------------------------
# Sweep roots to ingest.  Each entry maps a /pscratch sub-directory to how its directory
# names encode the three levers.  Layout (observed):
#
#   trajectory_branching*/branch_t_<T>/<problem>/child_<i>_metrics.json
#   hotspot_ablation/branch_t_<T>/<problem_variant>/child_<i>_metrics.json
#   beam_search*/branch_{t,s}_<N>/<problem>/child_<i>_metrics.json
#
# ``branch_t_<T>`` is the diffusion timestep the trajectory was branched/resumed from
# (genie3 uses a 0..T diffusion clock; higher T == noisier / earlier commit point).
# The hotspot mode and binder-length lever are encoded in the *problem variant* suffix
# (e.g. ``01_bhrf1``, ``01_bhrf1_longbinder``, ``01_bhrf1_ablate_others``,
# ``01_bhrf1_only_B92``) -- see instrumentation/trajectory_logger.py for the parser.
# --------------------------------------------------------------------------------------
SWEEP_ROOTS = [
    "trajectory_branching_v2",
    "trajectory_branching",
    "hotspot_ablation",
]

# --------------------------------------------------------------------------------------
# Action space (Section 2.1 of the project spec).
#
# The spec's nominal branch-timestep set {10,15,20,30} is a *step-index* convention; the
# bulk of the logged data (trajectory_branching_v2: 6 timesteps x 10 targets x 40 kids)
# uses the diffusion-clock convention below.  The action space is therefore data-driven:
# the env restricts itself to the lever values that actually have logged children for the
# selected target(s).  These tuples are the *canonical* candidate values; the env
# intersects them with what is present.
# --------------------------------------------------------------------------------------
BRANCH_TIMESTEPS = (700, 750, 800, 850, 900, 950)  # diffusion-clock branch points

#: Binder-length lever.  Realised in data as the base problem vs its ``_longbinder``
#: variant (+~60 residues); +30 is interpolated by the offline simulator when absent.
LENGTH_DELTAS = (0, 30, 60)

#: Hotspot conditioning mode (Section 2.1).  Realised in data via problem-variant suffix.
HOTSPOT_MODES = ("all", "missed_only", "ablate_competitors")

#: Canonical hotspot-mode -> problem-variant-suffix mapping used by the ingester.
HOTSPOT_MODE_SUFFIX = {
    "all": "",                       # base problem (condition on all hotspots)
    "missed_only": "only_",          # e.g. 01_bhrf1_only_B92 (condition on a missed one)
    "ablate_competitors": "ablate_", # e.g. 01_bhrf1_ablate_others
}

# --------------------------------------------------------------------------------------
# Targets used by the Stage-3 contrast experiment (Section 2.2, Stage 3).
# BHRF1: longer binders demonstrably helped recover missed hotspots.
# InsulinR: length did NOT generalise (B59/B91 resisted all interventions).
# --------------------------------------------------------------------------------------
STAGE3_LENGTH_HELPED = "01_bhrf1"
STAGE3_LENGTH_DID_NOT_HELP = "06_insulinr"


def ensure_data_dirs() -> None:
    """Create the (git-ignored) output directories if they do not yet exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
