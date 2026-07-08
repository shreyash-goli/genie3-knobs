"""Stage 4 — Live oracle: Genie3 → ProteinMPNN → ColabFold, via shell-out.

This module implements LiveRewardModel by calling the existing genie3 branching
scripts as subprocesses.  The RL stack (SB3, this repo) and genie3 (torch==2.7.1)
stay in separate conda environments; we never import genie3 in-process.

Architecture:
    LiveRewardModel.sample(target, timestep, hotspot_mode, length_delta)
        1. Build a temp output dir under RLKNOBS_LIVE_SCRATCH
        2. Shell out: conda run -n genie3 python branching/scripts/trajectory_branching.py
           (generates N children PDB files at the requested lever cell)
        3. Shell out: conda run -n genie3 python branching/scripts/eval.py
           (runs ProteinMPNN + batched ColabFold, writes child_*_metrics.json)
        4. Parse child_*_metrics.json and return metrics + backoff=0

AsyncLiveRewardModel wraps the above with a concurrent.futures.ThreadPoolExecutor
for batched async rollouts (needed for Stage 6/7 PPO where many envs run in parallel).

Configuration via env vars (see config.py):
    RLKNOBS_GENIE3_ROOT     path to genie3 repo (default: ~/genie3)
    RLKNOBS_GENIE3_CONFIG   path to experiment YAML (default: branching/configs/experiment_trajectory_branching.yaml)
    RLKNOBS_LIVE_SCRATCH    scratch dir for live oracle outputs (default: /tmp/rlknobs_live)
    RLKNOBS_CONDA_ENV       conda env name for genie3 (default: genie3)
    RLKNOBS_NUM_CHILDREN    children per oracle call (default: 5)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Optional

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conda_run(
    cmd: list[str], env_name: str, cwd: Path, timeout: int = 900,
    extra_env: Optional[dict[str, str]] = None,
) -> str:
    """Run ``cmd``, preferring the current Python if already in the target env.

    On NERSC compute nodes ``conda`` is not on PATH inside a job, and wrapping
    with ``conda run`` would fail.  If the active Python interpreter lives inside
    the target conda env we replace ``python`` with sys.executable and drop the
    ``conda run`` wrapper entirely.  Falls back to ``conda run`` when running from
    a different env (e.g. a login node test).

    ``extra_env``, if given, is merged over a copy of the current environment (e.g. to
    scope ``CUDA_VISIBLE_DEVICES`` for a specific subprocess -- see
    ``LiveRewardModel``'s device pinning). When omitted, the subprocess inherits the
    parent environment unchanged, exactly as before.
    """
    import sys
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if active_env == env_name or env_name in sys.executable:
        # already in the right env — replace bare "python" with sys.executable
        resolved = [sys.executable if c == "python" else c for c in cmd]
        full_cmd = resolved
    else:
        full_cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd
    log.debug("live_oracle shell: %s", " ".join(full_cmd))
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    result = subprocess.run(
        full_cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Shell command failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(full_cmd)}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
    return result.stdout


class NoDatasetVariant(ValueError):
    """No genie3 problem JSON exists for this (target, hotspot_mode, length_delta)
    combination. Distinct from ValueError so callers can catch it specifically and skip
    the cell rather than shelling out to a dataloader that will come back empty."""


# Per-target hotspot-variant problem-file suffixes. These are NOT uniform across
# targets: each ablation target was chosen individually by find_missed_hotspots.py
# (see genie3's branching/hotspot_ablation/scripts/make_ablation_problems.py) --
# BHRF1's missed hotspot is B92, InsulinR's is B83, hence "_only_B92" /
# "_ablate_b83" rather than a generic suffix. Only entries present here have an
# actual problem JSON in branching/hotspot_ablation/dataset/problems/; anything else
# raises NoDatasetVariant. InsulinR has no missed_only variant at all yet -- its
# "never attempted" hotspots (B59, B91) were identified but never turned into a
# problem file (see NEXT_STEPS.md).
_HOTSPOT_SUFFIX_BY_TARGET: dict[tuple[str, str], str] = {
    ("01_bhrf1", "ablate_competitors"): "_ablate_others",
    ("01_bhrf1", "missed_only"): "_only_B92",
    ("06_insulinr", "ablate_competitors"): "_ablate_b83",
}


def _needs_ablation_config(hotspot_mode: str, length_delta: int) -> bool:
    """True if this cell's problem JSON lives in the hotspot_ablation dataset tree
    (needs experiment_hotspot_ablation.yaml) rather than the base binderbench tree
    (needs experiment_trajectory_branching.yaml). Every variant file -- longbinder,
    ablate_*, only_* -- lives under branching/hotspot_ablation/dataset/problems/; only
    the bare (all, 0) selection lives in the base dataset."""
    return not (hotspot_mode == "all" and length_delta == 0)


def _build_selection(target: str, hotspot_mode: str, length_delta: int) -> str:
    """Build the Genie3 dataset selection string for a lever cell.

    Raises NoDatasetVariant if no problem JSON exists for this exact combination.
    Only single-axis variants currently exist -- there is no problem file combining a
    non-"all" hotspot_mode with length_delta=60 for any target (e.g. no
    "01_bhrf1_ablate_others_longbinder"), so any such combination is unfillable until
    someone generates it (see NEXT_STEPS.md).
    """
    if hotspot_mode == "all":
        if length_delta == 0:
            return target
        if length_delta == 60:
            return f"{target}_longbinder"
        raise NoDatasetVariant(
            f"No longbinder-style variant exists for {target} length_delta={length_delta}"
        )

    if length_delta != 0:
        raise NoDatasetVariant(
            f"No combined hotspot+length problem file exists for {target} "
            f"hotspot_mode={hotspot_mode!r} length_delta={length_delta}"
        )

    suffix = _HOTSPOT_SUFFIX_BY_TARGET.get((target, hotspot_mode))
    if suffix is None:
        raise NoDatasetVariant(
            f"No {hotspot_mode!r} problem-file variant exists for target {target!r}"
        )
    return target + suffix


def _parse_child_metrics(branch_dir: Path) -> list[dict[str, Any]]:
    """Read all child_*_metrics.json from a branch dir."""
    results = []
    for p in sorted(branch_dir.glob("child_*_metrics.json")):
        try:
            results.append(json.loads(p.read_text()))
        except Exception as e:
            log.warning("Failed to parse %s: %s", p, e)
    return results


def _gravy(seq: str) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle). Higher = more hydrophobic."""
    kd = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
        "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
        "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
        "Y": -1.3, "V": 4.2,
    }
    scores = [kd[aa] for aa in seq.upper() if aa in kd]
    return sum(scores) / len(scores) if scores else 0.0


def _net_charge_ph7(seq: str) -> float:
    """Approximate net charge at pH 7 (positive = basic, negative = acidic)."""
    seq = seq.upper()
    positive = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
    negative = seq.count("D") + seq.count("E")
    return positive - negative


# Developability thresholds for binder sequences.
# GRAVY > 0 means net hydrophobic — aggregation-prone; most good binders sit in [-1, 0].
# |net_charge| > 10 predicts off-target electrostatic stickiness.
_GRAVY_MAX = 0.0
_NET_CHARGE_ABS_MAX = 10.0


def _passes_developability(seq: str) -> tuple[bool, float, float]:
    """Return (passes, gravy, net_charge). Fails if GRAVY > 0 or |charge| > 10."""
    g = _gravy(seq)
    c = _net_charge_ph7(seq)
    return (g <= _GRAVY_MAX and abs(c) <= _NET_CHARGE_ABS_MAX), g, c


# ---------------------------------------------------------------------------
# Interface geometry: hotspot_coverage (geometric) and iCS (PAE-based)
#
# Both are computed from the ColabFold-predicted complex PDB. In that PDB the binder is
# chain A (written first in the ColabFold fasta) and the target is chain B; hotspot
# residues from the problem JSON (e.g. "B64") are chain B, residue 64 -- they map directly
# to the target chain's residue numbers. See NEXT_STEPS.md §1.7/§1.8.
# ---------------------------------------------------------------------------

# Interface contact cutoff: two residues are "in contact" if their Cβ atoms (Cα for
# glycine) are within this distance. 8 Å matches Proteina-Complexa's contact terms.
_CONTACT_CUTOFF_ANGSTROM = 8.0
# iCS temperature: p_ij = exp(-pae_ij / T). T ≈ 10 matches Promera's formulation.
_ICS_TEMPERATURE = 10.0


def _parse_pdb_cb_coords(pdb_path: Path) -> dict[tuple[str, int], "Any"]:
    """Parse a PDB into {(chain, resnum): Cβ coordinate}. Uses Cα for residues without a
    Cβ (glycine). Fixed-column parsing (not split()) so it is robust to missing spaces
    between fields in the ColabFold output."""
    import numpy as np
    cb: dict[tuple[str, int], Any] = {}
    ca: dict[tuple[str, int], Any] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom not in ("CB", "CA"):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            (cb if atom == "CB" else ca)[(chain, resnum)] = xyz
    for key, coord in ca.items():
        cb.setdefault(key, coord)  # glycine fallback
    return cb


def _compute_hotspot_coverage(
    pdb_path: Path,
    hotspot_residues: list[str],
    binder_chain: str = "A",
    cutoff: float = _CONTACT_CUTOFF_ANGSTROM,
) -> Optional[float]:
    """Fraction of hotspot residues within `cutoff` Å (Cβ–Cβ) of any binder residue.

    `hotspot_residues` are problem-JSON strings like "B64" (chain letter + residue number).
    Returns None if the binder chain is absent or no hotspot residue is present in the PDB
    (so a missing/garbage structure yields None, not a misleading 0.0).
    """
    import numpy as np
    coords = _parse_pdb_cb_coords(pdb_path)
    binder = np.array([c for (ch, _), c in coords.items() if ch == binder_chain])
    if binder.size == 0:
        return None
    covered = 0
    n_valid = 0
    for res in hotspot_residues:
        chain = res[0]
        resnum = int(res[1:])
        cb = coords.get((chain, resnum))
        if cb is None:
            continue
        n_valid += 1
        if np.linalg.norm(binder - cb, axis=1).min() <= cutoff:
            covered += 1
    if n_valid == 0:
        return None
    return covered / n_valid


def _interface_contact_pairs(
    pdb_path: Path,
    binder_chain: str = "A",
    target_chain: str = "B",
    cutoff: float = _CONTACT_CUTOFF_ANGSTROM,
) -> list[tuple[int, int]]:
    """All (binder_resnum, target_resnum) pairs within `cutoff` Å (Cβ–Cβ). These are the
    interface contacts iCS averages the PAE-derived contact probability over."""
    import numpy as np
    coords = _parse_pdb_cb_coords(pdb_path)
    binder = {rn: c for (ch, rn), c in coords.items() if ch == binder_chain}
    target = {rn: c for (ch, rn), c in coords.items() if ch == target_chain}
    pairs: list[tuple[int, int]] = []
    for bi, bc in binder.items():
        for ti, tc in target.items():
            if np.linalg.norm(bc - tc) <= cutoff:
                pairs.append((bi, ti))
    return pairs


def _compute_ics(
    pae_matrix: "Any",
    contact_pairs: list[tuple[int, int]],
    binder_len: int,
    temperature: float = _ICS_TEMPERATURE,
) -> Optional[float]:
    """Interface Contact Score (Promera / Jing et al.): mean PAE-derived contact
    probability over the interface contact pairs.

    `pae_matrix` is the ColabFold N×N PAE (N = binder_len + target_len), indexed by
    fasta position (binder residues 0..binder_len-1, then target residues). `contact_pairs`
    are (binder_resnum, target_resnum) with 1-based residue numbers from the PDB. Each
    pair's PAE is symmetrized (mean of pae[i][j] and pae[j][i]) then mapped to a probability
    p = exp(-pae / T); iCS is the mean p over all contacts. Returns None if there are no
    contacts (nothing to average) so it is distinguishable from a real low score.
    """
    import numpy as np
    if not contact_pairs:
        return None
    pae = np.asarray(pae_matrix, dtype=np.float32)
    n = pae.shape[0]
    probs = []
    for b_resnum, t_resnum in contact_pairs:
        i = b_resnum - 1                    # binder position in the PAE matrix
        j = binder_len + (t_resnum - 1)     # target position (offset past the binder)
        if not (0 <= i < n and 0 <= j < n):
            continue
        pae_ij = 0.5 * (pae[i, j] + pae[j, i])
        probs.append(float(np.exp(-pae_ij / temperature)))
    if not probs:
        return None
    return float(np.mean(probs))


def _aggregate_children(
    children: list[dict[str, Any]],
    mean_pairwise_rmsd: Optional[float] = None,
    x_T: Optional[list] = None,
    branch_dir: Optional[Path] = None,
    hotspot_residues: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Aggregate per-child metrics into a single metrics dict (best-child-by-iptm).

    mean_pairwise_rmsd: Cα RMSD across children from metadata.json — used as the
    diversity proxy (normalised to [0, 1] over a 10Å scale). Pass None if unavailable;
    diversity will be 0.0 rather than silently wrong.

    branch_dir + hotspot_residues: when both are provided, hotspot_coverage and iCS are
    computed for the best child from its predicted-complex PDB (`child_{id}_predicted.pdb`)
    and, for iCS, a `child_{id}_pae.npy` sidecar if eval.py persisted one. When either is
    None (e.g. offline records, or a run predating the PAE sidecar), those metrics stay None.
    """
    valid = [c for c in children if c.get("iptm") is not None and "error" not in c]
    if not valid:
        return {
            "complex_success": False,
            "iptm": 0.0,
            "avg_interface_pae": 30.0,
            "hotspot_coverage": 0.0,
            "diversity": 0.0,
            "x_T": None,
        }

    # Pre-ColabFold developability filter: discard sequences that are net-hydrophobic
    # (GRAVY > 0) or carry extreme charge (|net_charge| > 10). These fail basic
    # developability screens and would waste downstream wet-lab effort.
    # If all children fail, fall back to unfiltered pool so the env always returns
    # a signal (policy still penalised via low ipTM/pAE).
    n_before = len(valid)
    developable = [
        c for c in valid
        if c.get("binder_seq") is None
        or _passes_developability(c["binder_seq"])[0]
    ]
    if developable:
        valid = developable
        log.debug("Developability filter: %d/%d children passed", len(valid), n_before)
    else:
        log.warning(
            "All %d children failed developability filter (GRAVY/charge); "
            "using unfiltered pool — reward will be low",
            n_before,
        )

    best = max(valid, key=lambda c: c.get("iptm", 0.0))
    n_success = sum(1 for c in valid if c.get("complex_success"))
    # diversity: mean pairwise Cα RMSD across children, normalised to [0, 1] over 10Å
    rmsd = mean_pairwise_rmsd or 0.0
    diversity = min(1.0, rmsd / 10.0)
    # x_T: the frozen diffusion state (branch point noise), serialized as nested list
    # from trajectory_branching.py. Convert to numpy array for FrontierBuffer.
    x_T_np = None
    if x_T is not None:
        try:
            import numpy as np
            x_T_np = np.array(x_T, dtype=np.float32)
        except Exception:
            pass
    binder_seq = best.get("binder_seq")
    gravy, net_charge = (None, None)
    if binder_seq:
        _, gravy, net_charge = _passes_developability(binder_seq)

    # Geometric interface metrics for the best child, from its predicted-complex PDB.
    hotspot_coverage = None
    ics = None
    if branch_dir is not None and hotspot_residues:
        hotspot_coverage, ics = _best_child_interface_metrics(
            branch_dir, best.get("child_id"), hotspot_residues, binder_seq
        )

    return {
        "complex_success": best.get("complex_success", False),
        "iptm": best.get("iptm", 0.0),
        "avg_interface_pae": best.get("avg_interface_pae", 30.0),
        "min_interface_pae": best.get("min_interface_pae", 30.0),
        "hotspot_coverage": hotspot_coverage,
        "ics": ics,
        "diversity": diversity,
        "n_children": len(valid),
        "n_success": n_success,
        "binder_seq": binder_seq,
        "gravy": gravy,
        "net_charge": net_charge,
        "x_T": x_T_np,
    }


def _best_child_interface_metrics(
    branch_dir: Path,
    child_id: Optional[int],
    hotspot_residues: list[str],
    binder_seq: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    """(hotspot_coverage, ics) for one child, from its predicted PDB + optional PAE sidecar.

    Any failure (missing PDB, unparseable structure) degrades to (None, None) rather than
    raising — a metrics glitch must not crash an oracle call mid-episode.
    """
    if child_id is None:
        return None, None
    pdb_path = branch_dir / f"child_{child_id}_predicted.pdb"
    if not pdb_path.exists():
        log.warning("predicted PDB missing for interface metrics: %s", pdb_path)
        return None, None
    try:
        hotspot_coverage = _compute_hotspot_coverage(pdb_path, hotspot_residues)
    except Exception as e:
        log.warning("hotspot_coverage failed for %s: %s", pdb_path, e)
        hotspot_coverage = None

    ics = None
    pae_path = branch_dir / f"child_{child_id}_pae.npy"
    if pae_path.exists() and binder_seq:
        try:
            import numpy as np
            pae = np.load(pae_path)
            contacts = _interface_contact_pairs(pdb_path)
            ics = _compute_ics(pae, contacts, binder_len=len(binder_seq))
        except Exception as e:
            log.warning("iCS failed for %s: %s", pae_path, e)
            ics = None
    return hotspot_coverage, ics


# ---------------------------------------------------------------------------
# LiveRewardModel
# ---------------------------------------------------------------------------

class LiveRewardModel:
    """Live oracle backend: runs Genie3 → ProteinMPNN → ColabFold via shell-out.

    Drop-in replacement for OfflineRewardModel: same ``sample()`` signature, always
    returns backoff=0 (every cell is generated on-demand).

    Parameters
    ----------
    genie3_root : path to genie3 repo checkout (default: RLKNOBS_GENIE3_ROOT or ~/genie3)
    config_yaml : force one experiment YAML for every call, overriding the automatic
                  per-cell selection between the base and hotspot_ablation configs
                  (default: None -- auto-select, see ``_config_for``)
    scratch_dir : writable scratch for live outputs (default: RLKNOBS_LIVE_SCRATCH or /tmp/rlknobs_live)
    conda_env   : name of the conda env that has genie3 installed (default: RLKNOBS_CONDA_ENV or "genie3")
    num_children: how many children to generate per oracle call (default: RLKNOBS_NUM_CHILDREN or 5)
    device      : CUDA device id for ProteinMPNN + ColabFold (default: 0)
    timeout     : per-subprocess timeout in seconds (default: 900)
    """

    def __init__(
        self,
        genie3_root: Optional[Path] = None,
        config_yaml: Optional[Path] = None,
        scratch_dir: Optional[Path] = None,
        conda_env: Optional[str] = None,
        num_children: int = 0,
        device: int = 0,
        timeout: int = 900,
    ):
        self.genie3_root = Path(
            genie3_root
            or os.environ.get("RLKNOBS_GENIE3_ROOT")
            or Path.home() / "genie3"
        )
        # Explicit override (arg or env var) forces one config for every call. Otherwise
        # _config_for() picks per-cell between the base dataset (bare target, no
        # hotspot/length variant) and the hotspot_ablation dataset (every variant --
        # longbinder, ablate_*, only_* -- lives there, see _needs_ablation_config).
        env_override = os.environ.get("RLKNOBS_GENIE3_CONFIG")
        self._config_yaml_override = Path(config_yaml) if config_yaml else (
            Path(env_override) if env_override else None
        )
        self.scratch_dir = Path(
            scratch_dir
            or os.environ.get("RLKNOBS_LIVE_SCRATCH")
            or "/tmp/rlknobs_live"
        )
        self.conda_env = conda_env or os.environ.get("RLKNOBS_CONDA_ENV", "genie3")
        self.num_children = num_children or int(os.environ.get("RLKNOBS_NUM_CHILDREN", "5"))
        self.device = device
        self.timeout = timeout
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir_for(self, hotspot_mode: str, length_delta: int) -> Path:
        """Where eval.py should look for problem JSONs -- eval.py is genie3's own
        script (not one we wrap) and has the same base-vs-ablation dataset split as
        branching_wrapper.py, via its own --dataset-dir flag rather than a config YAML.
        Must stay consistent with _config_for's routing."""
        if _needs_ablation_config(hotspot_mode, length_delta):
            return self.genie3_root / "branching" / "hotspot_ablation" / "dataset"
        return self.genie3_root / "data" / "design" / "binder_design" / "binderbench"

    def _config_for(self, hotspot_mode: str, length_delta: int) -> Path:
        if self._config_yaml_override is not None:
            return self._config_yaml_override
        if _needs_ablation_config(hotspot_mode, length_delta):
            return (
                self.genie3_root / "branching" / "hotspot_ablation" / "configs"
                / "experiment_hotspot_ablation.yaml"
            )
        return (
            self.genie3_root / "branching" / "configs"
            / "experiment_trajectory_branching.yaml"
        )

    def _hotspot_residues(self, target: str) -> Optional[list[str]]:
        """The target's *full* hotspot residue set (e.g. ["B64", ...]) from the base
        problem JSON. Always the base set, never an ablated variant -- hotspot_coverage
        measures how many true hotspots the binder contacts, independent of which subset
        was used for conditioning. Returns None if the problem JSON can't be read."""
        problem_json = (
            self.genie3_root / "data" / "design" / "binder_design" / "binderbench"
            / "problems" / f"{target}.json"
        )
        try:
            data = json.loads(problem_json.read_text())
            return data["target_interface_residues"]["hotspot"]
        except Exception as e:
            log.warning("could not load hotspot residues for %s: %s", target, e)
            return None

    def sample(
        self,
        target: str,
        timestep: int,
        hotspot_mode: str = "all",
        length_delta: int = 0,
    ) -> tuple[dict[str, Any], int]:
        """Generate a binder for (target, timestep, hotspot_mode, length_delta) and score it.

        Returns
        -------
        (metrics_dict, backoff_level)
            metrics_dict has the same schema as offline dataset records.
            backoff_level is always 0 (generated on-demand).
        """
        run_id = uuid.uuid4().hex[:8]
        out_dir = self.scratch_dir / f"run_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            selection = _build_selection(target, hotspot_mode, length_delta)
            config_yaml = self._config_for(hotspot_mode, length_delta)
            log.info(
                "live_oracle: target=%s ts=%d hs=%s len=%d  selection=%s  config=%s  run=%s",
                target, timestep, hotspot_mode, length_delta, selection, config_yaml.name, run_id,
            )

            # Step 1: generate children via trajectory_branching.py
            self._run_branching(
                out_dir=out_dir,
                timestep=timestep,
                selection=selection,
                num_children=self.num_children,
                config_yaml=config_yaml,
            )

            # Step 2: evaluate with ProteinMPNN + ColabFold via eval.py
            branch_dir = out_dir / f"branch_t_{timestep}" / target
            if not branch_dir.exists():
                # try without suffix (plain target name)
                branch_dir = out_dir / f"branch_t_{timestep}" / selection
            self._run_eval(
                sweep_root=out_dir,
                problem=selection,
                timestep=timestep,
                dataset_dir=self._dataset_dir_for(hotspot_mode, length_delta),
            )

            # Step 3: parse and aggregate
            children = _parse_child_metrics(branch_dir)
            # read mean_pairwise_rmsd and x_T from metadata.json
            mean_rmsd = None
            x_T = None
            metadata_path = branch_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    mean_rmsd = meta.get("mean_pairwise_rmsd")
                    x_T = meta.get("x_T")  # nested list [N_atoms, 3] or None
                except Exception:
                    pass
            metrics = _aggregate_children(
                children, mean_pairwise_rmsd=mean_rmsd, x_T=x_T,
                branch_dir=branch_dir, hotspot_residues=self._hotspot_residues(target),
            )
            metrics["target"] = target
            metrics["branch_timestep"] = timestep
            metrics["hotspot_mode"] = hotspot_mode
            metrics["length_delta"] = length_delta
            metrics["run_id"] = run_id
            return metrics, 0

        except Exception as e:
            log.error("live_oracle failed for %s ts=%d: %s", target, timestep, e)
            # Return a zero-reward metrics dict so the env doesn't crash;
            # the caller can check "error" key to detect failure.
            return {
                "target": target,
                "branch_timestep": timestep,
                "hotspot_mode": hotspot_mode,
                "length_delta": length_delta,
                "complex_success": False,
                "iptm": 0.0,
                "avg_interface_pae": 30.0,
                "hotspot_coverage": 0.0,
                "diversity": 0.0,
                "error": str(e),
                "run_id": run_id,
            }, 0
        finally:
            # keep scratch if RLKNOBS_KEEP_LIVE_SCRATCH=1 for debugging
            if os.environ.get("RLKNOBS_KEEP_LIVE_SCRATCH") != "1":
                shutil.rmtree(out_dir, ignore_errors=True)

    def _run_branching(self, out_dir: Path, timestep: int, selection: str,
                       num_children: int, config_yaml: Path) -> None:
        # Use our own wrapper (oracle/branching_wrapper.py) rather than calling
        # genie3's trajectory_branching.py directly. The wrapper imports TrajectoryBrancher
        # in-process, monkey-patches _denoise_to_branch_point to capture xl_frozen, and
        # writes x_T into metadata.json — all without touching genie3's source.
        #
        # branching_wrapper.py has no --device flag -- it always does
        # torch.device("cuda") (i.e. whatever CUDA_VISIBLE_DEVICES exposes as index 0).
        # CUDA_VISIBLE_DEVICES scoping is therefore the *only* way to pin this step to a
        # specific physical GPU when running several LiveRewardModel instances
        # concurrently (see MultiGPULiveRewardModel below).
        import sys as _sys
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            _sys.executable, str(repo_root / "oracle" / "branching_wrapper.py"),
            "--config", str(config_yaml),
            "--timestep", str(timestep),
            "--num-children", str(num_children),
            "--output-dir", str(out_dir),
            "--selection", selection,
        ]
        _conda_run(
            cmd, self.conda_env, cwd=self.genie3_root, timeout=self.timeout,
            extra_env={"CUDA_VISIBLE_DEVICES": str(self.device)},
        )

    def _run_eval(self, sweep_root: Path, problem: str, timestep: int, dataset_dir: Path) -> None:
        # Scope CUDA_VISIBLE_DEVICES the same way as _run_branching, so a single
        # LiveRewardModel instance uses exactly one physical GPU end-to-end. Once scoped,
        # exactly one device is visible to this subprocess, so its own --device argument
        # must be "0" (the re-numbered index), not self.device -- passing self.device here
        # (e.g. "2") would ask eval.py for a device that doesn't exist from its restricted
        # point of view.
        #
        # eval.py is genie3's own script (not one this repo wraps) and looks up the
        # problem JSON itself via --dataset-dir (defaults to the base binderbench
        # dataset) -- must be passed explicitly for hotspot/length variants, which live
        # under branching/hotspot_ablation/dataset instead (see _dataset_dir_for).
        cmd = [
            "python", "branching/scripts/eval.py",
            "--sweep-root", str(sweep_root),
            "--problem", problem,
            "--timestep", str(timestep),
            "--device", "0",
            "--dataset-dir", str(dataset_dir),
        ]
        _conda_run(
            cmd, self.conda_env, cwd=self.genie3_root, timeout=self.timeout,
            extra_env={"CUDA_VISIBLE_DEVICES": str(self.device)},
        )

    # Alias so it matches OfflineRewardModel's introspection API
    def targets(self) -> list[str]:
        return list(config.STAGE3_TARGETS)

    def available_timesteps(self, target: str) -> list[int]:
        return list(config.BRANCH_TIMESTEPS)

    def available_modes(self, target: str) -> list[str]:
        return list(config.HOTSPOT_MODES)

    def available_length_deltas(self, target: str) -> list[int]:
        return list(config.LENGTH_DELTAS)


# ---------------------------------------------------------------------------
# Async wrapper for Stage 6/7 batched rollouts
# ---------------------------------------------------------------------------

class AsyncLiveRewardModel:
    """Wraps LiveRewardModel with a thread-pool for concurrent rollouts.

    Usage (Stage 6/7 PPO worker pool):

        oracle = AsyncLiveRewardModel(max_workers=4)
        fut = oracle.submit(target, timestep, hotspot_mode, length_delta)
        # ... do other work ...
        metrics, backoff = fut.result()  # blocks until done
    """

    def __init__(self, max_workers: int = 4, **kwargs):
        self._model = LiveRewardModel(**kwargs)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        target: str,
        timestep: int,
        hotspot_mode: str = "all",
        length_delta: int = 0,
    ) -> "Future[tuple[dict[str, Any], int]]":
        return self._pool.submit(self._model.sample, target, timestep, hotspot_mode, length_delta)

    def sample(self, target: str, timestep: int, hotspot_mode: str = "all",
               length_delta: int = 0) -> tuple[dict[str, Any], int]:
        """Synchronous call — same as LiveRewardModel.sample() but thread-safe."""
        return self._model.sample(target, timestep, hotspot_mode, length_delta)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


# ---------------------------------------------------------------------------
# Multi-GPU dispatch for embarrassingly-parallel batches (e.g. fill_lever_grid.py)
# ---------------------------------------------------------------------------

class MultiGPULiveRewardModel:
    """Distributes independent sample() calls across N distinct physical GPUs.

    Unlike ``AsyncLiveRewardModel`` (which pools concurrent calls onto a *single*
    device -- useful for overlapping I/O, but every call still contends for one GPU's
    compute), this creates one ``LiveRewardModel`` per device and gives each its own
    dedicated single-worker executor. That guarantees at most one call is ever running
    on a given device at a time (device isolation) while all N devices run concurrently
    -- true multi-GPU parallelism for a batch of fully independent oracle calls (no
    shared state between cells, e.g. a grid-fill sweep).

    This does *not* speed up any single call -- it only helps when you have more than
    one independent call to make. It requires the job to actually have ``len(devices)``
    GPUs allocated (e.g. ``--gpus=N`` in SLURM); device indices are relative to what's
    visible to this process (``CUDA_VISIBLE_DEVICES`` as set by SLURM), not absolute
    physical GPU IDs.

    Usage::

        multi = MultiGPULiveRewardModel(devices=[0, 1, 2, 3])
        futures = {
            multi.submit(i, target, timestep, mode, length_delta): (target, timestep, mode, length_delta)
            for i, (target, timestep, mode, length_delta) in enumerate(cells)
        }
        for fut in as_completed(futures):
            metrics, backoff = fut.result()
        multi.shutdown()
    """

    def __init__(self, devices: list[int], **kwargs):
        if not devices:
            raise ValueError("devices must be a non-empty list of GPU indices")
        self._n = len(devices)
        self._models = [LiveRewardModel(device=d, **kwargs) for d in devices]
        # One single-worker executor per device: submissions to executor[i] always run
        # strictly sequentially on device i, which is what makes device isolation safe
        # even though call durations vary (a shared pool with N workers would NOT
        # guarantee this -- a freed worker can pick up the next queued item regardless of
        # which device it targets, so two items for the same device could end up running
        # concurrently on two different worker threads).
        self._executors = [ThreadPoolExecutor(max_workers=1) for _ in devices]

    def submit(
        self,
        index: int,
        target: str,
        timestep: int,
        hotspot_mode: str = "all",
        length_delta: int = 0,
    ) -> "Future[tuple[dict[str, Any], int]]":
        """Submit one call, pinned to device ``index % num_devices``.

        Assign ``index`` round-robin over your batch (e.g. the enumerate() index) so
        calls spread evenly across all devices.
        """
        idx = index % self._n
        return self._executors[idx].submit(
            self._models[idx].sample, target, timestep, hotspot_mode, length_delta
        )

    def shutdown(self, wait: bool = True) -> None:
        for ex in self._executors:
            ex.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
