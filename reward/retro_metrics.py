"""Retroactively attach interface-geometry metrics to kept live-oracle scratch designs.

No GPU. Reads the ColabFold-predicted complex PDBs already on disk in the scratch dirs kept
via ``RLKNOBS_KEEP_LIVE_SCRATCH=1`` and computes, per child:

  * ``hotspot_coverage`` -- Cβ–Cβ contact (8 Å) of the binder against the target's TRUE full
    hotspot set (from the base problem JSON, never an ablated subset), reusing the exact same
    functions the live oracle uses so the retro value equals what a live run would log.
  * ``ics`` -- needs a per-child PAE sidecar (``child_N_pae.npy``); those were NOT persisted
    for the existing scratch runs (eval.py only started writing them 2026-07-07), so ``ics``
    stays None here until a PAE-persisting re-run (NEXT_STEPS.md §1.7). Supported for free if
    a sidecar is ever present.

This is the data backbone for the Tier B ranking-validity experiment: it turns the kept
scratch into a labelled table of (metrics + hotspot_coverage) with zero new oracle calls.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import config
from oracle.live_oracle import (
    _compute_hotspot_coverage,
    _compute_ics,
    _interface_contact_pairs,
)

log = logging.getLogger(__name__)

DEFAULT_SCRATCH = Path(
    os.environ.get("RLKNOBS_LIVE_SCRATCH", "/pscratch/sd/s/shreyash/rlknobs_live")
)


def base_target(variant: str) -> str:
    """Base target id from a (possibly variant) problem name.

    Variant dirs are named ``01_bhrf1``, ``01_bhrf1_ablate_others``, ``06_insulinr_longbinder``,
    etc. Targets are always ``NN_name``; everything after the second token is a lever suffix
    (``_longbinder`` / ``_ablate_*`` / ``_only_*``). hotspot_coverage is measured against the
    base target's full hotspot set regardless of the conditioning variant."""
    parts = variant.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else variant


@lru_cache(maxsize=None)
def hotspot_residues(target: str) -> Optional[tuple[str, ...]]:
    """The target's full hotspot residue set (e.g. ('B59','B83','B91')) from the base problem
    JSON. Cached. Returns None if unreadable."""
    problem_json = (
        config.GENIE3_ROOT / "data" / "design" / "binder_design" / "binderbench"
        / "problems" / f"{target}.json"
    )
    try:
        data = json.loads(problem_json.read_text())
        return tuple(data["target_interface_residues"]["hotspot"])
    except Exception as e:  # noqa: BLE001
        log.warning("could not load hotspot residues for %s: %s", target, e)
        return None


def _interface_metrics(pdb: Path, hotspots: list[str],
                       binder_seq: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """(hotspot_coverage, ics) for one predicted-complex PDB. ics only if a PAE sidecar
    ``<pdb stem without _predicted>_pae.npy`` exists next to it."""
    try:
        cov = _compute_hotspot_coverage(pdb, list(hotspots))
    except Exception as e:  # noqa: BLE001
        log.warning("hotspot_coverage failed for %s: %s", pdb, e)
        cov = None

    ics = None
    pae_path = pdb.parent / (pdb.name.replace("_predicted.pdb", "_pae.npy"))
    if pae_path.exists() and binder_seq:
        try:
            import numpy as np
            pae = np.load(pae_path)
            contacts = _interface_contact_pairs(pdb)
            ics = _compute_ics(pae, contacts, binder_len=len(binder_seq))
        except Exception as e:  # noqa: BLE001
            log.warning("ics failed for %s: %s", pae_path, e)
    return cov, ics


def collect_rows(scratch_dir: Path = DEFAULT_SCRATCH) -> list[dict[str, Any]]:
    """Every scratch child that has both a metrics JSON and a predicted-complex PDB, with
    hotspot_coverage / ics attached. Rows carry the raw metrics plus:
    ``target, variant, branch_t, run_id, hotspot_coverage, ics``."""
    rows: list[dict[str, Any]] = []
    n_seen = n_no_pdb = n_no_hotspots = 0
    for mpath in sorted(scratch_dir.glob("run_*/branch_t_*/*/child_*_metrics.json")):
        n_seen += 1
        try:
            metrics = json.loads(mpath.read_text())
        except Exception:  # noqa: BLE001
            continue
        if metrics.get("iptm") is None or "error" in metrics:
            continue
        child_id = metrics.get("child_id")
        pdb = mpath.parent / f"child_{child_id}_predicted.pdb"
        if not pdb.exists():
            n_no_pdb += 1
            continue
        variant = mpath.parent.name
        target = base_target(variant)
        hs = hotspot_residues(target)
        if not hs:
            n_no_hotspots += 1
            continue
        branch_t = int(mpath.parent.parent.name.replace("branch_t_", ""))
        run_id = mpath.parent.parent.parent.name
        cov, ics = _interface_metrics(pdb, list(hs), metrics.get("binder_seq"))

        row = dict(metrics)
        row.update({
            "target": target,
            "variant": variant,
            "branch_t": branch_t,
            "run_id": run_id,
            "hotspot_coverage": cov,
            "ics": ics,
        })
        rows.append(row)

    log.info(
        "collect_rows: %d rows from %d child metrics (%d missing predicted PDB, "
        "%d missing hotspots)", len(rows), n_seen, n_no_pdb, n_no_hotspots,
    )
    return rows
