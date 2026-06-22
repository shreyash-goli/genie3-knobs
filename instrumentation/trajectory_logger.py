"""Stage 0 -- logging infrastructure.

Two responsibilities, one record schema:

1. ``TrajectoryLogger`` -- a lightweight append-only logger that *new* sweep / generation
   scripts can call once per generated structure.  This is the forward-looking hook the
   spec asks to wire into ``run_hotspot_longbinder_sweep.sh`` & friends.  It writes JSONL
   (always) and mirrors to SQLite (optional, for easy querying in Stage 3).

2. ``ingest_existing_sweeps`` -- walks the already-computed /pscratch sweep outputs and
   emits the *same* record schema, producing an offline labelled dataset at zero new
   generation cost.  This is the data source for the env, the bandit, and the baselines.

A "record" is one generated child structure with the three levers that produced it and
the resulting oracle metrics:

    target            problem family, e.g. "01_bhrf1"
    branch_timestep   diffusion-clock branch point (int)
    hotspot_mode      one of config.HOTSPOT_MODES
    length_delta      binder-length lever, residues added vs base (int)
    binder_length     realised binder length (len of designed sequence)
    iptm, ptm, avg_interface_pae, min_interface_pae, binder_plddt,
    binder_scrmsd, complex_scrmsd                  (oracle metrics; floats / None)
    genie3_success, complex_success                (bools)
    hotspot_coverage  fraction of this target's hotspots contacted (float / None)
    diversity         1 - max sequence identity to siblings in the same cell (float / None)
    binder_seq        designed sequence (kept for diversity / future surrogate use)
    backbone_pdb      path to the child backbone pdb (logged now; cheap, future surrogate)
    source_sweep, source_dir, child_id            provenance
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional

import config

# --------------------------------------------------------------------------------------
# Record schema
# --------------------------------------------------------------------------------------
RECORD_FIELDS = [
    "target", "branch_timestep", "hotspot_mode", "length_delta", "binder_length",
    "iptm", "ptm", "avg_interface_pae", "min_interface_pae", "binder_plddt",
    "binder_scrmsd", "complex_scrmsd", "genie3_success", "complex_success",
    "hotspot_coverage", "diversity", "binder_seq", "backbone_pdb",
    "source_sweep", "source_dir", "child_id",
]


@dataclass
class TrajectoryRecord:
    target: str
    branch_timestep: int
    hotspot_mode: str
    length_delta: int
    binder_length: Optional[int] = None
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    avg_interface_pae: Optional[float] = None
    min_interface_pae: Optional[float] = None
    binder_plddt: Optional[float] = None
    binder_scrmsd: Optional[float] = None
    complex_scrmsd: Optional[float] = None
    genie3_success: Optional[bool] = None
    complex_success: Optional[bool] = None
    hotspot_coverage: Optional[float] = None
    diversity: Optional[float] = None
    binder_seq: Optional[str] = None
    backbone_pdb: Optional[str] = None
    source_sweep: Optional[str] = None
    source_dir: Optional[str] = None
    child_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Forward-looking logger (the "hook" for live sweep scripts)
# --------------------------------------------------------------------------------------
class TrajectoryLogger:
    """Append-only structured logger.

    Usage inside a generation/sweep script::

        from instrumentation.trajectory_logger import TrajectoryLogger, TrajectoryRecord
        logger = TrajectoryLogger()                       # appends to data/records.jsonl
        logger.log(TrajectoryRecord(target="01_bhrf1", branch_timestep=800,
                                    hotspot_mode="all", length_delta=0, iptm=0.88, ...))
        logger.close()
    """

    def __init__(self, jsonl_path: Path | str | None = None,
                 sqlite_path: Path | str | None = None, mirror_sqlite: bool = True):
        config.ensure_data_dirs()
        self.jsonl_path = Path(jsonl_path) if jsonl_path else config.DATASET_JSONL
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.jsonl_path.open("a")
        self._conn: Optional[sqlite3.Connection] = None
        if mirror_sqlite:
            self.sqlite_path = Path(sqlite_path) if sqlite_path else config.DATASET_SQLITE
            self._conn = _open_sqlite(self.sqlite_path)

    def log(self, record: TrajectoryRecord) -> None:
        d = record.to_dict()
        self._fh.write(json.dumps(d) + "\n")
        self._fh.flush()
        if self._conn is not None:
            _insert_sqlite(self._conn, d)

    def close(self) -> None:
        self._fh.close()
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    cols = ",\n  ".join(f"{f} {_sqlite_type(f)}" for f in RECORD_FIELDS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS records (\n  {cols}\n)")
    conn.commit()
    return conn


def _sqlite_type(field_name: str) -> str:
    if field_name in {"branch_timestep", "length_delta", "binder_length", "child_id"}:
        return "INTEGER"
    if field_name in {"target", "hotspot_mode", "binder_seq", "backbone_pdb",
                      "source_sweep", "source_dir"}:
        return "TEXT"
    return "REAL"  # floats + bools (stored 0/1)


def _insert_sqlite(conn: sqlite3.Connection, d: dict[str, Any]) -> None:
    placeholders = ",".join("?" for _ in RECORD_FIELDS)
    vals = [d.get(f) for f in RECORD_FIELDS]
    conn.execute(f"INSERT INTO records VALUES ({placeholders})", vals)


# --------------------------------------------------------------------------------------
# Problem-variant parsing (maps directory names -> levers)
# --------------------------------------------------------------------------------------
def parse_problem_variant(variant: str) -> tuple[str, str, int]:
    """Map a problem-variant directory name to (base_problem, hotspot_mode, length_delta).

    Examples::

        01_bhrf1              -> ("01_bhrf1", "all", 0)
        01_bhrf1_longbinder   -> ("01_bhrf1", "all", 60)
        01_bhrf1_ablate_others-> ("01_bhrf1", "ablate_competitors", 0)
        01_bhrf1_only_B92     -> ("01_bhrf1", "missed_only", 0)
        06_insulinr_ablate_b83-> ("06_insulinr", "ablate_competitors", 0)
    """
    length_delta = 0
    hotspot_mode = "all"
    name = variant

    if name.endswith("_longbinder"):
        length_delta = 60  # the _longbinder variants add ~60 residues
        name = name[: -len("_longbinder")]

    # order matters: check the more specific markers first
    if "_ablate_" in name or name.endswith("_ablate_others"):
        hotspot_mode = "ablate_competitors"
        name = name.split("_ablate_")[0]
    elif "_only_" in name:
        hotspot_mode = "missed_only"
        name = name.split("_only_")[0]

    return name, hotspot_mode, length_delta


def _branch_timestep_from_dir(dirname: str) -> Optional[int]:
    """``branch_t_800`` / ``branch_s_10`` -> 800 / 10."""
    if not (dirname.startswith("branch_t_") or dirname.startswith("branch_s_")):
        return None
    try:
        return int(dirname.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------------------
# Offline ingestion of existing sweeps
# --------------------------------------------------------------------------------------
def _load_child_metrics(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _sequence_identity(a: str, b: str) -> float:
    """Length-normalised position-wise identity (no alignment -- cheap proxy)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def _diversity_for_cell(seqs: list[Optional[str]]) -> list[Optional[float]]:
    """diversity_i = 1 - max identity of child i to any sibling in the same cell."""
    out: list[Optional[float]] = []
    for i, s in enumerate(seqs):
        if not s:
            out.append(None)
            continue
        max_id = 0.0
        for j, t in enumerate(seqs):
            if i == j or not t:
                continue
            max_id = max(max_id, _sequence_identity(s, t))
        out.append(1.0 - max_id)
    return out


def _coverage_for_cell(child_paths: list[Path], hotspots) -> Optional[dict[str, float]]:
    """Per-hotspot contact fraction across the cell, via genie3's pdb_utils.

    Returns None if genie3's pdb_utils is unavailable or no backbones were found.
    Computing this reads PDBs, so it is opt-in (``compute_coverage=True``).
    """
    if not child_paths or not hotspots:
        return None
    try:
        sys.path.insert(0, str(config.GENIE3_ROOT / "branching"))
        from common.pdb_utils import compute_hotspot_contact_fraction  # type: ignore
    except Exception:
        return None
    try:
        return compute_hotspot_contact_fraction([str(p) for p in child_paths], hotspots)
    except Exception:
        return None


def _load_hotspots(base_problem: str, variant: str):
    """Load hotspot residues for a problem variant from the dataset dir, if present."""
    try:
        sys.path.insert(0, str(config.GENIE3_ROOT / "branching"))
        from common.pdb_utils import load_hotspots  # type: ignore
        return load_hotspots(config.DATASET_DIR, variant)
    except Exception:
        return []


def ingest_existing_sweeps(
    sweep_roots: Optional[Iterable[str]] = None,
    pscratch_root: Optional[Path] = None,
    compute_coverage: bool = False,
    output_jsonl: Optional[Path] = None,
    mirror_sqlite: bool = True,
    verbose: bool = True,
) -> list[TrajectoryRecord]:
    """Walk existing /pscratch sweep outputs and emit TrajectoryRecords.

    Returns the list of records and (side effect) writes them to ``output_jsonl`` and,
    if ``mirror_sqlite``, to the SQLite mirror.  Re-running overwrites the JSONL/SQLite
    so the dataset is deterministic w.r.t. what is on disk.
    """
    sweep_roots = list(sweep_roots) if sweep_roots is not None else config.SWEEP_ROOTS
    pscratch_root = Path(pscratch_root) if pscratch_root else config.PSCRATCH_GENIE3
    output_jsonl = Path(output_jsonl) if output_jsonl else config.DATASET_JSONL

    config.ensure_data_dirs()
    records: list[TrajectoryRecord] = []

    for sweep in sweep_roots:
        sweep_dir = pscratch_root / sweep
        if not sweep_dir.is_dir():
            if verbose:
                print(f"[ingest] skip (missing): {sweep_dir}")
            continue
        for branch_dir in sorted(sweep_dir.iterdir()):
            if not branch_dir.is_dir():
                continue
            ts = _branch_timestep_from_dir(branch_dir.name)
            if ts is None:
                continue
            for variant_dir in sorted(branch_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                metric_files = sorted(variant_dir.glob("child_*_metrics.json"))
                if not metric_files:
                    continue
                base_problem, hotspot_mode, length_delta = parse_problem_variant(
                    variant_dir.name
                )

                # gather the cell first (need siblings for diversity + coverage)
                cell: list[tuple[int, dict[str, Any], Path]] = []
                for mf in metric_files:
                    metrics = _load_child_metrics(mf)
                    if metrics is None:
                        continue
                    cid = metrics.get("child_id")
                    if cid is None:
                        try:
                            cid = int(mf.stem.split("_")[1])
                        except (ValueError, IndexError):
                            cid = -1
                    pdb = variant_dir / f"child_{cid}.pdb"
                    cell.append((cid, metrics, pdb))
                if not cell:
                    continue

                seqs = [m.get("binder_seq") for _, m, _ in cell]
                diversities = _diversity_for_cell(seqs)

                coverage_map = None
                cov_value_by_cid: dict[int, Optional[float]] = {}
                if compute_coverage:
                    hotspots = _load_hotspots(base_problem, variant_dir.name)
                    paths = [p for _, _, p in cell if p.exists()]
                    coverage_map = _coverage_for_cell(paths, hotspots)
                    # cell-level coverage = mean per-hotspot contact fraction
                    if coverage_map:
                        cell_cov = sum(coverage_map.values()) / len(coverage_map)
                        for cid, _, _ in cell:
                            cov_value_by_cid[cid] = cell_cov

                for (cid, metrics, pdb), div in zip(cell, diversities):
                    seq = metrics.get("binder_seq")
                    rec = TrajectoryRecord(
                        target=base_problem,
                        branch_timestep=ts,
                        hotspot_mode=hotspot_mode,
                        length_delta=length_delta,
                        binder_length=len(seq) if seq else None,
                        iptm=metrics.get("iptm"),
                        ptm=metrics.get("ptm"),
                        avg_interface_pae=metrics.get("avg_interface_pae"),
                        min_interface_pae=metrics.get("min_interface_pae"),
                        binder_plddt=metrics.get("binder_plddt"),
                        binder_scrmsd=metrics.get("binder_scrmsd"),
                        complex_scrmsd=metrics.get("complex_scrmsd"),
                        genie3_success=metrics.get("genie3_success"),
                        complex_success=metrics.get("complex_success"),
                        hotspot_coverage=cov_value_by_cid.get(cid),
                        diversity=div,
                        binder_seq=seq,
                        backbone_pdb=str(pdb) if pdb.exists() else None,
                        source_sweep=sweep,
                        source_dir=str(variant_dir),
                        child_id=cid,
                    )
                    records.append(rec)

    # write outputs (overwrite)
    with output_jsonl.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    if mirror_sqlite:
        conn = _open_sqlite(config.DATASET_SQLITE)
        conn.execute("DELETE FROM records")
        for rec in records:
            _insert_sqlite(conn, rec.to_dict())
        conn.commit()
        conn.close()

    if verbose:
        print(f"[ingest] wrote {len(records)} records -> {output_jsonl}")
        _print_ingest_summary(records)
    return records


def _print_ingest_summary(records: list[TrajectoryRecord]) -> None:
    by_target: dict[str, int] = {}
    by_ts: dict[int, int] = {}
    by_mode: dict[str, int] = {}
    n_success = 0
    for r in records:
        by_target[r.target] = by_target.get(r.target, 0) + 1
        by_ts[r.branch_timestep] = by_ts.get(r.branch_timestep, 0) + 1
        by_mode[r.hotspot_mode] = by_mode.get(r.hotspot_mode, 0) + 1
        if r.complex_success:
            n_success += 1
    print(f"[ingest]   targets: {dict(sorted(by_target.items()))}")
    print(f"[ingest]   branch_timesteps: {dict(sorted(by_ts.items()))}")
    print(f"[ingest]   hotspot_modes: {by_mode}")
    print(f"[ingest]   complex_success: {n_success}/{len(records)}")


def load_records(jsonl_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Read a previously-ingested dataset back into a list of dicts."""
    jsonl_path = Path(jsonl_path) if jsonl_path else config.DATASET_JSONL
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"No dataset at {jsonl_path}. Run: python -m instrumentation.trajectory_logger"
        )
    with jsonl_path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest existing genie3 sweeps -> offline dataset")
    p.add_argument("--coverage", action="store_true",
                   help="compute hotspot contact fraction from PDBs (slower, reads pdb files)")
    p.add_argument("--no-sqlite", action="store_true", help="skip the SQLite mirror")
    args = p.parse_args()
    ingest_existing_sweeps(compute_coverage=args.coverage, mirror_sqlite=not args.no_sqlite)
