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

def _conda_run(cmd: list[str], env_name: str, cwd: Path, timeout: int = 900) -> str:
    """Run ``cmd``, preferring the current Python if already in the target env.

    On NERSC compute nodes ``conda`` is not on PATH inside a job, and wrapping
    with ``conda run`` would fail.  If the active Python interpreter lives inside
    the target conda env we replace ``python`` with sys.executable and drop the
    ``conda run`` wrapper entirely.  Falls back to ``conda run`` when running from
    a different env (e.g. a login node test).
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
    result = subprocess.run(
        full_cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Shell command failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(full_cmd)}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
    return result.stdout


def _length_delta_to_variant_suffix(length_delta: int) -> str:
    """Map length_delta integer to the dataset selection suffix Genie3 expects."""
    if length_delta == 0:
        return ""
    return "_longbinder"  # +30 and +60 both use this variant; the actual extra length
                          # is controlled by the problem JSON, not the suffix


def _hotspot_to_variant_suffix(hotspot_mode: str, target: str) -> str:
    """Map hotspot_mode to dataset selection suffix (target-specific for missed_only)."""
    if hotspot_mode == "all":
        return ""
    if hotspot_mode == "ablate_competitors":
        return "_ablate_others"
    if hotspot_mode == "missed_only":
        # missed_only uses _only_<hotspot> naming in Genie3 sweep dirs;
        # we use the plain _missed suffix in the live oracle selection
        return "_missed"
    raise ValueError(f"Unknown hotspot_mode: {hotspot_mode!r}")


def _build_selection(target: str, hotspot_mode: str, length_delta: int) -> str:
    """Build the Genie3 dataset selection string for a lever cell."""
    base = target
    hs_suffix = _hotspot_to_variant_suffix(hotspot_mode, target)
    len_suffix = _length_delta_to_variant_suffix(length_delta)
    return base + hs_suffix + len_suffix


def _parse_child_metrics(branch_dir: Path) -> list[dict[str, Any]]:
    """Read all child_*_metrics.json from a branch dir."""
    results = []
    for p in sorted(branch_dir.glob("child_*_metrics.json")):
        try:
            results.append(json.loads(p.read_text()))
        except Exception as e:
            log.warning("Failed to parse %s: %s", p, e)
    return results


def _aggregate_children(
    children: list[dict[str, Any]],
    mean_pairwise_rmsd: Optional[float] = None,
) -> dict[str, Any]:
    """Aggregate per-child metrics into a single metrics dict (best-child-by-iptm).

    mean_pairwise_rmsd: Cα RMSD across children from metadata.json — used as the
    diversity proxy (normalised to [0, 1] over a 10Å scale). Pass None if unavailable;
    diversity will be 0.0 rather than silently wrong.
    """
    valid = [c for c in children if c.get("iptm") is not None and "error" not in c]
    if not valid:
        return {
            "complex_success": False,
            "iptm": 0.0,
            "avg_interface_pae": 30.0,
            "hotspot_coverage": 0.0,
            "diversity": 0.0,
        }
    best = max(valid, key=lambda c: c.get("iptm", 0.0))
    n_success = sum(1 for c in valid if c.get("complex_success"))
    # diversity: mean pairwise Cα RMSD across children, normalised to [0, 1] over 10Å
    rmsd = mean_pairwise_rmsd or 0.0
    diversity = min(1.0, rmsd / 10.0)
    return {
        "complex_success": best.get("complex_success", False),
        "iptm": best.get("iptm", 0.0),
        "avg_interface_pae": best.get("avg_interface_pae", 30.0),
        "min_interface_pae": best.get("min_interface_pae", 30.0),
        "hotspot_coverage": None,  # not computed by eval.py; requires contact analysis
        "diversity": diversity,
        "n_children": len(valid),
        "n_success": n_success,
        "binder_seq": best.get("binder_seq"),
    }


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
    config_yaml : path to experiment YAML (default: genie3_root/branching/configs/experiment_trajectory_branching.yaml)
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
        self.config_yaml = Path(
            config_yaml
            or os.environ.get("RLKNOBS_GENIE3_CONFIG")
            or self.genie3_root / "branching" / "configs" / "experiment_trajectory_branching.yaml"
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
            log.info(
                "live_oracle: target=%s ts=%d hs=%s len=%d  selection=%s  run=%s",
                target, timestep, hotspot_mode, length_delta, selection, run_id,
            )

            # Step 1: generate children via trajectory_branching.py
            self._run_branching(
                out_dir=out_dir,
                timestep=timestep,
                selection=selection,
                num_children=self.num_children,
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
            )

            # Step 3: parse and aggregate
            children = _parse_child_metrics(branch_dir)
            # read mean_pairwise_rmsd from metadata.json (written by trajectory_branching.py)
            mean_rmsd = None
            metadata_path = branch_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    mean_rmsd = meta.get("mean_pairwise_rmsd")
                except Exception:
                    pass
            metrics = _aggregate_children(children, mean_pairwise_rmsd=mean_rmsd)
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
                       num_children: int) -> None:
        cmd = [
            "python", "branching/scripts/trajectory_branching.py",
            "--config", str(self.config_yaml),
            "--timestep", str(timestep),
            "--num-children", str(num_children),
            "--output-dir", str(out_dir),
            "--selection", selection,
        ]
        _conda_run(cmd, self.conda_env, cwd=self.genie3_root, timeout=self.timeout)

    def _run_eval(self, sweep_root: Path, problem: str, timestep: int) -> None:
        cmd = [
            "python", "branching/scripts/eval.py",
            "--sweep-root", str(sweep_root),
            "--problem", problem,
            "--timestep", str(timestep),
            "--device", str(self.device),
        ]
        _conda_run(cmd, self.conda_env, cwd=self.genie3_root, timeout=self.timeout)

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
