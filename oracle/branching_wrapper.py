"""Wrapper around Genie3's TrajectoryBrancher that captures xl_frozen (x_T).

Called as a subprocess by LiveRewardModel._run_branching() instead of calling
trajectory_branching.py directly. Runs entirely inside the genie3 conda env,
so in-process genie3 imports are safe here.

This script owns all x_T capture logic — genie3's own scripts are untouched.

Usage (internal — called by live_oracle.py):
    python -m oracle.branching_wrapper \\
        --config  <path/to/experiment_yaml> \\
        --timestep <int> \\
        --num-children <int> \\
        --output-dir <path> \\
        --selection <problem_name> \\
        [--seed <int>]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genie3 branching wrapper with x_T capture")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timestep", type=int, required=True)
    parser.add_argument("--num-children", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    genie3_root = Path(os.environ.get("RLKNOBS_GENIE3_ROOT", Path.home() / "genie3"))
    _orig_dir = os.getcwd()
    os.chdir(str(genie3_root))

    # --- imports (safe: this script runs inside the genie3 conda env) ---
    import torch
    from genie3.config import load_experiment_config, to_generation_config
    from genie3.generation.config.registry import build_sample_config_from_dict
    from genie3.generation.model.registry import get_model
    from collections import OrderedDict

    # These live in genie3's branching/scripts — add to path
    branching_scripts = genie3_root / "branching" / "scripts"
    if str(branching_scripts) not in sys.path:
        sys.path.insert(0, str(branching_scripts))

    from trajectory_branching import (
        TrajectoryBrancher,
        save_trajectory_outputs,
        load_model_and_sampler,
        prepare_batch,
    )

    # --- load model ---
    config_path = Path(args.config)
    run_config = load_experiment_config(str(config_path))
    generation_config = to_generation_config(run_config, shard_id=0, num_shards=1)
    sample_config = build_sample_config_from_dict(generation_config)
    checkpoint_path = str((genie3_root / sample_config.base.checkpoint).resolve())

    model = get_model(sample_config.model.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    updated_sd = OrderedDict()
    for k, v in state_dict.items():
        k2 = k.replace("_orig_mod.", "").replace(".linear_motif_template.", ".linear_cond_template.")
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        updated_sd[k2] = v
    model.load_state_dict(updated_sd, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    diffusion, sampler = load_model_and_sampler(generation_config, device)

    # --- prepare batch ---
    os.chdir(_orig_dir)
    batch = prepare_batch(
        config=sample_config,
        selection=args.selection,
        device=device,
    )
    os.chdir(str(genie3_root))

    # --- run branching, capturing xl_frozen ---
    brancher = TrajectoryBrancher(
        model=model,
        diffusion=diffusion,
        sampler=sampler,
        device=device,
        branch_timestep=args.timestep,
        num_children=args.num_children,
        base_seed=args.seed,
        is_baseline=False,
    )

    # Monkey-patch _denoise_to_branch_point to intercept xl_frozen without
    # modifying the genie3 source. The original method returns (xl_frozen, batch);
    # we wrap it to also stash xl_frozen on the brancher instance.
    _orig_denoise = brancher._denoise_to_branch_point

    def _patched_denoise(batch_arg):
        xl_frozen, batch_out = _orig_denoise(batch_arg)
        brancher._captured_xl_frozen = xl_frozen
        return xl_frozen, batch_out

    brancher._denoise_to_branch_point = _patched_denoise

    children_outputs = brancher.run_branching_experiment(batch)

    # --- save PDBs + metadata (using genie3's own I/O) ---
    output_dir = Path(args.output_dir)
    save_trajectory_outputs(
        children_outputs=children_outputs,
        batch=batch,
        output_dir=output_dir,
        branch_timestep=args.timestep,
        problem_name=args.selection,
    )

    # --- append x_T to metadata.json (our addition, not genie3's) ---
    branch_dir = output_dir / f"branch_t_{args.timestep}" / args.selection
    metadata_path = branch_dir / "metadata.json"
    xl_frozen = getattr(brancher, "_captured_xl_frozen", None)
    if xl_frozen is not None and metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            meta["x_T"] = xl_frozen.squeeze(0).cpu().numpy().tolist()
            metadata_path.write_text(json.dumps(meta, indent=2))
            log.info("x_T written to %s (shape %s)", metadata_path, list(xl_frozen.shape))
        except Exception as e:
            log.warning("Could not write x_T to metadata: %s", e)
    else:
        log.warning("xl_frozen not captured or metadata.json missing — x_T not saved")


if __name__ == "__main__":
    main()
