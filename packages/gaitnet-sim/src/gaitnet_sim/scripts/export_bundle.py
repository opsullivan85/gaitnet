"""Export a trained run as a policy bundle, the file evaluation and deployment load.

    python -m gaitnet_sim.scripts.export_bundle --run logs/rsl_rl/gaitnet_holes/<timestamp> --out bundle.pt
    python -m gaitnet_sim.scripts.export_bundle --mlflow_run <run id> --out bundle.pt

`--checkpoint model_<i>.pt` picks a checkpoint; the latest by default. From MLflow the
bundle is also logged back to the run, under `bundles/`. Needs no simulator.
"""

from __future__ import annotations

import argparse
import sys

from gaitnet_core.bundle import load_bundle, save_bundle
from gaitnet_sim.rl.export import bundle_from_mlflow, bundle_from_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Local run directory.")
    source.add_argument("--mlflow_run", help="MLflow run id (tracking server from MLFLOW_TRACKING_URI).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file name, e.g. model_500.pt.")
    parser.add_argument("--out", required=True, help="Bundle file to write.")
    args = parser.parse_args(argv)

    if args.run:
        bundle = bundle_from_run(args.run, args.checkpoint)
    else:
        bundle = bundle_from_mlflow(args.mlflow_run, args.checkpoint)
    path = save_bundle(args.out, bundle)
    # read it back through the same checks deployment runs
    load_bundle(path)
    print(f"wrote {path}: {type(bundle.actor).__name__}, {bundle.extra.get('checkpoint')}, features {list(bundle.features)}")

    if args.mlflow_run:
        from mlflow import MlflowClient

        MlflowClient().log_artifact(args.mlflow_run, str(path), artifact_path="bundles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
