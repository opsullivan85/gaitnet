from pathlib import Path
from gaitnet import PROJECT_ROOT
import argparse

parser = argparse.ArgumentParser(description="ContactNet Utilities")
parser.add_argument(
    "--datasets",
    type=str,
    nargs="*",
    default=None,
    help="Paths to dataset directories to use for evaluation",
)
parser.add_argument(
    "--checkpoint-name",
    type=str,
    default=None,
    help="Path to the model checkpoint to evaluate. Should be the name of a folder within training/contactnet/checkpoints. best_model.pt is automatically used from there.",
)
args, unused_args = parser.parse_known_args()


def get_dataset_paths() -> list[Path]:
    """Get the list of dataset paths from the datasets directory.

    If "--datasets" argument is provided, use those paths. Otherwise, return all dataset directories.

    Returns:
        list[Path]: List of dataset directory paths.
    """
    if args.datasets:
        return [Path(d) for d in args.datasets]
    else:
        dataset_dir = PROJECT_ROOT / "data" / "datasets"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
        dataset_paths = [d for d in dataset_dir.iterdir() if not d.is_dir() and d.suffix == ".pkl"]
        if not dataset_paths:
            raise FileNotFoundError(
                f"No datasets found in {dataset_dir}. Please generate data first."
            )
        return dataset_paths


def get_checkpoint_path() -> Path:
    """Get the path to the most recent model checkpoint.

    If "--checkpoint-name" argument is provided, use that path. Otherwise, return the most recent checkpoint.

    Returns:
        Path: Path to the model checkpoint.
    """
    checkpoint_dir = PROJECT_ROOT / "training" / "contactnet" / "checkpoints"
    if args.checkpoint_name:
        checkpoint_path = checkpoint_dir / args.checkpoint_name / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
        return checkpoint_path
    else:
        checkpoint_paths = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not checkpoint_paths:
            raise FileNotFoundError("No model directories found in checkpoints.")
        most_recent_checkpoint = checkpoint_paths[0] / "best_model.pt"
        if not most_recent_checkpoint.exists():
            raise FileNotFoundError(
                f"Most recent checkpoint {most_recent_checkpoint} does not exist."
            )
        return most_recent_checkpoint
