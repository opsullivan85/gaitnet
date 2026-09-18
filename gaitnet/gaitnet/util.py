from pathlib import Path
from gaitnet import PROJECT_ROOT
from gaitnet import get_logger

logger = get_logger()


def add_checkpoint_arg(parser) -> None:
    """Add the checkpoint selection flag to an entry point's argument parser."""
    parser.add_argument(
        "--checkpoint-name-gaitnet",
        "--checkpoint_path",
        dest="checkpoint_name",
        type=str,
        default=None,
        help="Checkpoint file or run folder within training/gaitnet/runs. Defaults to the most recent run.",
    )


def get_checkpoint_path(checkpoint_name: str | None = None) -> Path:
    """Get the path to a model checkpoint.

    Args:
        checkpoint_name: A checkpoint file or run folder within training/gaitnet/runs.
            A folder resolves to its newest checkpoint. None uses the most recent run.

    Returns:
        Path: Path to the model checkpoint.
    """
    checkpoint_dir = PROJECT_ROOT / "training" / "gaitnet" / "runs"
    if checkpoint_name is None:
        # no checkpoint name provided, use most recent checkpoint folder sorted by name
        # gaitnet_YYYYMMDD_HHMMSS
        model_paths = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not model_paths:
            raise FileNotFoundError("No checkpoints found in training/gaitnet/runs.")
        checkpoint_name = model_paths[0].name

    checkpoint_path: Path = checkpoint_dir / checkpoint_name
    if checkpoint_path.is_file():
        logger.info(f"using checkpoint file: {checkpoint_path}")
        return checkpoint_path

    elif checkpoint_path.is_dir():
        logger.info(f"searching checkpoint directory: {checkpoint_path}")
        model_paths = sorted(
            [d for d in checkpoint_path.iterdir() if d.is_file() and d.name.endswith((".pt",))],
            key=lambda d: d.stat().st_mtime,
        )
        if not model_paths:
            raise FileNotFoundError("No checkpoints found in specified directory.")
        
        latest_model_path = model_paths[-1]
        logger.info(f"using checkpoint file: {latest_model_path}")
        return latest_model_path
    
    else:
        raise FileNotFoundError(f"checkpoint file/directory \"{checkpoint_path}\" does not exist.")