from pathlib import Path
from typing import Any, TypeAlias
import numpy as np
import torch
from gaitnet import timestamp, PROJECT_ROOT
import atexit
from time import time
import matplotlib.pyplot as plt
from datetime import datetime

from gaitnet import get_logger

logger = get_logger()

# Ensure images directory exists
image_dir = PROJECT_ROOT / "data" / "debug-images"
image_dir.mkdir(exist_ok=True)

# Delete all old images
for file in image_dir.iterdir():
    if file.is_file() and file.suffix in [".png", ".jpg", ".jpeg"]:
        try:
            file.unlink()
        except Exception as e:
            logger.warning(f"failed to delete old image file {file}: {e}")
logger.info(f"cleared old images in {image_dir}")

i = 0  # global counter for images


def save_img(
    img: np.ndarray,
    name: str | None = None,
    cmap_limits: tuple[float, float] | None = None,
):
    """Save an image to the logs directory with a timestamp."""
    global i
    name = name if name is not None else "image"
    timestamp = (
        datetime.now()
        .isoformat(timespec="microseconds")
        .replace(".", "-")
        .replace(":", "-")
    )
    image_file = image_dir / f"{timestamp}_{i}_{name}.png"
    plt.imsave(image_file, img, vmin=cmap_limits[0] if cmap_limits else None, vmax=cmap_limits[1] if cmap_limits else None)
    logger.info(f"saved image to {image_file}")
    i += 1


def save_fig(
    img: np.ndarray,
    name: str | None = None,
    cmap_limits: tuple[float, float] | None = None,
    gridlines: bool = False,
) -> None:
    """Save an image to the logs directory with a timestamp."""
    global i
    name = name if name is not None else "image"
    timestamp = (
        datetime.now()
        .isoformat(timespec="microseconds")
        .replace(".", "-")
        .replace(":", "-")
    )
    image_file = image_dir / f"{timestamp}_{i}_{name}.png"
    # plt.imsave(image_file, img)
    # make a plot with the image and a colorbar
    plt.figure()
    plt.imshow(
        img,
        vmin=cmap_limits[0] if cmap_limits else None,
        vmax=cmap_limits[1] if cmap_limits else None,
    )
    if gridlines:
        # show gridlines between pixels, no ticks
        # Major ticks
        x_size = img.shape[1]
        y_size = img.shape[0]
        plt.xticks(np.arange(0, x_size, 1))
        plt.yticks(np.arange(0, y_size, 1))

        # Minor ticks
        plt.xticks(np.arange(-.5, x_size, 1), minor=True)
        plt.yticks(np.arange(-.5, y_size, 1), minor=True)

        # Gridlines based on minor ticks
        plt.grid(which='minor', color='b', linestyle='-', linewidth=1)

        # Remove minor ticks
        plt.tick_params(which='minor', bottom=False, left=False)
    else:
        plt.axis('off')

    plt.colorbar()
    plt.title(name)
    # remove axes
    plt.savefig(image_file)
    plt.close()
    logger.info(f"saved image to {image_file}")
    i += 1
