import argparse
import pickle

import numpy as np
from gaitnet import logger
from gaitnet.contactnet.debug import view_footstep_cost_map
from gaitnet.contactnet.tree import StepNode
import gaitnet.simulation.cfg.footstep_scanner_constants as fs
from gaitnet.contactnet.util import get_dataset_paths
import matplotlib.pyplot as plt
from matplotlib import colors


parser = argparse.ArgumentParser(description="Data information")
parser.add_argument(
    "--costmaps",
    action="store_true",
    help="Generate costmaps for the dataset"
)
parser.add_argument(
    "--placement-maps",
    action="store_true",
    help="Generate footstep placement maps for the dataset"
)
args, unused_args = parser.parse_known_args()

def log_data_info():
    """Logs information about the data files in the data directory."""
    total_points = 0
    data_files = get_dataset_paths()
    for data_file in data_files:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        msg = f"Found data file: {data_file}\n"
        msg += f"\t- training_data: {len(data['training_data'])} points\n"
        total_points += len(data['training_data'])
        metadata_msg = "\n\t\t- " + "\n\t\t- ".join(f"{k}: {v}" for k, v in data['metadata'].items())
        msg += f"\t- metadata: {metadata_msg}\n"
        logger.info(msg)
        
    logger.info(f"Total training data points across {len(data_files)} files: {total_points}")

def generate_all_costmaps():
    """Generates costmaps for all data files in the data directory."""
    for data_file in get_dataset_paths():
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Generating costmaps for data file: {data_file}")

        for idx, instance in enumerate(data['training_data']):
            instance: StepNode
            if instance.cost_map is None:
                logger.warning(f"No cost map found for instance {idx} in {data_file}, skipping.")
                continue
            cost_map: np.ndarray = instance.cost_map
            # reshape cost map to 4,n,m
            cost_map = cost_map.reshape((4, *fs._depricated_grid_size))

            title = f"{data_file.name} - {idx}"
            view_footstep_cost_map(
                cost_map=cost_map,
                title=title,
                save_figure=True,
            )

def generate_footstep_placement_maps():
    """Generates 2D histogram maps of footstep placements."""

    # Single plot with all feet overlaid
    fig, ax = plt.subplots(figsize=(10, 10))
    colormaps = ['Reds', 'Greens', 'Blues', 'Purples']
    foot_positions_all = [[], [], [], []]

    for data_file in get_dataset_paths():
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        
        for datum in data['training_data']:
            datum: StepNode
            foot_positions = datum.state.obs.foot_positions_b
            for i in range(4):
                foot_positions_all[i].append(foot_positions[i][::-1])
    
    # Calculate max_density across all histograms
    max_density = 1
    for foot_positions in foot_positions_all:
        foot_positions = np.array(foot_positions)
        x = foot_positions[:, 0]
        y = foot_positions[:, 1]
        H, _, _ = np.histogram2d(x, y, bins=25)
        max_density = max(max_density, np.max(H))
    
    norm = colors.LogNorm(1, float(max_density))
    
    for foot_positions, cmap in zip(foot_positions_all, colormaps):
        foot_positions = np.array(foot_positions)
        x = foot_positions[:, 0]
        y = foot_positions[:, 1]
        ax.hist2d(x, y, bins=75, cmap=cmap, alpha=1.0, norm=norm)
    
    # Set limits to enclose all data
    all_x = []
    all_y = []
    for foot_positions in foot_positions_all:
        foot_positions = np.array(foot_positions)
        all_x.extend(foot_positions[:, 0])
        all_y.extend(foot_positions[:, 1])
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Lateral Foot Offset (m)")
    ax.set_ylabel("Longitudinal Foot Offset (m)")
    # ax.set_title("Footstep Placement Heatmaps")

    # Add colorbar for density estimation
    mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Footstep Placement Density\n(Scale shared across feet)")

    plt.tight_layout()
    plt.show()


def main():
    log_data_info()
    if args.costmaps:
        generate_all_costmaps()
    if args.placement_maps:
        generate_footstep_placement_maps()

if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        main()