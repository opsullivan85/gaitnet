import torch
import torch.nn as nn
import numpy as np
from gaitnet.contactnet.contactnet import FootstepDataset, ContactNet
from gaitnet.contactnet.debug import view_footstep_cost_map
import argparse
import time

from gaitnet.contactnet.util import get_checkpoint_path, get_dataset_paths
from gaitnet import get_logger

logger = get_logger()

parser = argparse.ArgumentParser(description="Data information")
parser.add_argument(
    "--samples",
    type=int,
    default=5,
    help="Number of samples to visualize",
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Run benchmark for model evaluation speed",
)
parser.add_argument(
    "--random-samples",
    action="store_true",
    help="Use random samples from the dataset instead of evenly spaced samples",
)
args, unused_args = parser.parse_known_args()


def compare_model_output(
    model: nn.Module,
    dataset: FootstepDataset,
    device: torch.device,
    num_samples: int = 5,
    random_samples: bool = True,
):
    """
    Visualize model outputs for a few samples from the dataset.

    Args:
        model (nn.Module): The trained ContactNet model.
        dataset (Dataset): The dataset to sample from.
        device (torch.device): The device to run the model on.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    if random_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        state, expected_costmap = dataset[idx]
        # Add batch dimension and move state to device
        state = state.unsqueeze(0).to(
            device
        )  # Add batch dimension: [input_dim] -> [1, input_dim]

        with torch.inference_mode():
            calculated_costmap = model(state)

        # Remove batch dimension before reshaping
        calculated_costmap = (
            calculated_costmap.squeeze(0)
            .cpu()
            .numpy()
            .reshape(4, 5, 5)  # Remove batch dim first
        )  # Reshape back to (4, 5, 5)
        expected_costmap = (
            expected_costmap.cpu().numpy().reshape(4, 5, 5)
        )  # Reshape back to (4, 5, 5)

        view_footstep_cost_map(
            calculated_costmap,
            title=f"Calculated Cost Map Sample {idx}",
            save_figure=True,
        )
        view_footstep_cost_map(
            expected_costmap, 
            title=f"Expected Cost Map Sample {idx}", 
            save_figure=True
        )


def benchmark_model_evaluation(
    model: nn.Module,
    device: torch.device,
    input_dim: int,
    num_runs: int = 1000,
):
    """
    Benchmark the model's evaluation speed by running it on random input data.

    Args:
        model (nn.Module): The trained ContactNet model.
        device (torch.device): The device to run the model on.
        input_dim (int): The input dimension of the model.
        num_runs (int): Number of evaluation runs to perform.
    """
    model.eval()
    with torch.inference_mode():
        start_time = time.time()
        for _ in range(num_runs):
            random_input = torch.randn(1, input_dim).to(device)
            _ = model(random_input)
        end_time = time.time()
        total_time = end_time - start_time
        evaluations_per_second = num_runs / total_time
        print(f"Evaluations per second: {evaluations_per_second:.2f}")


def main():

    # Load dataset
    dataset = FootstepDataset(get_dataset_paths())

    # Load model
    model = ContactNet(
        input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(get_checkpoint_path(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    compare_model_output(
        model,
        dataset,
        device=device,
        num_samples=args.samples,
        random_samples=args.random_samples,
    )

    if args.benchmark:
        benchmark_model_evaluation(model, device, dataset.input_dim)


if __name__ == "__main__":
    from gaitnet.util import log_exceptions

    with log_exceptions(logger):
        main()
