import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from isaaclab.envs import ManagerBasedRLEnv
from gaitnet.contactnet.tree import IsaacStateCPU, StepNode
from gaitnet import PROJECT_ROOT
from gaitnet.contactnet.util import get_checkpoint_path, get_dataset_paths
from gaitnet import get_logger
import gaitnet.simulation.cfg.footstep_scanner_constants as fs
import gaitnet.constants as const

logger = get_logger()


class FootstepDataset(Dataset):
    """
    Custom Dataset for quadruped locomotion data.

    This dataset handles the loading and preprocessing of StepNode data,
    flattening the state information and cost maps for neural network training.
    """

    def __init__(self, data_paths: list[Path]):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the pickled data file
        """
        logger.info(f"loading data from {data_paths}")

        self.training_data: List[StepNode] = []
        self.metadatas: List[dict] = []

        for data_path in data_paths:
            with open(data_path, "rb") as f:
                data = pickle.load(f)

            self.training_data.extend(data["training_data"])
            self.metadatas.append(data["metadata"])

        if not self._metadatas_compatable():
            raise ValueError(
                "Incompatible metadata across data files. See logs for details."
            )

        # Filter out samples with None cost_map
        # shouldn't be nessesary but just in case
        self.training_data = [
            node for node in self.training_data if node.cost_map is not None
        ]

        logger.info(f"loaded {len(self.training_data)} valid training samples")

        # Calculate input dimensions from the first sample
        sample_state = self.flatten_state(self.training_data[0].state)
        self.input_dim = len(sample_state)
        # we know the metadatas match for this entry
        self.output_dim = (
            self.metadatas[0]["footstep_grid_size"][0]
            * self.metadatas[0]["footstep_grid_size"][1]
        )
        logger.info(f"input dimension: {self.input_dim}")
        logger.info(f"output dimension: {self.output_dim}")

    @staticmethod
    def flatten_state(state: IsaacStateCPU) -> np.ndarray:
        """
        Flatten the robot state into a single vector.

        This combines joint positions, velocities, and body state into one
        feature vector for the neural network input.
        """
        return np.concatenate([
            state.obs.foot_positions_b.flatten(),
            [state.obs.height_w],
            state.obs.vel_b,
            state.obs.omega_b,
            state.obs.control,
        ])

    @staticmethod
    def normalize_cost_map(cost_map: np.ndarray) -> np.ndarray:
        """Normalize the cost map to [0, 1] range.
        where values only retain relative ordering information.

        as in \\cite{bratta2024contactnet}, except we are using 1-their values (lower is better)
        """
        # normalize each major index to [0, 1] range, excluding inf values
        finite_mask = np.isfinite(cost_map)
        finite_map = cost_map[finite_mask]
        minimums = np.min(finite_map)
        maximums = np.max(finite_map)
        # normalize cost_map. Ignore divide by zero since something
        # needs to have gone horribly wrong if max == min
        norm_map = (cost_map - minimums) / (maximums - minimums)
        # set inf values to 1 (worst possible cost)
        norm_map[~finite_mask] = 1.0
        return norm_map



    def _metadatas_compatable(self) -> bool:
        """Check if all metadata entries are compatiable."""
        first_metadata = self.metadatas[0]
        matching_entries = [
            "footstep_grid_size",
            "footstep_grid_resolution",
            "step_dt",
            "physics_dt",
            "iterations_between_mpc",
        ]
        for metadata in self.metadatas[1:]:
            for entry in matching_entries:
                if metadata[entry] != first_metadata[entry]:
                    logger.warning(
                        f"metadata entry '{entry}' does not match across data files: {metadata[entry]} != {first_metadata[entry]}"
                    )
                    logger.info("use --data-info to inspect data files.")
                    return False

        if any(
            [
                metadata["git_hash"] != first_metadata["git_hash"]
                for metadata in self.metadatas[1:]
            ]
        ):
            logger.warning(
                "git hashes do not match across data files.\n\tBe sure to verify compatibility manually."
            )

        return True

    def __len__(self) -> int:
        return len(self.training_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Tuple of (flattened_state, cost_maps) as tensors
        """
        node = self.training_data[idx]

        # Flatten the state for input
        state_flat = self.flatten_state(node.state)

        # normalize cost map
        norm_cost_map = self.normalize_cost_map(node.cost_map)  # type: ignore
        # Flatten cost map from (4, 5, 5) to (4, 25) for 4 separate foot models
        cost_map_flat = norm_cost_map.reshape(4, -1)  # Shape: (4, 25) # type: ignore

        return torch.FloatTensor(state_flat), torch.FloatTensor(cost_map_flat)


class FootstepModel(nn.Module):
    """
    Individual neural network model for a single foot.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super(FootstepModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # Reshape to 4D for conv layer (assuming 8x8 spatial dimensions)
            nn.Unflatten(1, (1, 8, 8)),  # Reshape 64 -> (1, 8, 8)
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten back to 1D for final linear layer
            nn.Linear(2 * 8 * 8, output_dim),  # Adjust input size accordingly
        )

        # Initialize weights using Xavier/Glorot initialization
        # This helps with gradient flow and training stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContactNet(nn.Module):
    """
    Combined model containing 4 foot models in a 'black box' architecture.

    This model trains 4 separate foot models simultaneously while keeping
    them logically separate for potential future modifications (e.g., mirror symmetry).
    """

    def __init__(self, input_dim: int, output_dim_per_foot: int):
        super(ContactNet, self).__init__()

        # Create 4 separate models for each foot
        # This design allows for future mirror symmetry implementation
        self.foot_models = nn.ModuleList(
            [FootstepModel(input_dim, output_dim_per_foot) for _ in range(4)]
        )

        logger.info(f"created ContactNet model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all 4 foot models.

        Args:
            x: Input state tensor of shape (batch_size, input_dim)

        Returns:
            Combined output tensor of shape (batch_size, 4, output_dim_per_foot)
        """
        outputs = []

        # Process input through each foot model
        for foot_model in self.foot_models:
            output = foot_model(x)
            outputs.append(output)

        # Stack outputs: (batch_size, 4, output_dim_per_foot)
        return torch.stack(outputs, dim=1)


class ContactNetTrainer:
    """
    Training class that handles the complete training pipeline.

    Implements best practices including:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Tensorboard logging
    - Gradient clipping for stability
    """

    def __init__(
        self,
        model: ContactNet,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function - MSE is appropriate for regression tasks
        self.criterion = nn.MSELoss()

        # Adam optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )

        # Learning rate scheduler - reduces LR when loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        # Setup logging and checkpointing
        self.setup_logging()

        # Early stopping parameters
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.early_stop_patience = 20

    def setup_logging(self):
        """Setup tensorboard logging and checkpoint directories."""
        timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

        # Create directories
        self.log_dir = PROJECT_ROOT / "training" / "contactnet" / "runs" / f"quadruped_{timestamp}"
        self.checkpoint_dir = PROJECT_ROOT / "training" / "contactnet" / "checkpoints" / f"quadruped_{timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        logger.info(f"tensorboard logs: {self.log_dir}")
        logger.info(f"checkpoints: {self.checkpoint_dir}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (states, cost_maps) in enumerate(self.train_loader):
            states = states.to(self.device)
            cost_maps = cost_maps.to(self.device)  # Shape: (batch_size, 4, 25)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(states)  # Shape: (batch_size, 4, 25)

            # Calculate loss
            loss = self.criterion(outputs, cost_maps)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

            # Log batch-level metrics
            if batch_idx % 100 == 0:
                logger.info(
                    f"epoch {epoch+1}, batch {batch_idx}/{num_batches}, loss: {loss.item():.6f}"
                )

                # Log to tensorboard
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar("Training/Batch_Loss", loss.item(), global_step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, epoch: int) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0

        with torch.inference_mode():
            for states, cost_maps in self.val_loader:
                states = states.to(self.device)
                cost_maps = cost_maps.to(self.device)

                outputs = self.model(states)
                loss = self.criterion(outputs, cost_maps)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"validation loss: {avg_loss:.6f}")

        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
        }

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"new best model saved with loss: {loss:.6f}")

    def train(self, num_epochs: int = 100):
        """Main training loop."""
        logger.info(f"starting training for {num_epochs} epochs")
        logger.info(f"device: {self.device}")
        logger.info(
            f"model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        for epoch in range(num_epochs):
            logger.info(f"epoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate(epoch)

            # Learning rate scheduling
            self.scheduler.step(val_loss if self.val_loader else train_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            self.writer.add_scalar("Training/Epoch_Loss", train_loss, epoch)
            if self.val_loader:
                self.writer.add_scalar("Validation/Loss", val_loss, epoch)
            self.writer.add_scalar("Training/Learning_Rate", current_lr, epoch)

            logger.info(
                f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )

            # Early stopping and checkpointing
            monitor_loss = val_loss if self.val_loader else train_loss
            is_best = monitor_loss < self.best_loss

            if is_best:
                self.best_loss = monitor_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint every 10 epochs or if best
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, monitor_loss, is_best)

            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                logger.info(f"early stopping at epoch {epoch + 1}")
                break

        logger.info("training completed!")
        self.writer.close()


def main():
    """Main training script."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    # Load dataset
    dataset = FootstepDataset(get_dataset_paths())

    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    # Batch size should be tuned based on available memory and dataset size
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
    )

    logger.info(f"train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    # Create model
    model = ContactNet(input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim)

    # Create trainer and start training
    trainer = ContactNetTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device
    )

    # Train the model
    trainer.train(num_epochs=200)


class CostMapGenerator:
    def __init__(self, device: str, eval: bool = True) -> None:
        # Load model
        dataset = FootstepDataset(get_dataset_paths())
        self.model = ContactNet(
            input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim
        )
        del dataset
        checkpoint = torch.load(get_checkpoint_path(), map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        if eval:
            self.model.eval()

    def predict(self, contactnet_obs: torch.Tensor) -> torch.Tensor:
        """Predict footstep cost maps from the current environment state.

        Args:
            env (ManagerBasedRLEnv): The environment containing the robot state.
            contactnet_obs (torch.Tensor): Preprocessed observation tensor of shape (num_envs, obs_dim).
                expects the same observation as flatten_state in FootstepDataset.
        
        Returns:
            torch.Tensor: Predicted cost maps of shape (num_envs, 4, 5, 5)
        """
        costmaps = self.model(contactnet_obs).reshape(-1, 4, const.contact_net.grid_size[0], const.contact_net.grid_size[1])

        return costmaps




if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        main()
