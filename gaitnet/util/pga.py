import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from scipy.ndimage import distance_transform_edt
import numpy as np


def precompute_distance_transform(
    terrain_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute distance transform for efficient projection to valid terrain.

    Args:
        terrain_mask: (batch, height, width) binary mask where 1=valid, 0=invalid

    Returns:
        nearest_i: (batch, height, width) nearest valid row index for each position
        nearest_j: (batch, height, width) nearest valid column index for each position
    """
    device = terrain_mask.device
    batch_size, height, width = terrain_mask.shape
    
    nearest_i = torch.zeros_like(terrain_mask, dtype=torch.long)
    nearest_j = torch.zeros_like(terrain_mask, dtype=torch.long)
    
    # Process each batch element using scipy's distance_transform_edt
    for b in range(batch_size):
        mask_np = terrain_mask[b].cpu().numpy()
        # distance_transform_edt returns indices of nearest background (0) pixel
        # We want nearest valid (1) pixel, so invert the mask
        invalid_mask = 1 - mask_np
        
        if mask_np.sum() == 0:
            # No valid terrain - just use center as fallback
            nearest_i[b] = height // 2
            nearest_j[b] = width // 2
        else:
            # Get indices of nearest valid pixel for each position
            _, indices = distance_transform_edt(invalid_mask, return_indices=True)
            nearest_i[b] = torch.from_numpy(indices[0]).to(device)
            nearest_j[b] = torch.from_numpy(indices[1]).to(device)
    
    return nearest_i, nearest_j


def project_to_valid_terrain(
    x: torch.Tensor,
    y: torch.Tensor,
    nearest_i: torch.Tensor,
    nearest_j: torch.Tensor,
    pos_to_idx: Callable,
    idx_to_pos: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project positions to nearest valid terrain.

    Args:
        x, y: (batch, num_samples) continuous coordinates in meters
        nearest_i, nearest_j: (batch, height, width) precomputed nearest valid indices
        pos_to_idx: function to convert continuous coords to grid indices
        idx_to_pos: function to convert grid indices back to continuous coords

    Returns:
        x_proj, y_proj: (batch, num_samples) projected valid coordinates in meters
    """
    batch_size, num_samples = x.shape
    device = x.device
    
    # Convert continuous coords to grid indices
    xy = torch.stack([x, y], dim=-1)  # (batch, num_samples, 2)
    indices, in_bounds = pos_to_idx(xy)  # indices: (batch, num_samples, 2)
    
    # Clamp indices to valid range for lookup
    height, width = nearest_i.shape[1], nearest_i.shape[2]
    idx_i = indices[..., 0].clamp(0, height - 1).long()  # (batch, num_samples)
    idx_j = indices[..., 1].clamp(0, width - 1).long()   # (batch, num_samples)
    
    # Look up nearest valid indices for each sample
    # nearest_i/j are (batch, height, width), we need to gather at (idx_i, idx_j)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_samples)
    
    proj_i = nearest_i[batch_indices, idx_i, idx_j]  # (batch, num_samples)
    proj_j = nearest_j[batch_indices, idx_i, idx_j]  # (batch, num_samples)
    
    # Convert projected indices back to continuous coordinates
    proj_indices = torch.stack([proj_i, proj_j], dim=-1)  # (batch, num_samples, 2)
    proj_xy = idx_to_pos(proj_indices)  # (batch, num_samples, 2)
    
    x_proj = proj_xy[..., 0]
    y_proj = proj_xy[..., 1]
    
    return x_proj, y_proj


def generate_footstep_action(
    state: torch.Tensor,
    terrain_mask: torch.Tensor,
    leg: int,
    gaitnet,
    pos_to_idx: Callable,
    idx_to_pos: Optional[Callable] = None,
    guess: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    learning_rate: float = 0.1,
    num_iterations: int = 10,
    num_samples: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to use projected gradient ascent (PGA) to maximize footstep action value.
    called seperatley for each leg, with correct terrain masks.

    Internally, sets up a number of samples, which are iteratively updated via projected gradient ascent.
    At the end, the best action and its value are returned. Samples are initilized with a gaussian distribution
    (sigma=0.1) centered about (0 m, 0 m) unless a guess is provided.

    Args:
        state: current state tensor (batch, ...)
        terrain_mask: (batch, height, width) mask indicating valid terrain locations
            already specific to this leg. height and width are in grid space. Use
            pos_to_idx to convert from (x, y) coordinates in meters to grid indices.
        leg: The leg index for which to generate the action
            only really needed for the one-hot encoding in the observation, and for pos_to_idx
        gaitnet: GaitNet model
            Usage:
                observation = torch.concat(
                    [
                        state,
                        leg_one_hot,
                        x.unsqueeze(-1),
                        y.unsqueeze(-1),
                        torch.zeros_like(x.unsqueeze(-1)),
                    ],
                    dim=-1,
                )
                value, duration = gaitnet(observation)
        pos_to_idx: function to convert (x, y) coordinates to grid indices and validity
        idx_to_pos: function to convert grid indices back to (x, y) coordinates
        guess: optional initial (x, y) guess for the optimization
        learning_rate: step size for gradient ascent
        num_iterations: number of optimization iterations
        num_samples: number of samples per iteration

    Returns:
        torch.Tensor: (batch, 4) generated footstep action [leg, x, y, duration]
        torch.Tensor: (batch, 1) estimated value of the action
    """
    batch_size = state.shape[0]
    device = state.device
    
    # Check for completely invalid terrain - return early with zeros
    valid_per_batch = terrain_mask.sum(dim=(1, 2))  # (batch,)
    if (valid_per_batch == 0).all():
        # No valid terrain for any batch element
        action = torch.zeros((batch_size, 4), device=device)
        action[:, 0] = leg
        value = torch.full((batch_size, 1), float("-inf"), device=device)
        return action, value
    
    # Precompute distance transform for projection
    nearest_i, nearest_j = precompute_distance_transform(terrain_mask)
    
    # Initialize samples: (batch, num_samples) for x and y
    if guess is not None:
        # Center samples around the guess
        x_center, y_center = guess
        x = x_center.unsqueeze(1) + torch.randn(batch_size, num_samples, device=device) * 0.1
        y = y_center.unsqueeze(1) + torch.randn(batch_size, num_samples, device=device) * 0.1
    else:
        # Center samples around (0, 0)
        x = torch.randn(batch_size, num_samples, device=device) * 0.1
        y = torch.randn(batch_size, num_samples, device=device) * 0.1
    
    # Project initial samples to valid terrain
    x, y = project_to_valid_terrain(x, y, nearest_i, nearest_j, pos_to_idx, idx_to_pos)
    
    # Create leg one-hot encoding: (batch, num_samples, 5)
    leg_one_hot = F.one_hot(
        torch.full((batch_size, num_samples), leg, device=device, dtype=torch.long),
        num_classes=5
    ).float()
    
    # Expand state for samples: (batch, state_dim) -> (batch, num_samples, state_dim)
    state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
    
    # PGA optimization loop
    for _ in range(num_iterations):
        # Enable gradients for x and y
        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)
        
        # Build observation: (batch, num_samples, state_dim + 5 + 3)
        observation = torch.cat(
            [
                state_expanded,
                leg_one_hot,
                x.unsqueeze(-1),
                y.unsqueeze(-1),
                torch.zeros_like(x.unsqueeze(-1)),
            ],
            dim=-1,
        )
        
        # Flatten for gaitnet: (batch * num_samples, obs_dim)
        obs_flat = observation.view(batch_size * num_samples, -1)
        
        # Get value predictions
        value_flat, duration_flat = gaitnet(obs_flat)
        value = value_flat.view(batch_size, num_samples)
        
        # Compute gradients w.r.t. x and y (maximize value)
        total_value = value.sum()
        total_value.backward()
        
        # Gradient ascent step
        with torch.no_grad():
            x = x + learning_rate * x.grad
            y = y + learning_rate * y.grad
            
            # Project back to valid terrain
            x, y = project_to_valid_terrain(x, y, nearest_i, nearest_j, pos_to_idx, idx_to_pos)
    
    # Final evaluation to get values and durations
    with torch.no_grad():
        observation = torch.cat(
            [
                state_expanded,
                leg_one_hot,
                x.unsqueeze(-1),
                y.unsqueeze(-1),
                torch.zeros_like(x.unsqueeze(-1)),
            ],
            dim=-1,
        )
        obs_flat = observation.view(batch_size * num_samples, -1)
        value_flat, duration_flat = gaitnet(obs_flat)
        
        value = value_flat.view(batch_size, num_samples)
        duration = duration_flat.view(batch_size, num_samples)
        
        # Find best sample per batch element
        best_idx = value.argmax(dim=1)  # (batch,)
        batch_indices = torch.arange(batch_size, device=device)
        
        best_x = x[batch_indices, best_idx]
        best_y = y[batch_indices, best_idx]
        best_duration = duration[batch_indices, best_idx]
        best_value = value[batch_indices, best_idx]
        
        # Build action tensor: [leg, x, y, duration]
        action = torch.stack(
            [
                torch.full((batch_size,), leg, device=device, dtype=best_x.dtype),
                best_x,
                best_y,
                best_duration,
            ],
            dim=-1,
        )
        
        # Handle batch elements with no valid terrain
        invalid_batch = valid_per_batch == 0
        if invalid_batch.any():
            action[invalid_batch] = 0
            action[invalid_batch, 0] = leg
            best_value[invalid_batch] = float("-inf")
    
    return action, best_value.unsqueeze(-1)
