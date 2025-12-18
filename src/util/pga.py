import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.ndimage import distance_transform_edt
import numpy as np


def precompute_distance_transform(terrain_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute distance transform for efficient projection to valid terrain.
    
    Args:
        terrain_mask: (batch, height, width) binary mask where 1=valid, 0=invalid
    
    Returns:
        nearest_i: (batch, height, width) nearest valid row index for each position
        nearest_j: (batch, height, width) nearest valid column index for each position
    """
    batch_size, height, width = terrain_mask.shape
    device = terrain_mask.device
    
    nearest_i = torch.zeros_like(terrain_mask, dtype=torch.long)
    nearest_j = torch.zeros_like(terrain_mask, dtype=torch.long)
    
    # Process each batch element
    for b in range(batch_size):
        mask_np = terrain_mask[b].cpu().numpy()
        
        # Find nearest valid point for each position
        if mask_np.sum() == 0:
            # No valid terrain - default to center
            nearest_i[b] = height // 2
            nearest_j[b] = width // 2
        else:
            # Compute distance transform
            invalid_mask = 1 - mask_np
            dist, indices = distance_transform_edt(invalid_mask, return_indices=True)
            
            # For valid points, they are their own nearest point
            valid_i, valid_j = np.where(mask_np)
            indices[0, valid_i, valid_j] = valid_i
            indices[1, valid_i, valid_j] = valid_j
            
            nearest_i[b] = torch.from_numpy(indices[0]).to(device)
            nearest_j[b] = torch.from_numpy(indices[1]).to(device)
    
    return nearest_i, nearest_j


def project_to_valid_terrain(
    x: torch.Tensor, 
    y: torch.Tensor, 
    leg: int,
    terrain_mask: torch.Tensor,
    nearest_i: torch.Tensor,
    nearest_j: torch.Tensor,
    pos_to_idx
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project positions to nearest valid terrain.
    
    Args:
        x, y: (batch,) continuous coordinates
        leg: leg index
        terrain_mask: (batch, height, width)
        nearest_i, nearest_j: precomputed nearest valid indices
        pos_to_idx: function to convert continuous coords to grid indices
    
    Returns:
        x_proj, y_proj: (batch,) projected valid coordinates
    """
    batch_size = x.shape[0]
    device = x.device
    
    # Check if current positions are valid
    in_bounds, grid_i, grid_j = pos_to_idx(x, y, leg)
    
    x_proj = x.clone()
    y_proj = y.clone()
    
    # For each batch element, project if needed
    for b in range(batch_size):
        if in_bounds[b] and terrain_mask[b, grid_i[b], grid_j[b]]:
            # Already valid, no projection needed
            continue
        
        # Need to project - find nearest valid point
        if in_bounds[b]:
            # In bounds but invalid terrain
            nearest_valid_i = nearest_i[b, grid_i[b], grid_j[b]]
            nearest_valid_j = nearest_j[b, grid_i[b], grid_j[b]]
        else:
            # Out of bounds - clamp to grid bounds first
            height, width = terrain_mask.shape[1:]
            clamped_i = torch.clamp(grid_i[b], 0, height - 1)
            clamped_j = torch.clamp(grid_j[b], 0, width - 1)
            nearest_valid_i = nearest_i[b, clamped_i, clamped_j]
            nearest_valid_j = nearest_j[b, clamped_i, clamped_j]
        
        # Convert back to continuous coordinates (simple approximation)
        # This assumes pos_to_idx does some scaling - adjust as needed for your coordinate system
        x_proj[b] = nearest_valid_j.float()
        y_proj[b] = nearest_valid_i.float()
    
    return x_proj, y_proj


def generate_footstep_action(
    state: torch.Tensor,
    terrain_mask: torch.Tensor,
    leg: int,
    gaitnet,
    pos_to_idx,
    guess: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    learning_rate: float = 0.1,
    num_iterations: int = 10
) -> torch.Tensor:
    """Generate footstep actions given the current state and terrain mask.
    
    Args:
        state: current state tensor (batch, ...)
        terrain_mask: (batch, height, width) mask indicating valid terrain locations
        leg: The leg index for which to generate the action
        gaitnet: GaitNet model
        pos_to_idx: function to convert (x, y, leg) to grid indices and validity
        guess: Optional (x, y) initial guess as tuple of (batch,) tensors
        learning_rate: step size for gradient ascent
        num_iterations: number of optimization iterations
    
    Returns:
        torch.Tensor: (batch, 5) generated footstep action [leg, x, y, duration, value]
    """
    batch_size = state.shape[0]
    device = state.device
    height, width = terrain_mask.shape[1:]
    
    # Precompute distance transform for efficient projection
    nearest_i, nearest_j = precompute_distance_transform(terrain_mask)
    
    # Initialize x, y
    if guess is not None:
        guess_x, guess_y = guess
        # Initialize with Gaussian noise around guess
        x = guess_x + torch.randn(batch_size, device=device) * 0.5
        y = guess_y + torch.randn(batch_size, device=device) * 0.5
    else:
        # Uniform random initialization
        x = torch.rand(batch_size, device=device) * width
        y = torch.rand(batch_size, device=device) * height
    
    # Project initial positions to valid terrain
    x, y = project_to_valid_terrain(x, y, leg, terrain_mask, nearest_i, nearest_j, pos_to_idx)
    
    # Make x, y require gradients
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    
    # Track best results
    best_value = torch.full((batch_size,), float('-inf'), device=device)
    best_x = x.clone().detach()
    best_y = y.clone().detach()
    best_duration = torch.zeros(batch_size, device=device)
    
    # Optimization loop
    for iteration in range(num_iterations):
        # Forward pass through GaitNet
        value, duration = gaitnet(state, x, y, leg)
        
        # Track best
        improved = value > best_value
        best_value = torch.where(improved, value, best_value)
        best_x = torch.where(improved.unsqueeze(-1), x.detach(), best_x)
        best_y = torch.where(improved.unsqueeze(-1), y.detach(), best_y)
        best_duration = torch.where(improved, duration.detach(), best_duration)
        
        # Gradient ascent (maximize value)
        value.sum().backward()
        
        with torch.no_grad():
            # Update positions
            x_new = x + learning_rate * x.grad
            y_new = y + learning_rate * y.grad
            
            # Project to valid terrain
            x_new, y_new = project_to_valid_terrain(
                x_new, y_new, leg, terrain_mask, nearest_i, nearest_j, pos_to_idx
            )
            
            # Update and reset gradients
            x.copy_(x_new)
            y.copy_(y_new)
            x.grad.zero_()
            y.grad.zero_()
    
    # Construct action tensor [leg, x, y, duration, value]
    leg_tensor = torch.full((batch_size, 1), leg, device=device, dtype=torch.float32)
    action = torch.cat([
        leg_tensor,
        best_x.unsqueeze(-1),
        best_y.unsqueeze(-1),
        best_duration.unsqueeze(-1),
        best_value.unsqueeze(-1)
    ], dim=-1)
    
    return action