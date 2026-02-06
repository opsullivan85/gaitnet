from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gaitnet.constants as const


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
    learning_rate: float = 0.001,
    num_iterations: int = 4,
    num_samples: int = 8,
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
    debug_plots = const.experiments.pga_debug_plots
    
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
    # Note: one-hot is [no_op, leg0, leg1, leg2, leg3], so add 1 to leg index
    leg_one_hot = F.one_hot(
        torch.full((batch_size, num_samples), leg + 1, device=device, dtype=torch.long),
        num_classes=5
    ).float()
    
    # Expand state for samples: (batch, state_dim) -> (batch, num_samples, state_dim)
    state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
    
    # Debug: track all sample trajectories for first batch element
    if debug_plots:
        all_trajectory_x = []  # list of (num_samples,) arrays per iteration
        all_trajectory_y = []
        all_trajectory_val = []
    
    # PGA optimization loop
    for iteration in range(num_iterations):
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
        
        # Debug: track all samples for first batch element
        if debug_plots:
            all_trajectory_x.append(x[0].detach().cpu().numpy().copy())
            all_trajectory_y.append(y[0].detach().cpu().numpy().copy())
            all_trajectory_val.append(value[0].detach().cpu().numpy().copy())
        
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
    
    # Debug: generate visualization plots
    if debug_plots:
        _generate_pga_debug_plot(
            leg=leg,
            terrain_mask=terrain_mask[0],
            all_trajectory_x=all_trajectory_x,
            all_trajectory_y=all_trajectory_y,
            all_trajectory_val=all_trajectory_val,
            final_x=best_x[0].cpu().item(),
            final_y=best_y[0].cpu().item(),
            final_val=best_value[0].cpu().item(),
            best_sample_idx=best_idx[0].cpu().item(),
            state=state[0:1],
            gaitnet=gaitnet,
            pos_to_idx=pos_to_idx,
            idx_to_pos=idx_to_pos,
            device=device,
        )
    
    return action, best_value.unsqueeze(-1)


def _generate_pga_debug_plot(
    leg: int,
    terrain_mask: torch.Tensor,
    all_trajectory_x: list,
    all_trajectory_y: list,
    all_trajectory_val: list,
    final_x: float,
    final_y: float,
    final_val: float,
    best_sample_idx: int,
    state: torch.Tensor,
    gaitnet,
    pos_to_idx: Callable,
    idx_to_pos: Callable,
    device: torch.device,
):
    """Generate debug visualization of PGA optimization.
    
    Creates a figure with:
    - Left: Coarse grid evaluation of model value across valid terrain + all sample trajectories
    - Right: Value evolution over iterations for all samples
    """
    # Create output directory
    output_dir = Path("data/debug-images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    height, width = terrain_mask.shape
    terrain_np = terrain_mask.cpu().numpy()
    
    # Generate coarse grid of positions to evaluate
    # Use idx_to_pos to get the coordinate range
    corner_indices = torch.tensor([[0, 0], [height-1, width-1]], device=device)
    corner_xy = idx_to_pos(corner_indices)
    x_min, y_min = corner_xy[0, 0].item(), corner_xy[0, 1].item()
    x_max, y_max = corner_xy[1, 0].item(), corner_xy[1, 1].item()
    
    # Create coarse grid (use ~20x20 for visualization)
    grid_resolution = 20
    grid_x = torch.linspace(x_min, x_max, grid_resolution, device=device)
    grid_y = torch.linspace(y_min, y_max, grid_resolution, device=device)
    grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
    
    # Flatten for batch evaluation
    eval_x = grid_xx.flatten()  # (grid_resolution^2,)
    eval_y = grid_yy.flatten()
    
    # Check which grid points are on valid terrain
    eval_xy = torch.stack([eval_x, eval_y], dim=-1)  # (N, 2)
    eval_indices, in_bounds = pos_to_idx(eval_xy)
    eval_i = eval_indices[:, 0].clamp(0, height - 1).long()
    eval_j = eval_indices[:, 1].clamp(0, width - 1).long()
    valid_mask = terrain_mask[eval_i, eval_j].bool() & in_bounds
    
    # Evaluate gaitnet on all grid points
    with torch.no_grad():
        # Note: one-hot is [no_op, leg0, leg1, leg2, leg3], so add 1 to leg index
        leg_one_hot = F.one_hot(
            torch.full((len(eval_x),), leg + 1, device=device, dtype=torch.long),
            num_classes=5
        ).float()
        state_expanded = state.expand(len(eval_x), -1)
        
        observation = torch.cat(
            [
                state_expanded,
                leg_one_hot,
                eval_x.unsqueeze(-1),
                eval_y.unsqueeze(-1),
                torch.zeros_like(eval_x.unsqueeze(-1)),
            ],
            dim=-1,
        )
        
        value_flat, _ = gaitnet(observation)
        value_grid = value_flat.view(grid_resolution, grid_resolution).cpu().numpy()
    
    # Mask invalid terrain with NaN for visualization
    valid_grid = valid_mask.view(grid_resolution, grid_resolution).cpu().numpy()
    value_grid_masked = np.where(valid_grid, value_grid, np.nan)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Value landscape + trajectory
    ax1 = axes[0]
    
    # Plot value heatmap
    extent = [y_min, y_max, x_min, x_max]  # Note: imshow expects [left, right, bottom, top]
    im = ax1.imshow(
        value_grid_masked,
        extent=extent,
        origin='lower',
        cmap='viridis',
        aspect='equal',
    )
    plt.colorbar(im, ax=ax1, label='Value')
    
    # Overlay terrain mask (show invalid as semi-transparent red)
    terrain_display = np.where(terrain_np, np.nan, 0.5)
    terrain_extent = [y_min, y_max, x_min, x_max]
    ax1.imshow(
        np.ones((height, width, 4)) * [1, 0, 0, 0.3],  # Red with alpha
        extent=terrain_extent,
        origin='lower',
        aspect='equal',
        alpha=np.where(terrain_np, 0, 0.3),
    )
    
    # Plot all sample trajectories
    if len(all_trajectory_x) > 0:
        num_iterations = len(all_trajectory_x)
        num_samples = len(all_trajectory_x[0])
        
        # Convert to arrays: (num_iterations, num_samples)
        traj_x = np.array(all_trajectory_x)
        traj_y = np.array(all_trajectory_y)
        
        # Create color map for samples
        sample_colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
        
        # Plot each sample's trajectory
        for s in range(num_samples):
            sample_x = traj_x[:, s]
            sample_y = traj_y[:, s]
            
            # Draw trajectory line (thicker for best sample)
            linewidth = 3 if s == best_sample_idx else 1
            alpha = 1.0 if s == best_sample_idx else 0.5
            ax1.plot(sample_y, sample_x, '-', color=sample_colors[s], 
                    linewidth=linewidth, alpha=alpha)
            
            # Mark start position
            ax1.scatter(sample_y[0], sample_x[0], c=[sample_colors[s]], 
                       s=80, marker='o', edgecolors='white', linewidth=1, zorder=10)
            
            # Mark end position (before final selection)
            ax1.scatter(sample_y[-1], sample_x[-1], c=[sample_colors[s]], 
                       s=80, marker='s', edgecolors='white', linewidth=1, zorder=10)
    
    # Mark final best position
    ax1.scatter(final_y, final_x, c='red', s=300, marker='*', edgecolors='black', linewidth=2, label='Final Best', zorder=11)
    
    ax1.set_xlabel('Y (m)')
    ax1.set_ylabel('X (m)')
    ax1.set_title(f'PGA Value Landscape (Leg {leg})\nFinal value: {final_val:.4f}')
    ax1.legend(loc='upper right')
    
    # Right plot: Value evolution over iterations for all samples
    ax2 = axes[1]
    if len(all_trajectory_val) > 0:
        num_iterations = len(all_trajectory_val)
        num_samples = len(all_trajectory_val[0])
        iterations = list(range(num_iterations))
        
        # Convert to array: (num_iterations, num_samples)
        traj_val = np.array(all_trajectory_val)
        
        # Plot each sample's value evolution
        for s in range(num_samples):
            linewidth = 2 if s == best_sample_idx else 1
            alpha = 1.0 if s == best_sample_idx else 0.4
            label = f'Sample {s} (best)' if s == best_sample_idx else None
            ax2.plot(iterations, traj_val[:, s], '-o', color=sample_colors[s],
                    linewidth=linewidth, markersize=4, alpha=alpha, label=label)
        
        ax2.axhline(y=final_val, color='r', linestyle='--', linewidth=2, label=f'Final: {final_val:.4f}')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Sample Value')
    ax2.set_title('Value Evolution During PGA (All Samples)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    import time
    timestamp = int(time.time() * 1000)
    filename = output_dir / f"pga_debug_{timestamp}_leg{leg}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"PGA debug plot saved to: {filename}")
