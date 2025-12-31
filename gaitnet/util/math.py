import numpy as np
import torch


# https://stackoverflow.com/a/56207565
def _quat_to_euler(w, x, y, z, backend) -> tuple:
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = backend.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = backend.clip(t2, -1.0, 1.0)
    Y = backend.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = backend.arctan2(t3, t4)

    return X, Y, Z


def quat_to_euler_np(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts quaternion (w, x, y, z) to euler angles (X, Y, Z)
    in radians

    Args:
        w: Scalar component of the quaternion.
        x: X component of the quaternion.
        y: Y component of the quaternion.
        z: Z component of the quaternion.

    Returns:
        A tuple of three numpy arrays representing the Euler angles (X, Y, Z) in radians.
    """
    return _quat_to_euler(w, x, y, z, backend=np)


def quat_to_euler_torch(
    w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts quaternion (w, x, y, z) to euler angles (X, Y, Z)
    in radians

    Args:
        w: Scalar component of the quaternion.
        x: X component of the quaternion.
        y: Y component of the quaternion.
        z: Z component of the quaternion.

    Returns:
        A tuple of three tensors representing the Euler angles (X, Y, Z) in radians.
    """
    return _quat_to_euler(w, x, y, z, backend=torch)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Fast 64-bit hash function for pseudo-randomness, vectorized on GPU."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x

def _mix32(x: torch.Tensor) -> torch.Tensor:
    """Cheap 32-bit integer mix function, vectorized on GPU."""
    x = (x ^ (x >> 17)) * 0xED5AD4BB & 0xFFFFFFFF
    x = (x ^ (x >> 11)) * 0xAC4C1B51 & 0xFFFFFFFF
    x = (x ^ (x >> 15)) * 0x31848BAB & 0xFFFFFFFF
    x = x ^ (x >> 14)
    return x & 0xFFFFFFFF


def seeded_uniform_noise(seed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Generate fast, deterministic pseudo-random noise for each seed row.

    Chat's implementation of a 32 bit linear congruential generator (LCG) seeded
    with a tensor.

    importantly, each row in the output is determined only by the corresponding row in the seed.
    so re-ordering the major axis of the seed will re-order the output in the same way.

    Args:
        seed (torch.Tensor): Input tensor of shape (num_envs, obs_dim).
        shape (tuple, optional): Desired output shape, without batch dimension.

    Returns:
        torch.Tensor: Pseudo-random noise tensor of shape (num_envs, *shape) on the same device as seed.
    """
    num_envs = seed.shape[0]
    flat_size = int(torch.prod(torch.tensor(shape, device=seed.device)))

    # --- Step 1: flatten seed into row integers ---
    seed_flat = seed.reshape(num_envs, -1)
    seed_int = (seed_flat * 1e3).to(torch.int64)

    # simple low-entropy row seed (sum + mod 2^32)
    seeds = seed_int.sum(dim=1) % (2**32)   # (num_envs,)

    # --- Step 2: expand across required length ---
    idx = torch.arange(flat_size, device=seed.device, dtype=torch.int64).unsqueeze(0)
    seeds = (seeds.unsqueeze(1) + idx) & 0xFFFFFFFF

    # --- Step 3: apply mixing hash for "grainy" noise ---
    rand_ints = _mix32(seeds)
    rand = rand_ints.float() / float(2**32)   # uniform [0,1)

    return rand.view((num_envs,) + shape)