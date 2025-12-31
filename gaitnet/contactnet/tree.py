from dataclasses import dataclass
import random
from typing import Any
import anytree
import torch

import numpy as np
from nptyping import Float32, NDArray, Shape, Bool
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# TODO: the ammount of transfering small arrays from GPU to CPU is slowing things down a ton
@dataclass()
class Observation:
    """Stores data used in the ML model"""
    foot_positions_b: NDArray[Shape["4, 2"], Float32]
    """XY positions of the feet relative to the body frame."""
    height_w: float
    """Height of the body in the world frame."""
    vel_b: NDArray[Shape["3"], Float32]
    """Velocity of the body in the body frame."""
    omega_b: NDArray[Shape["3"], Float32]
    """Angular velocity of the body in the body frame."""
    control: NDArray[Shape["3"], Float32]
    """Last control input."""

    @staticmethod
    def from_idxs(env: "ManagerBasedEnv", idxs: np.ndarray) -> list["Observation"]:
        """Create a list of Observations from an environment and a list of indices.

        Args:
            env (ManagerBasedEnv): The environment to extract observations from.
            idxs (np.ndarray): The indices of the environments to extract observations from.

        Returns:
            list[Observation]: A list of Observation objects.
        """
        foot_positions_b = env.scene["foot_transforms"].data.target_pos_source[idxs].cpu().numpy()[:, :, :2]

        body_states = env.scene["robot"].data.root_com_pose_w[idxs].cpu().numpy()
        heights_w = body_states[:, 2]
        vel_bs = env.scene["robot"].data.root_link_lin_vel_b[idxs].cpu().numpy()
        omega_bs = env.scene["robot"].data.root_link_ang_vel_b[idxs].cpu().numpy()

        return [
            Observation(
                foot_positions_b=foot_positions_b[i],
                height_w=heights_w[i],
                vel_b=vel_bs[i],
                omega_b=omega_bs[i],
                control=np.zeros((3,), dtype=np.float32)  # placeholder, must be set after
            ) for i in range(len(idxs))
        ]

@dataclass()
class IsaacStateCPU:
    """Class for keeping track of an Isaac state."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_state: np.ndarray
    obs: Observation

    def to_torch(self, device: Any) -> "IsaacStateTorch":
        """Convert to torch tensors on the specified device."""
        return IsaacStateTorch(
            joint_pos=torch.from_numpy(self.joint_pos).to(device),
            joint_vel=torch.from_numpy(self.joint_vel).to(device),
            body_state=torch.from_numpy(self.body_state).to(device),
            obs=self.obs,
        )
    
    @staticmethod
    def from_idxs(env: "ManagerBasedEnv", idxs: np.ndarray) -> list["IsaacStateCPU"]:
        """Create a list of IsaacStateCPU from an environment and a list of indices.
        obs.control must be manually set after
        """
        torch_states = IsaacStateTorch.from_idxs(env, idxs)
        return [s.to_numpy() for s in torch_states]


@dataclass()
class IsaacStateTorch:
    """Class for keeping track of an Isaac state."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_state: torch.Tensor
    obs: Observation

    def to_numpy(self) -> IsaacStateCPU:
        """Convert to numpy arrays."""
        return IsaacStateCPU(
            joint_pos=self.joint_pos.cpu().numpy(),
            joint_vel=self.joint_vel.cpu().numpy(),
            body_state=self.body_state.cpu().numpy(),
            obs=self.obs,
        )
    
    @staticmethod
    def from_idxs(env: "ManagerBasedEnv", idxs: np.ndarray) -> list["IsaacStateTorch"]:
        """Create a list of IsaacStateTorch from an environment and a list of indices.
        obs.control must be manually set after
        """
        joint_pos = env.scene["robot"].data.joint_pos[idxs]
        joint_vel = env.scene["robot"].data.joint_vel[idxs]
        body_state = env.scene["robot"].data.root_state_w[idxs]
        observations = Observation.from_idxs(env, idxs)

        return [
            IsaacStateTorch(
                joint_pos=joint_pos[i],
                joint_vel=joint_vel[i],
                body_state=body_state[i],
                obs=observations[i]
            ) for i in range(len(idxs))
        ]





@dataclass()
class StepNode:
    """Initialize a StepNode.

    Args:
        state (IsaacStateCPU): The state of the robot.
        cost_map (NDArray[Shape["4, N, M"], Float32] | None): The cost map for the step.
            None if not computed yet.
    """
    state: IsaacStateCPU
    cost_map: NDArray[Shape["4, N, M"], Float32] | None


class TreeNode(anytree.NodeMixin):
    def __init__(self, data: StepNode, dead: bool = False, action: int | None = None, **kwargs) -> None:
        """Initialize a TreeNode.

        Args:
            data (StepNode): The data associated with the node.
            dead (bool): Whether the node is dead.
            action (int | None): The action taken. None if root node.
        """
        self.data = data
        self.dead = dead
        self.action = action

        # dynamically set all properties from kwargs
        # NodeMixin uses parent and child setters instead of __init__
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return super().__repr__() + f" (dead={self.dead})"

    def _mark_dead(self) -> None:
        """Mark the node and all its descendants as dead."""
        self.dead = True
        for child in self.descendants:
            child.dead = True
    
    def mark_dead(self, height: int) -> None:
        """Mark the node's parent at a certain height, and all its descendants, as dead.

        will specifically not mark the root as dead.

        Args:
            height (int): The height at which to mark the node as dead.
                if the height is greater than the number of ancestors, the root will be marked dead.
                0 means the node itself.
                1 means the parent.
                etc.
        """
        if height < 0:
            raise ValueError("Height must be non-negative.")
        if height == 0:
            self._mark_dead()
        # note that self.ancestor's 0th index is the root
        elif height <= len(self.ancestors):
            # do not kill the root node
            if height == len(self.ancestors):
                height -= 1
            ancestor = self.ancestors[-height]
            ancestor._mark_dead()
        else:
            # Mark the root as dead
            root = self.root
            root._mark_dead()

    def get_explored_nodes(self) -> list["TreeNode"]:
        """Get all non-dead leaf nodes in the subtree rooted at this node.
        also excludes this node (the root)"""
        nodes: list["TreeNode"] = [n for n in self.descendants if not n.dead and not n.is_leaf]
        return nodes
    
    def get_living_leaf(self) -> "TreeNode":
        """Get all non-dead leaf nodes in the subtree rooted at this node.

        weights to prioritize depth
        """
        leaves: list["TreeNode"] = [n for n in self.leaves if not n.dead]
        if not leaves:
            raise ValueError("No living leaves found.")

        # +1 prevents root from having 0 weight
        weights = [n.depth**2 + 1 for n in leaves]
        return random.choices(leaves, weights=weights, k=1)[0]
    
    def add_best_children(self, n: int, cost_map: np.ndarray, terminal_states: np.ndarray) -> None:
        """Add the best n children based on the cost map.

        Args:
            n (int): The number of children to add.
            cost_map (np.ndarray): The cost map for the step.
            terminal_states (np.ndarray): The terminal states corresponding to the cost map.
        """
        flat_cost_map = cost_map.flatten()
        lowest_indices = np.argpartition(flat_cost_map, n)[:n]
        for index in lowest_indices:
            state: IsaacStateCPU = terminal_states.flatten()[index]  # type: ignore
            action = int(index)
            child_data = StepNode(state=state, cost_map=None)
            TreeNode(data=child_data, parent=self, action=action)

    def depth_distribution(self) -> dict[int, int]:
        """Get the distribution of depths of all explored, non-dead nodes in the subtree rooted at this node.

        Returns:
            dict[int, int]: Number of nodes at each depth.
        """
        distribution: dict[int, int] = {}
        nodes = [n for n in self.descendants if not n.dead and not n.is_leaf] + ([self] if not self.dead else [])
        for node in nodes:
            depth = node.depth
            if depth not in distribution:
                distribution[depth] = 0
            distribution[depth] += 1
        # re-order by depth
        distribution = dict(sorted(distribution.items()))
        return distribution


# def expand(node):
#     """Stub simulation: creates 10 children with random states."""
#     success = random.random() > 0.2  # 80% chance success
#     if not success:
#         return False
#     for i in range(10):
#         SimNode(state=f"{node.state}-{i}", action=i, parent=node)
#     return True


# def random_non_dead_leaf(root):
#     leaves = [n for n in root.leaves if not n.dead]
#     return random.choice(leaves) if leaves else None


# def collect_non_dead_nodes(root):
#     return [n for n in root.descendants if not n.dead] + (
#         [root] if not root.dead else []
#     )


# # --- Example run ---
# root = SimNode("root")

# for _ in range(30):  # number of expansions
#     node = random_non_dead_leaf(root)
#     if node is None:
#         break
#     success = expand(node)
#     if not success:
#         mark_greatgrandparent_dead(node)

# nodes = collect_non_dead_nodes(root)
# print("Collected nodes:", [n.state for n in nodes])
