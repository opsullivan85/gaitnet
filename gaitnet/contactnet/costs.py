from typing import Callable

import numpy as np
from isaaclab.envs import ManagerBasedEnv
import torch
from gaitnet.sim2real import SimInterface
from gaitnet.util import VectorPool
from abc import abstractmethod, ABC
from gaitnet.contactnet.debug import view_footstep_cost_map
from gaitnet.util.math import quat_to_euler_torch
import gaitnet.simulation.cfg.footstep_scanner_constants as fs


class TerminalCost(ABC):
    @abstractmethod
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = "TerminalCost"

    @abstractmethod
    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Calculate the terminal cost for the given environment.

        Args:
            env (ManagerBasedEnv): The environment to calculate the terminal cost for.

        Returns:
            torch.Tensor: A tensor of shape (num_envs,) representing the terminal cost for each environment.
        """
        pass

    @abstractmethod
    def debug_plot(self, title: str | None, **kwargs):
        """Plot the cost map for debugging.

        Args:
            title (str | None): Title for the plot.
            **kwargs: Additional keyword arguments for the plotting function.
        """


class RunningCost(TerminalCost):
    @abstractmethod
    def update_running_cost(self, env: ManagerBasedEnv):
        """Update the running cost for the given environment.

        Note: The average running cost is returned with the terminal_cost() function.

        Args:
            env (ManagerBasedEnv): The environment to calculate the running cost for.
        """
        pass


class SimpleIntegrator(RunningCost):
    def __init__(
        self, weight: float, cost_function: Callable[[ManagerBasedEnv], torch.Tensor]
    ):
        super().__init__(weight)
        self.running_cost: None | torch.Tensor = None
        self.cost_function = cost_function
        self.name = f"Running: {self.cost_function.__name__}"
        self.updates = 0

    def update_running_cost(self, env: ManagerBasedEnv):
        if self.running_cost is None:
            self.running_cost = torch.zeros((env.num_envs,), device=env.device)
        self.running_cost += self.cost_function(env) * self.weight
        self.updates += 1

    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        if self.running_cost is None:
            return torch.zeros((env.num_envs,), device=env.device)
        return self.running_cost / self.updates

    def debug_plot(self, title: str | None, **kwargs):
        if self.running_cost is None:
            raise ValueError("Running cost has not been updated yet.")
        cost = self.terminal_cost(None).cpu().numpy()  # type: ignore
        title = (
            title
            if title is not None
            else self.name
        )
        view_footstep_cost_map(
            cost.reshape((4, fs._depricated_grid_size[0], fs._depricated_grid_size[1])), title=title, **kwargs
        )


class SimpleTerminalCost(TerminalCost):
    def __init__(
        self, weight: float, cost_function: Callable[[ManagerBasedEnv], torch.Tensor]
    ):
        super().__init__(weight)
        self.cost_function = cost_function
        self.name = f"Terminal: {self.cost_function.__name__}"
        self.previous_terminal_cost: None | torch.Tensor = None

    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        self.previous_terminal_cost = self.cost_function(env) * self.weight
        return self.previous_terminal_cost

    def debug_plot(self, title: str | None, **kwargs):
        if self.previous_terminal_cost is None:
            raise ValueError("Terminal cost has not been calculated yet.")
        title = (
            title
            if title is not None
            else self.name
        )
        view_footstep_cost_map(
            self.previous_terminal_cost.cpu()
            .numpy()
            .reshape((4, fs._depricated_grid_size[0], fs._depricated_grid_size[1])),
            title=title,
            **kwargs,
        )


class BallanceFootCosts(RunningCost):
    """Ballances the footstep cost map such that each foot has the same average cost.

    can help with metrics that inherently favor certian feet (front/back)
    """
    def __init__(self, cost_class: RunningCost, ballance_factor: float = 1.0):
        """

        Args:
            cost_class (RunningCost): The cost class to ballance.
            ballance_factor (float, optional): The factor by which to ballance the costs. Defaults to 1.0.
                1.0 means full ballance, 0.5 means half ballance, 0.0 means no ballance.
        """
        super().__init__(1.0)
        self.name = cost_class.name + " (bal)"
        self.cost_class = cost_class
        self.ballance_factor = ballance_factor
    
    def update_running_cost(self, env: ManagerBasedEnv):
        self.cost_class.update_running_cost(env)
    
    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        cost = self.cost_class.terminal_cost(env)
        # assume 4 feet, an even amount of costs each
        # Reshape to (4, num_locations) assuming foot-major flattening, then transpose to (num_locations, 4)
        num_locations = cost.shape[0] // 4
        cost_reshaped: torch.Tensor = cost.reshape(4, num_locations).T
        foot_means = torch.mean(cost_reshaped, dim=0)
        overall_mean = torch.mean(foot_means)
        # adjust all to overall mean
        foot_adjustments = overall_mean - foot_means
        adjusted_cost = cost_reshaped + foot_adjustments * self.ballance_factor
        # Transpose back and flatten
        return adjusted_cost.T.reshape(-1)
    
    def debug_plot(self, title: str | None, **kwargs):
        cost = self.terminal_cost(None).cpu().numpy()  # type: ignore
        title = (
            title
            if title is not None
            else self.name
        )
        view_footstep_cost_map(
            cost.reshape((4, fs._depricated_grid_size[0], fs._depricated_grid_size[1])), title=title, **kwargs
        )

class ControlErrorCost(RunningCost):
    def __init__(self, weight: float, control: torch.Tensor, env: ManagerBasedEnv):
        super().__init__(weight)
        self.name = "Running: control_error"
        self.control = control
        root_pose_w = env.scene["robot"].data.root_link_pose_w
        self.initial_position = root_pose_w[:, :3]
        initial_quat = root_pose_w[:, 3:]
        _, _, self.Z_initial = quat_to_euler_torch(
            initial_quat[:, 0],
            initial_quat[:, 1],
            initial_quat[:, 2],
            initial_quat[:, 3],
        )
        self.elapsed_time = 0.0
        self.previous_terminal_cost: None | torch.Tensor = None

    def update_running_cost(self, env: ManagerBasedEnv):
        self.elapsed_time += env.step_dt

    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        root_pose_w = env.scene["robot"].data.root_link_pose_w
        position = root_pose_w[:, :3]
        quat = root_pose_w[:, 3:]
        linear_displacement = position - self.initial_position

        _, _, Z_current = quat_to_euler_torch(
            quat[:, 0],
            quat[:, 1],
            quat[:, 2],
            quat[:, 3],
        )
        angular_displacement = Z_current - self.Z_initial
        # wrap to [-pi, pi]
        angular_displacement = (angular_displacement + np.pi) % (2 * np.pi) - np.pi

        linear_velocity = linear_displacement / self.elapsed_time
        angular_velocity = angular_displacement / self.elapsed_time

        linear_velocity_target = self.control[:, :2]
        angular_velocity_target = self.control[:, 2:]

        linear_error = torch.norm(
            linear_velocity[:, :2] - linear_velocity_target, dim=1
        )
        angular_error = torch.norm(
            angular_velocity - angular_velocity_target, dim=1
        )

        self.previous_terminal_cost = (linear_error + angular_error) * self.weight
        return self.previous_terminal_cost  # type: ignore

    def debug_plot(self, title: str | None, **kwargs):
        if self.previous_terminal_cost is None:
            raise ValueError("Terminal cost has not been calculated yet.")
        title = title if title is not None else self.name
        view_footstep_cost_map(
            self.previous_terminal_cost.cpu()
            .numpy()
            .reshape((4, fs._depricated_grid_size[0], fs._depricated_grid_size[1])),
            title=title,
            **kwargs,
        )

class ControllerSwingErrorCost(RunningCost):
    def __init__(self, weight: float, env: ManagerBasedEnv):
        super().__init__(weight)
        self.name = "Running: controller_swing_error"

        controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
        self.controller_swing_durations = controllers.call(
            SimInterface.get_swing_durations,
            mask=None,
        )
        self.swing_durations = np.zeros_like(self.controller_swing_durations)

    def update_running_cost(self, env: ManagerBasedEnv):
        self.swing_durations += env.step_dt

    def terminal_cost(self, env: ManagerBasedEnv) -> torch.Tensor:
        error = np.abs(self.controller_swing_durations - self.swing_durations)
        return torch.tensor(error.sum(axis=1) * self.weight, device=env.device)


def controller_swing_error(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Error based on the difference between controller swing time and actual swing time.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The error in seconds between controller swing time and actual swing time.
    """
    incomplete_swing_multiplier = 1.0  # penalize incomplete swings more

    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    controller_swing_durations = controllers.call(
        SimInterface.get_swing_durations,
        mask=None,
    )
    controller_swing_durations = torch.tensor(
        controller_swing_durations, device=env.device
    )
    # (num_envs, 4, 1) -> (num_envs, 4)
    controller_swing_durations = controller_swing_durations.squeeze(2)

    contact_forces = env.scene["contact_forces"].data.net_forces_w
    # robot is done if all feet have contact force above threshold
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    swing_complete = torch.all(foot_contacts, dim=1)
    # if the swing is complete take the last air time as the real swing duration
    # else take the current air time as the real swing duration
    real_swing_durations = torch.where(
        swing_complete.unsqueeze(1),
        env.scene["contact_forces"].data.last_air_time,
        env.scene["contact_forces"].data.current_air_time,
    )

    error = torch.abs(controller_swing_durations - real_swing_durations)
    error[~swing_complete] *= incomplete_swing_multiplier
    return error.sum(dim=1)  # sum over legs


def triangle_area(
    vertex1: torch.Tensor, vertex2: torch.Tensor, vertex3: torch.Tensor
) -> torch.Tensor:
    """Calculate the area of triangles using the cross product method.

    This function mimics the interface of tensorflow_graphics.geometry.representation.triangle.area().

    Args:
        vertex1 (torch.Tensor): First vertices of triangles, shape (..., 3)
        vertex2 (torch.Tensor): Second vertices of triangles, shape (..., 3)
        vertex3 (torch.Tensor): Third vertices of triangles, shape (..., 3)

    Returns:
        torch.Tensor: Areas of the triangles, shape (...)
    """
    # Calculate vectors from vertex1 to vertex2 and vertex1 to vertex3
    v1 = vertex2 - vertex1  # (..., 3)
    v2 = vertex3 - vertex1  # (..., 3)

    # Calculate cross product
    cross_product = torch.cross(v1, v2, dim=-1)  # (..., 3)

    # Calculate magnitude of cross product
    cross_magnitude = torch.norm(cross_product, dim=-1)  # (...)

    # Area is half the magnitude of the cross product
    area = 0.5 * cross_magnitude

    return area


def polygon_area_2d(vertices: torch.Tensor) -> torch.Tensor:
    """Calculate the area of 2D polygons using the Shoelace formula.

    Works for both convex and non-convex (simple) polygons.
    NOTE: this will not work for self-intersecting polygons.

    Args:
        vertices (torch.Tensor): Polygon vertices, shape (..., N, 2) where N is number of vertices

    Returns:
        torch.Tensor: Areas of the polygons, shape (...)
    """
    # Shift vertices to get pairs (x_i, y_i) and (x_{i+1}, y_{i+1})
    x = vertices[..., 0]  # (..., N)
    y = vertices[..., 1]  # (..., N)

    # Roll to get next vertices (cyclically)
    x_next = torch.roll(x, shifts=-1, dims=-1)
    y_next = torch.roll(y, shifts=-1, dims=-1)

    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    cross_terms = x * y_next - x_next * y
    area = 0.5 * torch.abs(torch.sum(cross_terms, dim=-1))

    return area


def distance_from_virtual_line(
    point: torch.Tensor, line_start: torch.Tensor, line_end: torch.Tensor
) -> torch.Tensor:
    """Calculate the shortest distance from a point to a line defined by two points in 2D.

    Args:
        point (torch.Tensor): The point to measure the distance from, shape (..., 2)
        line_start (torch.Tensor): One endpoint of the line segment, shape (..., 2)
        line_end (torch.Tensor): The other endpoint of the line segment, shape (..., 2)

    Returns:
        torch.Tensor: The shortest distance from the point to the line segment, shape (...)
    """
    p0 = point
    x0 = p0[:, 0]
    y0 = p0[:, 1]
    p1 = line_start
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = line_end
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    m_numerator = y2 - y1
    m_denominator = x2 - x1
    # Avoid division by zero, prevent branching
    m_denominator = torch.where(m_denominator == 0, torch.tensor(1e-8), m_denominator)
    m = m_numerator / m_denominator

    d_numerator = torch.abs(m * x0 - y0 + y2 - m * x2)
    d_denominator = torch.sqrt(m * m + 1)
    distance = d_numerator / d_denominator

    return distance


def support_polygon_area(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Calculate the area of the support polygon formed by the feet in contact with the ground.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The area of the support polygon for each environment.
    """
    # use triangel.area to calculate the area of the support polygon
    # first check for which feet are in contact
    # if less than 3 feet are in contact, area is 0
    # if 3 feet are in contact, area is the area of the triangle formed by the feet
    # if 4 feet are in contact, area is the sum of the areas of the two triangles formed by the feet
    contact_forces = env.scene["contact_forces"].data.net_forces_w
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices]  # type: ignore
    num_contacts = foot_contacts.sum(dim=1)

    areas = torch.zeros((env.num_envs,), device=env.device)
    # 3 contacts -> area = area of triangle
    three_contacts = num_contacts == 3
    if torch.any(three_contacts):
        foot_positions_3 = foot_positions[three_contacts]  # (N, 4, 3)
        foot_contacts_3 = foot_contacts[three_contacts]  # (N, 4)
        # reshape is needed because torch implicitly flattens for some reason
        foot_contact_positions = foot_positions_3[foot_contacts_3].reshape(
            -1, 3, 3
        )  # (N, 3, 3)
        areas[three_contacts] = triangle_area(
            foot_contact_positions[:, 0],
            foot_contact_positions[:, 1],
            foot_contact_positions[:, 2],
        )
    # 4 contacts -> area = area of quadrilateral using Shoelace formula
    four_contacts = num_contacts == 4
    if torch.any(four_contacts):
        foot_positions_4 = foot_positions[four_contacts]  # (N, 4, 3)
        # Reorder feet to avoid self-intersecting polygon: [FL, FR, RL, RR] -> [FR, FL, RL, RR]
        foot_positions_4 = foot_positions_4[:, [1, 0, 2, 3]]  # (N, 4, 3)
        # Project to 2D (use x, y coordinates) for the Shoelace formula
        foot_positions_2d = foot_positions_4[..., :2]  # (N, 4, 2)
        areas[four_contacts] = polygon_area_2d(foot_positions_2d)

    return areas


def inscribed_circle_radius(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Calculate the radius of the inscribed circle of the support polygon formed by the feet in contact with the ground.

    NOTE: this assumes that the COM is inside the support polygon. If the COM is outside the support polygon,
        the radius will be positive, and non-sensical.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The radius of the inscribed circle of the support polygon for each environment.
    """
    contact_forces = env.scene["contact_forces"].data.net_forces_w
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices]  # type: ignore
    com_position = env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
    num_contacts = foot_contacts.sum(dim=1)

    # Initialize radii to zero
    radii = torch.zeros((env.num_envs,), device=env.device)
    # use distance_from_virtual_line to calculate the radius of the inscribed circle
    three_contacts = num_contacts == 3
    if torch.any(three_contacts):
        foot_positions_3 = foot_positions[three_contacts]  # (N, 4, 3)
        foot_contacts_3 = foot_contacts[three_contacts]  # (N, 4)
        com_position_3 = com_position[three_contacts]  # (N, 2)
        # reshape is needed because torch implicitly flattens for some reason
        foot_contact_positions = foot_positions_3[foot_contacts_3].reshape(
            -1, 3, 3
        )  # (N, 3, 3)
        foot_contact_positions_2d = foot_contact_positions[:, :, :2]  # (N, 3, 2)
        distances = torch.stack(
            [
                distance_from_virtual_line(
                    com_position_3,
                    foot_contact_positions_2d[:, 0],
                    foot_contact_positions_2d[:, 1],
                ),
                distance_from_virtual_line(
                    com_position_3,
                    foot_contact_positions_2d[:, 1],
                    foot_contact_positions_2d[:, 2],
                ),
                distance_from_virtual_line(
                    com_position_3,
                    foot_contact_positions_2d[:, 2],
                    foot_contact_positions_2d[:, 0],
                ),
            ],
            dim=1,
        )  # (N, 3)
        radii[three_contacts] = torch.min(distances, dim=1).values

    four_contacts = num_contacts == 4
    if torch.any(four_contacts):
        foot_positions_4 = foot_positions[four_contacts]  # (N, 4, 3)
        com_position_4 = com_position[four_contacts]  # (N, 2)
        # Reorder feet to avoid self-intersecting polygon: [FL, FR, RL, RR] -> [FR, FL, RL, RR]
        foot_positions_4 = foot_positions_4[:, [1, 0, 2, 3]]  # (N, 4, 3)
        foot_positions_2d = foot_positions_4[..., :2]  # (N, 4, 2)
        distances = torch.stack(
            [
                distance_from_virtual_line(
                    com_position_4, foot_positions_2d[:, 0], foot_positions_2d[:, 1]
                ),
                distance_from_virtual_line(
                    com_position_4, foot_positions_2d[:, 1], foot_positions_2d[:, 2]
                ),
                distance_from_virtual_line(
                    com_position_4, foot_positions_2d[:, 2], foot_positions_2d[:, 3]
                ),
                distance_from_virtual_line(
                    com_position_4, foot_positions_2d[:, 3], foot_positions_2d[:, 0]
                ),
            ],
            dim=1,
        )  # (N, 4)
        radii[four_contacts] = torch.min(distances, dim=1).values

    return radii


def foot_com_distance(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Gets the cost associated with the feet being too far away

    projects everything to the same z level for R2 distance claculation

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs,) representing how far the feet are away.
    """
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices, :2]  # type: ignore
    com_position = env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)

    # Calculate distances from COM to each foot
    distances = torch.norm(foot_positions - com_position.unsqueeze(1), dim=2)
    # Sum distances for all feet
    distance = distances.sum(dim=1)
    return distance


def foot_hip_distance(env: ManagerBasedEnv) -> torch.Tensor:
    """Gets the cost associated with the feet being too far away from the hips

    projects everything to the same z level for R2 distance claculation

    technically we are using the thigh positions as a proxy for the hip positions
    the have what you would expect to be the hip positions

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: A float tensor of shape (num_envs,) representing how far the feet are away from the hips.
    """
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices, :2]  # type: ignore
    hip_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.thigh_indices, :2]  # type: ignore

    # get the scalar euclidean distance
    distances = torch.norm(foot_positions - hip_positions, dim=2)
    # sum distances for all feet
    distance = distances.sum(dim=1)
    return distance


# def control_velocity_alignment(
#     env: ManagerBasedEnv, control: torch.Tensor
# ) -> torch.Tensor:
#     """Cost based on the alignment of the commanded velocity and the actual velocity.

#     TODO: make this be the average over the motion period instead of just the final velocity

#     Args:
#         env (ManagerBasedEnv): The environment.
#         control (torch.Tensor): The control input, shape (num_envs, 3) where the last dimension is (vx, vy, omega).
#             control is taken to be in the base frame

#     Returns:
#         torch.Tensor: A float tensor of shape (num_envs,) representing the control velocity alignment cost.
#     """
#     linear_velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]  # (num_envs, 2)
#     angular_velocity = env.scene["robot"].data.root_ang_vel_b[:, 2]  # (num_envs,)
#     commanded_linear_velocity = control[:, :2]  # (num_envs, 2)
#     commanded_angular_velocity = control[:, 2]  # (num_envs,)
#     # Calculate the differences
#     linear_diff = torch.norm(
#         linear_velocity - commanded_linear_velocity, dim=1
#     )  # (num_envs,)
#     angular_diff = torch.abs(
#         angular_velocity - commanded_angular_velocity
#     )  # (num_envs,)
#     # Combine the differences into a single cost
#     cost = linear_diff + angular_diff
#     return cost
