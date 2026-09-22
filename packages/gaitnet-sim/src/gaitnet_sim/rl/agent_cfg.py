"""RSL-RL runner configs for the GaitNet tasks.

PPO settings carry over from the Isaac Lab 2 training script. The actor is
`gaitnet_sim.rl.model.GaitNetActor` over a `gaitnet_core.networks` scoring network; the
critic is RSL-RL's own MLP. Metrics go to MLflow (`MlflowLogWriter`) and to a TensorBoard
log in the run directory.

Presets (`presets=<name>[,<name>...]`), each matched by the env cfg's observation groups:
- spatial: the dense spatial CNN (D1) actor, reading the terrain group
- crop: the candidate scorer with the local-crop encoder, reading the terrain group
- privileged: the critic also reads the privileged group
- slowdown: the actor runs the step-confidence slowdown observer, reading base_command
- redirect: the actor runs the blocked-leg redirect observer, reading base_command and footholds
- slowdown_redirect: both observers, their nudges summed
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.utils import preset

from gaitnet_sim.env.contract import GaitNetCfg


class WriterCfg(dict):
    """An RSL-RL log writer, `{"class_name": "module.Class", **kwargs}`.

    Hashable because Isaac Lab 3.0's train entry point checks
    `agent_cfg.logger in {"wandb", "neptune"}`, assuming the plain string form, while RSL-RL
    needs this dict form for any writer of its own. `to_dict()` turns it back into a dict.

    Name the class `module.Class`, not `module:Class`: configclass wraps "module:Class"
    strings for lazy import and rebuilds the dict around them as a plain, unhashable one.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)


_CONTRACT = GaitNetCfg()
FOOTHOLD_GRID = {
    "resolution": _CONTRACT.grid_resolution,
    "size": list(_CONTRACT.grid_size),
    "border": _CONTRACT.grid_border,
}
"""The env's default foothold grid, which networks that read terrain are built for. A run
with a different grid needs e.g. `agent.actor.network.grid.resolution=...` too; export
refuses a mismatch."""

CANDIDATE_SCORER = {
    "class_name": "CandidateScorer",
    "candidate_features": "xyz",
    "shared_sizes": [128, 128, 128],
    "candidate_sizes": [64, 64],
    "trunk_sizes": [128, 128, 128],
}
FIXED_SWING_DURATION = 0.25
"""Swing duration (s) of the `swing_duration_ablation` preset, overridable as
`agent.actor.network.fixed_duration=0.3`."""
CROP_SCORER ={**CANDIDATE_SCORER, "candidate_features": "xyz_crop", "grid": FOOTHOLD_GRID, "crop_radius": 2}
DENSE_SPATIAL_CNN = {
    "class_name": "DenseSpatialCNN",
    "grid": FOOTHOLD_GRID,
    "channels": [16, 16, 16],
    "state_sizes": [128, 64],
    "noop_sizes": [64],
}
SLOWDOWN_OBSERVERS = {"step_confidence_slowdown": {"patience": 10, "margin": 0.0, "scale": 0.5}}
REDIRECT_OBSERVERS = {"blocked_leg_redirect": {"full_fraction": 1.0, "push": 0.0}}


@configclass
class GaitNetActorCfg:
    """Keyword arguments of `GaitNetActor`, see there."""

    class_name: str = "gaitnet_sim.rl.model:GaitNetActor"
    network = preset(
        default=CANDIDATE_SCORER,
        spatial=DENSE_SPATIAL_CNN,
        crop=CROP_SCORER,
        swing_duration_ablation={**CANDIDATE_SCORER, "fixed_duration": FIXED_SWING_DURATION},
    )
    candidates_group: str = "candidates"
    terrain_group: str = "terrain"
    observers = preset(
        default={},
        slowdown=SLOWDOWN_OBSERVERS,
        redirect=REDIRECT_OBSERVERS,
        slowdown_redirect={**SLOWDOWN_OBSERVERS, **REDIRECT_OBSERVERS},
    )
    base_command_group: str = "base_command"
    footholds_group: str = "footholds"
    duration_std: float = 0.05
    # Isaac Lab's cfg handling reads these on every model cfg; the actor's distribution is fixed
    distribution_cfg: None = None
    stochastic: bool = False


@configclass
class GaitNetPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 250  # half an episode per rollout
    max_iterations = 10000
    save_interval = 5
    experiment_name = "gaitnet_holes"
    obs_groups = {"actor": ["state"], "critic": preset(default=["state"], privileged=["state", "privileged"])}
    actor = GaitNetActorCfg()
    critic = RslRlMLPModelCfg(hidden_dims=[64] * 6, activation="relu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.3,
        entropy_coef=0.02,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,  # only used by schedule="adaptive"
        max_grad_norm=1.0,
    )
    logger = WriterCfg(class_name="gaitnet_sim.rl.mlflow_writer.MlflowLogWriter", experiment_name="gaitnet")


@configclass
class GaitNetPillarsPpoRunnerCfg(GaitNetPpoRunnerCfg):
    experiment_name = "gaitnet_pillars"
