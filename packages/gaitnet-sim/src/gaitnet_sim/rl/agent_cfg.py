"""RSL-RL runner configs for the GaitNet tasks.

PPO settings carry over from the Isaac Lab 2 training script. The actor is
`gaitnet_sim.rl.model.GaitNetActor` over a `gaitnet_core.networks` scoring network; the
critic is RSL-RL's own MLP. Metrics go to MLflow (`MlflowLogWriter`) and to a TensorBoard
log in the run directory.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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


@configclass
class GaitNetActorCfg:
    """Keyword arguments of `GaitNetActor`, see there."""

    class_name: str = "gaitnet_sim.rl.model:GaitNetActor"
    network: dict = {
        "class_name": "CandidateScorer",
        "candidate_features": "xyz",
        "shared_sizes": [128, 128, 128],
        "candidate_sizes": [64, 64],
        "trunk_sizes": [128, 128, 128],
    }
    candidates_group: str = "candidates"
    terrain_group: str | None = None
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
    obs_groups = {"actor": ["state"], "critic": ["state"]}
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
