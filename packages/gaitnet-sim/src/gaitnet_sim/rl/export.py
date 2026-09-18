"""A trained run -> a `gaitnet_core` policy bundle.

A run directory, as Isaac Lab's train entry point writes it, has `params/env.yaml`,
`params/agent.yaml` and RSL-RL checkpoints `model_<iteration>.pt`. The bundle takes the
scoring network and its learned duration noise from a checkpoint, and the robot, foothold
grid and rules, state features and training sampler from the env cfg. MLflow runs hold the
same files as artifacts (see `MlflowLogWriter`) and are downloaded into that layout first.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import torch
import yaml

from gaitnet_core.bundle import BundleError, PolicyBundle
from gaitnet_core.features import feature_dim
from gaitnet_core.networks import build_network
from gaitnet_sim.env.contract import GaitNetCfg

_CHECKPOINT = re.compile(r"model_(\d+)\.pt")


class _ParamsLoader(yaml.SafeLoader):
    """Isaac Lab dumps cfgs with plain `yaml.dump`, which tags Python types
    (`!!python/tuple`, `!!python/object/apply:builtins.slice`, ...). Tuples are rebuilt;
    anything else becomes a descriptive string, since nothing read from it here is one."""


def _python_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node):
    if suffix == "tuple":
        return tuple(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
    else:
        value = loader.construct_scalar(node)
    return f"<{suffix} {value}>"


_ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


def load_params(path: str | Path) -> dict:
    """A cfg Isaac Lab dumped to a run's `params/`."""
    with open(path) as f:
        return yaml.load(f, Loader=_ParamsLoader)


def latest_checkpoint(names: list[str]) -> str:
    """The highest-iteration `model_<i>.pt` among `names`."""
    numbered = [(int(match.group(1)), name) for name in names if (match := _CHECKPOINT.fullmatch(name))]
    if not numbered:
        raise FileNotFoundError(f"no model_<iteration>.pt checkpoint among {sorted(names)}")
    return max(numbered)[1]


def bundle_from_run(run_dir: str | Path, checkpoint: str | None = None, extra: dict | None = None) -> PolicyBundle:
    """Build a bundle from a local run directory.

    Args:
        checkpoint: file name in `run_dir`, the latest `model_<i>.pt` if None
        extra: added to the bundle's free-form metadata
    """
    run_dir = Path(run_dir)
    env = load_params(run_dir / "params" / "env.yaml")
    agent = load_params(run_dir / "params" / "agent.yaml")
    if checkpoint is None:
        checkpoint = latest_checkpoint([path.name for path in run_dir.iterdir()])

    contract = GaitNetCfg(**env["gaitnet"])
    spec = contract.robot_spec()
    features = tuple(env["observations"]["state"]["robot_state"]["params"]["features"])
    # the planner builds the network's state from the features alone
    actor_groups = list(agent["obs_groups"]["actor"])
    if actor_groups != ["state"]:
        raise BundleError(f"the actor read observation groups {actor_groups}; a bundle's planner provides only 'state'")
    candidates = env["observations"]["candidates"]["candidates"]["params"]
    train_sampler = {"name": candidates["sampler"], **(candidates.get("sampler_kwargs") or {})}

    network_cfg = dict(agent["actor"]["network"])
    network_class = network_cfg.pop("class_name")
    network_cfg.setdefault("state_dim", feature_dim(features, spec.num_legs))
    network = build_network(network_class, network_cfg)

    saved = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    actor_state = saved["actor_state_dict"]
    network.load_state_dict({key.removeprefix("network."): value for key, value in actor_state.items() if key.startswith("network.")})
    duration_std = float(actor_state["duration_log_std"].exp())

    controller = env.get("actions", {}).get("footstep", {}).get("controller", {}).get("class_type")
    return PolicyBundle(
        actor=network.eval(),
        robot=spec,
        grid=contract.foothold_grid(),
        features=features,
        rules=contract.foothold_rules(),
        train_sampler=train_sampler,
        duration_std=duration_std,
        extra={
            "run_dir": str(run_dir),
            "checkpoint": checkpoint,
            "iteration": saved.get("iter"),
            "controller": str(controller),
            **(extra or {}),
        },
    )


def bundle_from_mlflow(run_id: str, checkpoint: str | None = None, tracking_uri: str | None = None) -> PolicyBundle:
    """Build a bundle from an MLflow run's artifacts (`params/`, `checkpoints/`)."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    if checkpoint is None:
        checkpoint = latest_checkpoint([Path(item.path).name for item in client.list_artifacts(run_id, "checkpoints")])
    run = client.get_run(run_id)
    with tempfile.TemporaryDirectory() as tmp:
        client.download_artifacts(run_id, "params", tmp)
        # a single file lands directly in the destination, without its artifact directory
        local = Path(client.download_artifacts(run_id, f"checkpoints/{checkpoint}", tmp))
        if local != Path(tmp, checkpoint):
            local.rename(Path(tmp, checkpoint))
        bundle = bundle_from_run(tmp, checkpoint, extra={"mlflow_run_id": run_id})
    bundle.extra["run_dir"] = run.data.tags.get("log_dir", "")
    if "git_commit" in run.data.tags:
        bundle.extra["git_commit"] = run.data.tags["git_commit"]
    return bundle
