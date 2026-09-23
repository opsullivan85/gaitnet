"""Event terms of the GaitNet env that aren't Isaac Lab's own (`env_cfg.EventsCfg`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def pump_kit_for_livestream(env: "ManagerBasedEnv", env_ids: torch.Tensor | None) -> None:
    """Keep a livestreamed run's viewport, and so the stream, updating while the env steps.

    In Isaac Lab 3.0 only the Kit visualizer runs Kit's app loop, which is what draws the
    viewport and hands the stream its frames, and it skips that whenever the app is headless.
    A livestreamed app always is, so the client connects and then shows black (or the last
    frame from before the scene was built). This has every `render()` run the app loop
    instead, unless a visualizer already did. Does nothing without a GUI, i.e. headless runs.
    """
    sim = env.sim
    if not sim.has_gui:
        return

    import omni.kit.app
    from isaaclab.app.settings_manager import get_settings_manager

    app = omni.kit.app.get_app()
    settings = get_settings_manager()

    def pump(_) -> None:
        if any(getattr(viz, "_app_pumped_this_step", False) for viz in sim.visualizers):
            return
        # the same guard the Kit visualizer uses: physics belongs to SimulationContext, so
        # Kit mustn't advance it inside update()
        settings.set_bool("/app/player/playSimulations", False)
        app.update()
        settings.set_bool("/app/player/playSimulations", True)

    sim.add_render_callback("gaitnet_livestream_pump", pump)
