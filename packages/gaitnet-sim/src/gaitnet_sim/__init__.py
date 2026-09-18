"""GaitNet on Isaac Lab.

A vanilla manager-based environment: candidates come from an observation term that calls
`gaitnet_core.samplers`, and one action term (`env.actions.FootstepControlAction`) owns the
low-level controller, executes footsteps and applies the planner's nudge. Nothing here
subclasses an Isaac Lab manager.

Importing this package does not start Isaac Sim; the modules that need a running app
import Isaac Lab themselves.
"""
