"""The MPC controller moved to gaitnet_mpc. This alias keeps the old sim layer working
until it is replaced (restructure plan P3)."""

from gaitnet_mpc.controller import MpcFootstepController as SimInterface
from gaitnet.sim2real.abstractinterface import Sim2RealInterface

__all__ = ["SimInterface", "Sim2RealInterface"]
