
from gaitnet.util import log_exceptions

from gaitnet import get_logger
logger = get_logger()


@log_exceptions(logger)
def main():
    print("Hello, RobotInterface!")
    import numpy as np

    from gaitnet.sim2real import SimInterface

    sim_interface = SimInterface(dt=0.01, debug_logging=True)

    joint_states = np.zeros(shape=(4, 3, 2), dtype=np.float32)
    body_state = np.zeros(shape=(13,), dtype=np.float32)
    command = np.zeros(shape=(3,), dtype=np.float32)
    sim_interface.get_torques(
        joint_states=joint_states, body_state=body_state, command=command
    )
    sim_interface.get_torques(
        joint_states=joint_states, body_state=body_state, command=command
    )
    sim_interface.get_torques(
        joint_states=joint_states, body_state=body_state, command=command
    )


if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        main()
