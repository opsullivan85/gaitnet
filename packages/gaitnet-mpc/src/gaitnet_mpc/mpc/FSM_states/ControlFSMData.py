from gaitnet_mpc.mpc.common.Quadruped import Quadruped
from gaitnet_mpc.mpc.common.LegController import LegController
# from gaitnet.control.MPC_Controller.Parameters import Parameters
from gaitnet_mpc.mpc.common.StateEstimator import StateEstimator
from gaitnet_mpc.mpc.common.DesiredStateCommand import DesiredStateCommand

class ControlFSMData:
    def __init__(self):
        self._quadruped:Quadruped = None
        self._stateEstimator:StateEstimator = None
        self._legController:LegController = None
        self._desiredStateCommand:DesiredStateCommand = None
        # self._gaitScheduler:GaitScheduler = None
        # self.userParameters:Parameters = None

