# Compute foot swing trajectory with a bezier curve
# phase : How far along we are in the swing (0 to 1)
# swingTime : How long the swing should take (seconds)
import numpy as np
from gaitnet_mpc.mpc.math_utils.interplation import cubicBezier, cubicBezierFirstDerivative, cubicBezierSecondDerivative

class FootSwingTrajectory:
    def __init__(self):
        # vec3 (3,1)
        self._p0:np.ndarray = None # np.zeros((3,1), dtype=np.float32)
        self._pf:np.ndarray = None # np.zeros((3,1), dtype=np.float32)
        self._p:np.ndarray = np.zeros((3,1), dtype=np.float32)
        self._v:np.ndarray = np.zeros((3,1), dtype=np.float32)
        self._a:np.ndarray = np.zeros((3,1), dtype=np.float32)
        # float or int
        self._height = 0.0

    def setInitialPosition(self, p0:np.ndarray):
        """
        Set the starting location of the foot
        """
        self._p0 = p0.copy()
    
    def setFinalPosition(self, pf:np.ndarray):
        """
        Set the desired final position of the foot
        """
        self._pf = pf.copy()

    def setHeight(self, h:float):
        """
        Set the swing apex's clearance above the higher of the start and final positions
        """
        self._height = h

    def getPosition(self):
        """
        Get the foot position at the current point along the swing
        """
        return self._p
    
    def getVelocity(self):
        """
        Get the foot velocity at the current point along the swing
        """
        return self._v
    
    def getAcceleration(self):
        """
        Get the foot acceleration at the current point along the swing
        """
        return self._a

    def computeSwingTrajectoryBezier(self, phase:float, swingTime:float):
        self._p = cubicBezier(self._p0, self._pf, phase)
        self._v = cubicBezierFirstDerivative(self._p0, self._pf, phase) / swingTime
        self._a = cubicBezierSecondDerivative(self._p0, self._pf, phase) / (swingTime * swingTime)

        # the apex clears the higher end, so a step up onto a raised foothold comes down
        # onto it rather than approaching it from below
        apex = max(self._p0[2], self._pf[2]) + self._height
        if phase < 0.5:
            zp = cubicBezier(self._p0[2], apex, phase * 2)
            zv = cubicBezierFirstDerivative(self._p0[2], apex, phase * 2) * 2 / swingTime
            za = cubicBezierSecondDerivative(self._p0[2], apex, phase * 2) * 4 / (swingTime * swingTime)
        else:
            zp = cubicBezier(apex, self._pf[2], phase * 2 - 1)
            zv = cubicBezierFirstDerivative(apex, self._pf[2], phase * 2 - 1) * 2 / swingTime
            za = cubicBezierSecondDerivative(apex, self._pf[2], phase * 2 - 1) * 4 / (swingTime * swingTime)

        self._p[2] = zp
        self._v[2] = zv
        self._a[2] = za
