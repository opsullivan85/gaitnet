import numpy as np
from abc import ABC, abstractmethod


class GaitABC(ABC):
    """
    A gait pattern generator for quadruped robots using offset-duration parameterization.

    Leg indexing convention: [FL, FR, RL, RR] (Front Left, Front Right, Rear Left, Rear Right)

    The class provides real-time gait state information including:
    - Contact states (0-1 values indicating stance phase progress)
    - Swing states (0-1 values indicating swing phase progress)
    - MPC table (binary contact schedule for the prediction horizon)
    - Timing information for footstep planning and trajectory generation
    """

    @abstractmethod
    def setIterations(self, iterationsPerMPC: int, currentIteration: int) -> None:
        """Update the current gait timing based on the global iteration counter.

        Args:
            iterationsPerMPC (int): Number of control iterations between MPC updates.
            currentIteration (int): Global iteration counter since system start.
        """
        ...

    @abstractmethod
    def getContactPhase(self) -> np.ndarray:
        """Calculate the current stance phase progress for each leg.

        Returns:
            np.ndarray: Stance progress for each leg, shape (4, 1).
                       Values range [0, 1] where:
                       - 0.0: Beginning of stance phase (foot just made ground contact)
                       - 0.5: Middle of stance phase (maximum ground reaction force)
                       - 1.0: End of stance phase (about to lift off)
                       - 0.0: Leg is in swing phase (no ground contact)
        """
        ...

    @abstractmethod
    def getSwingPhase(self) -> np.ndarray:
        """Calculate the current swing phase progress for each leg.

        Returns:
            np.ndarray: Swing progress for each leg, shape (4, 1).
                       Values range [0, 1] where:
                       - 0.0: Beginning of swing phase (foot just lifted off)
                       - 0.5: Middle of swing phase (maximum foot height/velocity)
                       - 1.0: End of swing phase (about to touch down)
                       - 0.0: Leg is in stance phase (foot on ground)
        """
        ...

    @abstractmethod
    def getMpcTable(self) -> list[int]:
        """Generate binary contact schedule for the MPC prediction horizon.

        Returns:
            list: Binary contact states for the prediction horizon.
                 Length = nSegment * 4, arranged as [leg0_t0, leg1_t0, leg2_t0, leg3_t0,
                                                      leg0_t1, leg1_t1, leg2_t1, leg3_t1, ...]
        """
        ...

    @abstractmethod
    def getCurrentSwingTime(self, dtMPC: float, leg: int) -> float:
        """Calculate the total swing duration for any leg in the gait.

        Args:
            dtMPC (float): MPC time step duration in seconds.
            leg (int): Leg index [0-3] for [FL, FR, RL, RR].

        Returns:
            float: Total swing phase duration in seconds.
        """
        ...

    @abstractmethod
    def getCurrentStanceTime(self, dtMPC: float, leg: int) -> float:
        """Calculate the total stance duration for any leg in the gait.

        Args:
            dtMPC (float): MPC time step duration in seconds.
            leg (int): Leg index [0-3] for [FL, FR, RL, RR].

        Returns:
            float: Total stance phase duration in seconds.
        """
        ...


class OffsetDurationGait(GaitABC):
    def __init__(
        self, nSegment: int, offset: np.ndarray, durations: np.ndarray, name: str
    ):
        """Initialize a gait pattern with offset-duration parameterization.

        Args:
            nSegment (int): Total number of MPC segments in one gait cycle. Typically 30 (10*3)
                           where 10 is the MPC horizon and 3 provides finer temporal resolution.
            offset (np.ndarray): Start time of stance phase for each leg in MPC segments.
                                Shape (4,) for [FL, FR, RL, RR] legs. Values should be in range [0, nSegment-1].
            durations (np.ndarray): Duration of stance phase for each leg in MPC segments.
                                   Shape (4,) for [FL, FR, RL, RR] legs. Values should be in range [1, nSegment].
                                   Higher values = longer ground contact = more stable but slower gaits.
            name (str): Human-readable name for the gait pattern (e.g., "Trotting", "Galloping").
        """
        # Implementation Details:
        #     - Swing duration = nSegment - stance duration for each leg
        #     - Phase values are normalized to [0, 1] for internal calculations
        #     - MPC table is pre-allocated as a flat array of size nSegment*4
        #     - Assumes first leg (FL) determines overall stance/swing timing for the gait

        # offset in mpc segments
        self._offsets = offset.flatten()
        # duration of step in mpc segments
        self._durations = durations.flatten()
        # offsets in phase (0 to 1)
        self._offsetsFloat = offset / nSegment
        # durations in phase (0 to 1)
        self._durationsFloat = durations / nSegment
        self._nIterations = nSegment
        self._name = name
        self._stance = durations[0]
        self._swing = nSegment - durations[0]
        self._mpc_table = [0 for _ in range(nSegment * 4)]

    def setIterations(self, iterationsPerMPC: int, currentIteration: int) -> None:
        # Implementation Details:
        #     - __iteration: Current discrete gait segment [0, nSegment-1]
        #     - __phase: Continuous gait phase [0, 1] where 0=start of cycle, 1=end of cycle
        #     - Phase calculation ensures smooth transitions and proper gait timing
        #     - Used by getContactState(), getSwingState(), and getMpcTable()

        self._iteration = (currentIteration / iterationsPerMPC) % self._nIterations
        self._phase = float(
            currentIteration % (iterationsPerMPC * self._nIterations)
        ) / float(iterationsPerMPC * self._nIterations)

    def getContactPhase(self) -> np.ndarray:
        # Implementation Details:
        #     - Progress calculated as (current_phase - offset_phase) / duration_phase
        #     - Handles phase wrapping when offset occurs late in gait cycle
        #     - Returns 0.0 when leg is outside its designated stance window
        #     - Used by ConvexMPCLocomotion.run() for state_estimator.setContactPhase()

        progress = self._phase - self._offsetsFloat

        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1.0

            if progress[i] > self._durationsFloat[i]:
                progress[i] = 0.0
            else:
                progress[i] = progress[i] / self._durationsFloat[i]

        return progress[:, None]  # convert to matrix

    def getSwingPhase(self) -> np.ndarray:
        # Implementation Details:
        #     - Swing starts when stance phase ends: swing_offset = offset + duration
        #     - Swing duration = 1.0 - stance_duration (in normalized phase)
        #     - Handles phase wrapping and zero-duration swings (pure stance gaits)
        #     - Used by ConvexMPCLocomotion.run() to drive FootSwingTrajectory.computeSwingTrajectoryBezier()

        swing_offset = self._offsetsFloat + self._durationsFloat
        for i in range(4):
            if swing_offset[i] > 1:
                swing_offset[i] -= 1.0
        swing_duration = np.ones_like(self._durationsFloat) - self._durationsFloat

        progress = self._phase - swing_offset

        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1.0

            if progress[i] > swing_duration[i]:
                progress[i] = 0.0
            else:
                if swing_duration[i] == 0.0:
                    progress[i] = 0.0
                else:
                    progress[i] = progress[i] / swing_duration[i]

        return progress[:, None]

    def getMpcTable(self) -> list[int]:
        # Implementation Details:
        #     - Predicts contact states from current time to horizon end
        #     - Uses discrete segment-based timing (not continuous phase)
        #     - Accounts for current gait phase when generating future states
        #     - Fed directly to ConvexMpc.compute_contact_forces() for optimization
        #     - Critical for MPC to plan appropriate ground reaction forces

        # Usage in MPC:
        #     - MPC uses this table to determine which legs can generate forces
        #     - Enables predictive control of contact transitions
        #     - Allows optimization of forces only when feet are expected to be in contact

        for i in range(self._nIterations):

            iter = (i + self._iteration + 1) % self._nIterations
            progress = iter - self._offsets
            for j in range(4):
                if progress[j] < 0:
                    progress[j] += self._nIterations
                if progress[j] < self._durations[j]:
                    self._mpc_table[i * 4 + j] = 1
                else:
                    self._mpc_table[i * 4 + j] = 0

        return self._mpc_table

    def getCurrentSwingTime(self, dtMPC: float, leg: int) -> float:
        # Note:
        #     Current implementation assumes all legs have the same swing duration,
        #     using only the first leg's stance duration. For asymmetric gaits,
        #     this could be modified to use leg-specific durations.

        return dtMPC * self._swing

    def getCurrentStanceTime(self, dtMPC: float, leg: int) -> float:
        return dtMPC * self._stance


class CalculatedGait(GaitABC):
    def __init__(
        self, controller_dt: float, iterations_per_mpc: int, mpc_horizon: int
    ) -> None:
        self.swing_start_times = np.zeros((4, 1), dtype=np.float32)
        """Start time of each active swing phase"""
        self.swing_durations = np.full((4, 1), 0.0, dtype=np.float32)
        """Total duration of each active swing phase"""
        self.controller_dt = controller_dt
        """Time step per iteration"""
        self.iterations_per_mpc: int = iterations_per_mpc
        """Number of iterations per MPC cycle"""
        self.mpc_horizon: int = mpc_horizon
        """MPC horizon"""
        self.mpc_dt = self.controller_dt * self.iterations_per_mpc
        self.time = 0.0
        """Time since initialization"""

    def initiate_footstep(
        self, leg: int, duration: float
    ) -> None:
        """Initiates a footstep for the specified leg.

        Args:
            leg (int): Index of the leg (0-3)
            duration (float): Duration of the footstep
        """
        self.swing_start_times[leg, 0] = self.time
        self.swing_durations[leg, 0] = duration

    def _contact_state(self, time: float) -> np.ndarray:
        """Gets the expected contact state at a specified time

        Returns:
            np.ndarray: The expected contact state for each leg
        """
        contact_state = self.swing_start_times + self.swing_durations <= time
        return contact_state.astype(np.int32)

    # override
    def setIterations(self, iterationsPerMPC: int, currentIteration: int) -> None:
        self.time = currentIteration * self.controller_dt

    # override
    def getContactPhase(self) -> np.ndarray:
        # We are just returning either a 1 or zero here since we
        # can't know when the feet will next be picked up
        # TODO: maybe we could assume the contact lengths will be simmilar to the previous?
        return self._contact_state(self.time).astype(np.float32)

    # override
    def getSwingPhase(self) -> np.ndarray:
        if np.all(self.swing_durations == 0):
            return np.zeros((4, 1), dtype=np.float32)

        # Compute swing phase, avoiding division by zero via masking
        diff = self.time - self.swing_start_times
        swing_phase = np.zeros_like(diff, dtype=np.float32)  # Initialize with zeros
        mask = self.swing_durations.flatten() != 0  # Flatten for 1D boolean indexing
        swing_phase[mask, 0] = (diff[mask, 0] / self.swing_durations[mask, 0])
        
        # Clamp to valid range: set negative or >=1 values to 0
        swing_phase = np.clip(swing_phase, 0.0, 1.0)
        swing_phase[swing_phase >= 1.0] = 0.0  # Explicitly handle end-of-swing
        
        return swing_phase

    # override
    def getMpcTable(self) -> list[int]:
        mpc_table = []
        for i in range(self.mpc_horizon):
            # TODO: should this be i or i+1?
            time = self.time + i * self.mpc_dt
            mpc_table.extend(self._contact_state(time).flatten().tolist())
        return mpc_table

    # override
    def getCurrentSwingTime(self, dtMPC: float, leg: int) -> float:
        assert self.mpc_dt == dtMPC, "Unexpected MPC dt"
        return self.swing_durations[leg, 0]

    # override
    def getCurrentStanceTime(self, dtMPC: float, leg: int) -> float:
        # For CalculatedGait, we don't have predetermined stance times like OffsetDurationGait
        # Return a reasonable default stance duration or estimate based on current state
        # This is used for footstep placement heuristics in the parent class
        assert self.mpc_dt == dtMPC, "Unexpected MPC dt"
        
        # Return a default stance time (could be made configurable)
        # This is approximately the time the foot will remain on the ground
        default_stance_time = 0.3  # seconds
        return default_stance_time
