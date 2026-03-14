"""
Axial Rotation Force - Full force mode with speedL control loop.

Uses force mode for full translational compliance (Fx=Fy=Fz=0) combined with
continuous speedL commands for rotation control around TCP z-axis. All three
positional axes float freely, providing maximum reactivity and smooth motion.
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils import axis_angle_to_rotation_matrix, transform_wrench
from config import defaults as CONFIG


def execute(self):
    """Execute axial rotation with full translational compliance (force mode + speedL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.rotation.speed)
    accel = self.kwargs.get('accel', CONFIG.rotation.acceleration)
    maxMoment = self.kwargs.get('max_moment', CONFIG.rotation.max_moment)

    # Force mode parameters
    forceModeXYLimit = self.kwargs.get('force_mode_xy_limit', CONFIG.rotation_force.force_mode_xy_limit)
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.rotation_force.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.rotation_force.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.rotation_force.force_mode_gain_scaling)
    controlLoopDt = self.kwargs.get('control_loop_dt', CONFIG.rotation_force.control_loop_dt)
    rotationSpeedFactor = self.kwargs.get('rotation_speed_factor', CONFIG.rotation_force.rotation_speed_factor)

    self.movement_started.emit()
    self.movement_progress.emit("Starting axial rotation (force: full compliance + speedL)...")

    try:
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)

        self.waypointCollector.collect(startPosition)

        # =====================================================
        # FORCE MODE SETUP — full translational compliance
        # =====================================================
        selection_vector = [1, 1, 1, 0, 0, 0]  # Fx, Fy, Fz force-controlled
        target_wrench = [0, 0, 0, 0, 0, 0]     # Target all forces = 0
        force_type = 2  # Frame not transformed
        limits = [forceModeXYLimit, forceModeXYLimit, forceModeZLimit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(forceModeDamping)
        self.rtde_c.forceModeSetGainScaling(forceModeGainScaling)

        self.movement_progress.emit(
            f"Force mode enabled: full compliance (XY limit: {forceModeXYLimit} m/s, Z limit: {forceModeZLimit} m/s)"
        )

        # =====================================================
        # CALCULATE TARGET ROTATION
        # =====================================================
        startOrientation = np.array(startPosition[3:6])
        targetOrientation = np.array(newPose[3:6])
        initialRotationError = _calculate_rotation_error(startOrientation, targetOrientation)

        angularSpeed = speed * rotationSpeedFactor  # rad/s

        self.movement_progress.emit(f"Rotating toward target at {angularSpeed:.3f} rad/s...")

        # =====================================================
        # CONTROL LOOP
        # =====================================================
        currentPose = list(self.rtde_r.getActualTCPPose())
        lastWaypointTime = time.time()
        waypointInterval = 0.05

        self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

        while not self._stop_requested:
            t_start = self.rtde_c.initPeriod()

            currentPose = list(self.rtde_r.getActualTCPPose())
            currentOrientation = np.array(currentPose[3:6])

            rotationError = _calculate_rotation_error(currentOrientation, targetOrientation)

            if rotationError < 0.01:
                self.movement_progress.emit("Target rotation reached!")
                break

            # =====================================================
            # FORCE / TORQUE MONITORING
            # =====================================================
            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpForce = transform_wrench(currentPose, tcpWrenchInBase)
            torqueMagnitudeZ = abs(tcpForce[5])
            forceXTcp = tcpForce[0]
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]

            if torqueMagnitudeZ > maxMoment:
                self.movement_progress.emit(
                    f"Contact detected! Torque {torqueMagnitudeZ:.2f} Nm exceeds limit {maxMoment:.2f} Nm. Stopping."
                )
                break

            # =====================================================
            # UPDATE TASK FRAME
            # =====================================================
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            # =====================================================
            # COMPUTE SPEED COMMAND
            # =====================================================
            rotationDirection = _normalize_rotation_vector(targetOrientation - currentOrientation)

            speedScale = min(1.0, rotationError / 0.3)
            angularVelocity = rotationDirection * angularSpeed * speedScale

            speedVector = [0, 0, 0, angularVelocity[0], angularVelocity[1], angularVelocity[2]]
            self.rtde_c.speedL(speedVector, accel, controlLoopDt)

            # =====================================================
            # WAYPOINT COLLECTION
            # =====================================================
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progressPct = (1.0 - rotationError / initialRotationError) * 100 if initialRotationError > 0 else 100
                self.movement_progress.emit(
                    f"Progress: {progressPct:.1f}% | Mz: {torqueMagnitudeZ:.2f} Nm (limit: {maxMoment:.2f}) | "
                    f"Fx: {forceXTcp:.3f} Fy: {forceYTcp:.3f} Fz: {forceZTcp:.3f} N (all force-controlled)"
                )

            self.rtde_c.waitPeriod(t_start)

        # =====================================================
        # CLEANUP
        # =====================================================
        self.rtde_c.speedStop()
        self.rtde_c.forceModeStop()

        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Movement completed with full translational compliance.")

        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(
            f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]"
        )

    except Exception as e:
        try:
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during rotation force movement: {str(e)}")
        raise


def _calculate_rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    """Calculate the rotation error between two axis-angle representations."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])

    R_error = R_target @ R_current.T

    r = Rotation.from_matrix(R_error)
    angle = np.linalg.norm(r.as_rotvec())

    return angle


def _normalize_rotation_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a rotation vector, handling zero case."""
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros(3)
    return vec / norm
