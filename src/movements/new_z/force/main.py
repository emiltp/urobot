"""
New Z Force - Full force mode with speedL control loop.

Uses force mode for Fx=0, Fy=0, Fz=0 compliance (all three translational axes)
combined with continuous speedL commands to orbit the TCP around the Ref frame's
z-axis. The safety limit is Mz (moment around TCP z-axis).
"""

import time
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils import axis_angle_to_rotation_matrix, transform_wrench, rotation_matrix_to_axis_angle
from config import defaults as CONFIG


def execute(self):
    """Execute new-z movement using force mode with speedL control loop."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.new_z.speed)
    accel = self.kwargs.get('accel', CONFIG.new_z.acceleration)
    momentLimitZ = self.kwargs.get('moment_limit_z', CONFIG.new_z.max_moment)
    pathFile = self.kwargs.get('path_file', None)

    z_cfg = CONFIG.new_z_force.z_limit
    xy_cfg = CONFIG.new_z_force.xy_limit
    forceModeXYLimit = self.kwargs.get('force_mode_xy_limit', xy_cfg.force_mode_xy_limit)
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', z_cfg.force_mode_z_limit)
    forceModeZDamping = self.kwargs.get('force_mode_z_damping', z_cfg.force_mode_damping)
    forceModeXYDamping = self.kwargs.get('force_mode_xy_damping', xy_cfg.force_mode_damping)
    forceModeDamping = (forceModeZDamping + forceModeXYDamping) / 2
    forceModeZGain = self.kwargs.get('force_mode_z_gain_scaling', z_cfg.force_mode_gain_scaling)
    forceModeXYGain = self.kwargs.get('force_mode_xy_gain_scaling', xy_cfg.force_mode_gain_scaling)
    forceModeGainScaling = (forceModeZGain + forceModeXYGain) / 2
    controlLoopDt = self.kwargs.get('control_loop_dt', z_cfg.control_loop_dt)
    rotationSpeedFactor = self.kwargs.get('rotation_speed_factor', z_cfg.rotation_speed_factor)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-z movement (force mode + speedL)...")

    try:
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)

        self.waypointCollector.collect(startPosition)

        selection_vector = [1, 1, 1, 0, 0, 0]
        target_wrench = [0, 0, 0, 0, 0, 0]
        force_type = 2
        limits = [forceModeXYLimit, forceModeXYLimit, forceModeZLimit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(forceModeDamping)
        self.rtde_c.forceModeSetGainScaling(forceModeGainScaling)

        self.movement_progress.emit("Force mode enabled: Fx/Fy/Fz compliance active")

        targetPos = np.array(newPose[:3])
        targetOrientation = np.array(newPose[3:6])
        startPos = np.array(startPosition[:3])
        startOrientation = np.array(startPosition[3:6])

        initialPosError = np.linalg.norm(targetPos - startPos)
        initialRotError = _calculate_rotation_error(startOrientation, targetOrientation)
        initialTotalError = initialPosError + initialRotError

        angularSpeed = speed * rotationSpeedFactor

        self.movement_progress.emit(
            f"Moving toward target at speed {speed:.4f} m/s, "
            f"angular speed {angularSpeed:.3f} rad/s..."
        )

        currentPose = list(self.rtde_r.getActualTCPPose())
        lastWaypointTime = time.time()
        waypointInterval = 0.05

        self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

        while not self._stop_requested:
            t_start = self.rtde_c.initPeriod()

            currentPose = list(self.rtde_r.getActualTCPPose())
            currentPos = np.array(currentPose[:3])
            currentOrientation = np.array(currentPose[3:6])

            posError = np.linalg.norm(targetPos - currentPos)
            rotError = _calculate_rotation_error(currentOrientation, targetOrientation)

            if posError < 0.001 and rotError < 0.01:
                self.movement_progress.emit("Target pose reached!")
                break

            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpForce = transform_wrench(currentPose, tcpWrenchInBase)
            forceXTcp = tcpForce[0]
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]
            momentZTcp = tcpForce[5]

            if direction == 'left':
                limitDisplay = f"limit: > -{momentLimitZ:.2f} Nm"
            elif direction == 'right':
                limitDisplay = f"limit: < {momentLimitZ:.2f} Nm"
            else:
                limitDisplay = f"limit: ±{momentLimitZ:.2f} Nm"

            momentExceeded = False
            if direction == 'left':
                if momentZTcp < -momentLimitZ:
                    momentExceeded = True
            elif direction == 'right':
                if momentZTcp > momentLimitZ:
                    momentExceeded = True
            else:
                if abs(momentZTcp) > momentLimitZ:
                    momentExceeded = True

            if momentExceeded:
                self.movement_progress.emit(
                    f"Moment threshold exceeded! Mz: {momentZTcp:.3f} Nm ({limitDisplay}). Stopping."
                )
                break

            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            posDirection = targetPos - currentPos
            posNorm = np.linalg.norm(posDirection)
            if posNorm > 1e-6:
                posDirection = posDirection / posNorm
            else:
                posDirection = np.zeros(3)

            rotDirection = _get_rotation_direction(currentOrientation, targetOrientation)

            totalError = posError + rotError
            speedScale = min(1.0, totalError / (0.3 * (initialTotalError if initialTotalError > 0 else 1.0)))
            speedScale = max(speedScale, 0.05)

            linearVelocity = posDirection * speed * speedScale
            angularVelocity = rotDirection * angularSpeed * speedScale

            speedVector = [
                linearVelocity[0], linearVelocity[1], linearVelocity[2],
                angularVelocity[0], angularVelocity[1], angularVelocity[2]
            ]
            self.rtde_c.speedL(speedVector, accel, controlLoopDt)

            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progressPct = (1.0 - totalError / initialTotalError) * 100 if initialTotalError > 0 else 100
                self.movement_progress.emit(
                    f"Progress: {progressPct:.1f}% | Mz: {momentZTcp:.3f} Nm ({limitDisplay}) | "
                    f"Fx: {forceXTcp:.3f} N, Fy: {forceYTcp:.3f} N (force-controlled) | "
                    f"Fz: {forceZTcp:.3f} N (force-controlled)"
                )

            self.rtde_c.waitPeriod(t_start)

        self.rtde_c.speedStop()
        self.rtde_c.forceModeStop()

        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Movement completed.")

        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")

    except Exception as e:
        try:
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during new-z force movement: {str(e)}")
        raise


def _calculate_rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    """Calculate the rotation error between two axis-angle representations."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    R_error = R_target @ R_current.T
    r = Rotation.from_matrix(R_error)
    return np.linalg.norm(r.as_rotvec())


def _get_rotation_direction(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the geodesic rotation direction from current to target orientation.
    Uses R_target @ R_current.T to get the shortest-path rotation axis, avoiding
    axis-angle subtraction which can point the wrong way near π wraparound."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    R_error = R_target @ R_current.T
    rotvec = Rotation.from_matrix(R_error).as_rotvec()
    norm = np.linalg.norm(rotvec)
    if norm < 1e-6:
        return np.zeros(3)
    return rotvec / norm
