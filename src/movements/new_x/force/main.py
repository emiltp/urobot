"""
New X Force - Full force mode with speedL control loop.

Uses force mode for Fz=0 compliance combined with continuous speedL commands.
Unlike flexion_x_force which only commands angular velocity (rotation in place),
this computes the full 6-DOF velocity needed to orbit the TCP around the Ref
frame axis toward the target pose.
"""

import time
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
from config import defaults as CONFIG


def execute(self):
    """Execute new-x movement using force mode with speedL control loop."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    if startPosition is None:
        self.movement_progress.emit("Error: start_position is required")
        return
    if newPose is None:
        self.movement_progress.emit("Error: new_pose is required")
        return
    speed = self.kwargs.get('speed', CONFIG.new_x.speed)
    accel = self.kwargs.get('accel', CONFIG.new_x.acceleration)
    forceLimitY = self.kwargs.get('force_limit_y', CONFIG.new_x.force_limit_y)
    pathFile = self.kwargs.get('path_file', None)

    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.new_x_force.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.new_x_force.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.new_x_force.force_mode_gain_scaling)
    controlLoopDt = self.kwargs.get('control_loop_dt', CONFIG.new_x_force.control_loop_dt)
    rotationSpeedFactor = self.kwargs.get('rotation_speed_factor', CONFIG.new_x_force.rotation_speed_factor)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-x movement (force mode + speedL)...")

    try:
        self.rtde_c.zeroFtSensor()
        time.sleep(0.2)

        self.waypointCollector.collect(startPosition)

        selection_vector = [0, 0, 1, 0, 0, 0]
        target_wrench = [0, 0, 0, 0, 0, 0]
        force_type = 2
        limits = [0.1, 0.1, forceModeZLimit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(forceModeDamping)
        self.rtde_c.forceModeSetGainScaling(forceModeGainScaling)

        self.movement_progress.emit("Force mode enabled: Fz compliance active")

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

            # Force monitoring (wrench at TCP in TCP frame)
            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]

            if direction == 'left':
                limitDisplay = f"limit: > -{forceLimitY:.2f} N"
            elif direction == 'right':
                limitDisplay = f"limit: < {forceLimitY:.2f} N"
            else:
                limitDisplay = f"limit: ±{forceLimitY:.2f} N"

            forceExceeded = False
            if direction == 'left':
                if forceYTcp < -forceLimitY:
                    forceExceeded = True
            elif direction == 'right':
                if forceYTcp > forceLimitY:
                    forceExceeded = True
            else:
                if abs(forceYTcp) > forceLimitY:
                    forceExceeded = True

            if forceExceeded:
                self.movement_progress.emit(
                    f"Force threshold exceeded! Fy: {forceYTcp:.3f} N ({limitDisplay}). Stopping."
                )
                break

            # Update task frame
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            # Compute linear velocity toward target position
            posDirection = targetPos - currentPos
            posNorm = np.linalg.norm(posDirection)
            if posNorm > 1e-6:
                posDirection = posDirection / posNorm
            else:
                posDirection = np.zeros(3)

            # Compute angular velocity toward target orientation (geodesic direction)
            rotDirection = _get_rotation_direction(currentOrientation, targetOrientation)

            # Slow down as we approach target
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

            # Waypoint collection
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progressPct = (1.0 - totalError / initialTotalError) * 100 if initialTotalError > 0 else 100
                self.movement_progress.emit(
                    f"Progress: {progressPct:.1f}% | Fy: {forceYTcp:.2f} N ({limitDisplay}) | "
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
        self.movement_progress.emit(f"Error during new-x force movement: {str(e)}")
        raise


def _calculate_rotation_error(current: np.ndarray, target: np.ndarray) -> float:
    """Calculate the rotation error between two axis-angle representations."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    R_error = R_target @ R_current.T
    r = Rotation.from_matrix(R_error)
    return np.linalg.norm(r.as_rotvec())


def _get_rotation_direction(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return geodesic rotation direction from current to target. Avoids axis-angle subtraction near π."""
    R_current = axis_angle_to_rotation_matrix(current[0], current[1], current[2])
    R_target = axis_angle_to_rotation_matrix(target[0], target[1], target[2])
    R_error = R_target @ R_current.T
    rotvec = Rotation.from_matrix(R_error).as_rotvec()
    norm = np.linalg.norm(rotvec)
    if norm < 1e-6:
        return np.zeros(3)
    return rotvec / norm
