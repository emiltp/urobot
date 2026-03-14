"""
New X Original - Software-based Fz compensation with stop-adjust-restart.

Asynchronous moveL to the orbital target with active Fz software compensation.
Unlike flexion_x, the TCP orbits around the Ref frame axis, so all three
position axes may change and force adjustments use the full TCP Z direction.
"""

import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix, transform_wrench
from config import defaults as CONFIG


def execute(self):
    """Execute new-x movement (original method)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.new_x.speed)
    accel = self.kwargs.get('accel', CONFIG.new_x.acceleration)
    forceControlGain = self.kwargs.get('force_control_gain', CONFIG.new_x_original.force_control_gain)
    forceDeadband = self.kwargs.get('force_deadband', CONFIG.new_x_original.force_deadband)
    maxAdjustmentPerStep = self.kwargs.get('max_adjustment_per_step', CONFIG.new_x_original.max_adjustment_per_step)
    minAdjustmentInterval = self.kwargs.get('min_adjustment_interval', CONFIG.new_x_original.min_adjustment_interval)
    forceLimitY = self.kwargs.get('force_limit_y', CONFIG.new_x.force_limit_y)
    pathFile = self.kwargs.get('path_file', None)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-x movement...")

    try:
        self.rtde_c.zeroFtSensor()
        time.sleep(0.1)

        self.waypointCollector.collect(startPosition)

        currentTargetPose = newPose.copy()
        adjustmentCount = 0
        lastAdjustmentTime = time.time()

        self.movement_progress.emit(f"Moving to new pose (speed: {speed} m/s, accel: {accel} m/s²)...")
        success = self.rtde_c.moveL(currentTargetPose, speed, accel, asynchronous=True)

        if not success:
            self.movement_progress.emit("Movement command returned False")
            return

        self.movement_progress.emit("Robot is moving asynchronously...")
        self.movement_progress.emit("Active force control: Maintaining force_z_tcp at 0.0 N")

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            progress = self.rtde_c.getAsyncOperationProgress()
            if progress >= 0:
                self.movement_progress.emit(f"Asynchronous operation progress: {progress}")

            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpPose = list(self.rtde_r.getActualTCPPose())
            tcpForce = transform_wrench(tcpPose, tcpWrenchInBase)
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]
            forceMagnitude = math.sqrt(tcpForce[0]**2 + tcpForce[1]**2 + tcpForce[2]**2)

            if direction == 'left':
                limitDisplay = f"limit: > -{forceLimitY:.2f} N"
            elif direction == 'right':
                limitDisplay = f"limit: < {forceLimitY:.2f} N"
            else:
                limitDisplay = f"limit: ±{forceLimitY:.2f} N"

            self.movement_progress.emit(
                f"Force in TCP frame:\n"
                f"  Fy: {forceYTcp:.3f} N ({limitDisplay})\n"
                f"  Fz: {forceZTcp:.3f} N (target: 0.0 N)\n"
                f"  Total: {forceMagnitude:.2f} N"
            )

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
                    f"Force threshold exceeded! Fy: {forceYTcp:.3f} N ({limitDisplay}). Stopping movement."
                )
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                time.sleep(1.0)
                return

            currentPose = list(self.rtde_r.getActualTCPPose())
            self.pose_updated.emit(currentPose)
            self.waypointCollector.collect(currentPose)

            currentTime = time.time()
            if abs(forceZTcp) > forceDeadband and (currentTime - lastAdjustmentTime) >= minAdjustmentInterval:
                if self._stop_requested:
                    break

                self.movement_progress.emit(f"Force deviation detected ({forceZTcp:.3f} N). Adjusting path...")

                rx, ry, rz = currentPose[3], currentPose[4], currentPose[5]
                R_tcp_to_base = axis_angle_to_rotation_matrix(rx, ry, rz)
                tcpZDirectionBase = R_tcp_to_base[:, 2]

                # Use full TCP Z direction for adjustment (not projected to YZ
                # as in flexion_x, since the orbital motion involves all axes)
                tcpZNorm = np.linalg.norm(tcpZDirectionBase)
                if tcpZNorm > 0.01:
                    tcpZUnit = tcpZDirectionBase / tcpZNorm
                else:
                    tcpZUnit = np.array([0, 0, -1])

                adjustmentMagnitude = min(abs(forceZTcp) * forceControlGain, maxAdjustmentPerStep)
                positionAdjustment = np.sign(forceZTcp) * tcpZUnit * adjustmentMagnitude

                adjustedTarget = currentTargetPose.copy()
                adjustedTarget[0] += positionAdjustment[0]
                adjustedTarget[1] += positionAdjustment[1]
                adjustedTarget[2] += positionAdjustment[2]

                self.movement_progress.emit(
                    f"Adjusting target: x += {positionAdjustment[0]:.6f} m, "
                    f"y += {positionAdjustment[1]:.6f} m, z += {positionAdjustment[2]:.6f} m"
                )

                self.rtde_c.stopL()
                time.sleep(0.1)

                if self._stop_requested:
                    break

                currentTargetPose = adjustedTarget.copy()
                adjustmentCount += 1
                lastAdjustmentTime = currentTime

                self.movement_progress.emit(f"Resuming movement with adjusted target (adjustment #{adjustmentCount})...")
                success = self.rtde_c.moveL(currentTargetPose, speed, accel, asynchronous=True)

                if not success:
                    self.movement_progress.emit("Warning: Failed to start adjusted movement. Stopping.")
                    return

            time.sleep(0.1)

        if self._stop_requested:
            try:
                self.rtde_c.stopL()
            except Exception:
                pass
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Asynchronous operation completed.")

        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")

    except Exception as e:
        self.movement_progress.emit(f"Error during new-x movement: {str(e)}")
        raise
