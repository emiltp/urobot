"""
New Z Original - Software-based force compensation with stop-adjust-restart.

Asynchronous moveL to the orbital target around the Ref frame's z-axis.
Uses Mz (moment around TCP z-axis) as the safety limit instead of Fy/Fx.
Active software compensation targets Fx=0, Fy=0, and Fz=0 simultaneously
by adjusting the target position along the full TCP X, Y, and Z directions.
"""

import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix, transform_wrench
from config import defaults as CONFIG


def execute(self):
    """Execute new-z movement (original method)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.new_z.speed)
    accel = self.kwargs.get('accel', CONFIG.new_z.acceleration)
    forceControlGain = self.kwargs.get('force_control_gain', CONFIG.new_z_original.force_control_gain)
    forceDeadband = self.kwargs.get('force_deadband', CONFIG.new_z_original.force_deadband)
    maxAdjustmentPerStep = self.kwargs.get('max_adjustment_per_step', CONFIG.new_z_original.max_adjustment_per_step)
    minAdjustmentInterval = self.kwargs.get('min_adjustment_interval', CONFIG.new_z_original.min_adjustment_interval)
    momentLimitZ = self.kwargs.get('moment_limit_z', CONFIG.new_z.max_moment)
    pathFile = self.kwargs.get('path_file', None)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-z movement...")

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
        self.movement_progress.emit("Active force control: Maintaining Fx, Fy, Fz at 0.0 N")

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            progress = self.rtde_c.getAsyncOperationProgress()
            if progress >= 0:
                self.movement_progress.emit(f"Asynchronous operation progress: {progress}")

            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpPose = list(self.rtde_r.getActualTCPPose())
            tcpForce = transform_wrench(tcpPose, tcpWrenchInBase)
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

            self.movement_progress.emit(
                f"Force in TCP frame:\n"
                f"  Fx: {forceXTcp:.3f} N (target: 0.0 N)\n"
                f"  Fy: {forceYTcp:.3f} N (target: 0.0 N)\n"
                f"  Fz: {forceZTcp:.3f} N (target: 0.0 N)\n"
                f"  Mz: {momentZTcp:.3f} Nm ({limitDisplay})"
            )

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
                    f"Moment threshold exceeded! Mz: {momentZTcp:.3f} Nm ({limitDisplay}). Stopping movement."
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
            forceXYZMag = math.sqrt(forceXTcp**2 + forceYTcp**2 + forceZTcp**2)
            if forceXYZMag > forceDeadband and (currentTime - lastAdjustmentTime) >= minAdjustmentInterval:
                if self._stop_requested:
                    break

                self.movement_progress.emit(f"Force deviation detected ({forceXYZMag:.3f} N). Adjusting path...")

                rx, ry, rz = currentPose[3], currentPose[4], currentPose[5]
                R_tcp_to_base = axis_angle_to_rotation_matrix(rx, ry, rz)

                adjustedTarget = currentTargetPose.copy()

                for axis_idx, forceVal in enumerate([forceXTcp, forceYTcp, forceZTcp]):
                    if abs(forceVal) > forceDeadband:
                        tcpAxisInBase = R_tcp_to_base[:, axis_idx]
                        adjMag = min(abs(forceVal) * forceControlGain, maxAdjustmentPerStep)
                        adj = np.sign(forceVal) * tcpAxisInBase * adjMag
                        adjustedTarget[0] += adj[0]
                        adjustedTarget[1] += adj[1]
                        adjustedTarget[2] += adj[2]

                self.movement_progress.emit(
                    f"Adjusting target position to compensate Fx/Fy/Fz"
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
        self.movement_progress.emit(f"Error during new-z movement: {str(e)}")
        raise
