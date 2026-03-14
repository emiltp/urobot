"""
New Y Hybrid - Force mode wrapper around moveL.

Uses force mode to maintain Fz=0 while executing a standard moveL
for the orbital motion around the Ref frame's y-axis. The robot follows
the planned trajectory with Z-axis compliance.
"""

import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix, transform_wrench
from config import defaults as CONFIG


def execute(self):
    """Execute new-y movement using hybrid approach (force mode + moveL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.new_y.speed)
    accel = self.kwargs.get('accel', CONFIG.new_y.acceleration)
    forceLimitX = self.kwargs.get('force_limit_x', CONFIG.new_y.force_limit_x)
    pathFile = self.kwargs.get('path_file', None)

    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.new_y_hybrid.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.new_y_hybrid.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.new_y_hybrid.force_mode_gain_scaling)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-y movement (hybrid: force mode + moveL)...")

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

        self.movement_progress.emit(f"Force mode enabled: Fz compliance active (Z limit: {forceModeZLimit} m/s)")

        initialPose = list(self.rtde_r.getActualTCPPose())
        self.rtde_c.forceMode(initialPose, selection_vector, target_wrench, force_type, limits)

        self.movement_progress.emit(f"Moving to target pose (speed: {speed} m/s, accel: {accel} m/s²)...")
        success = self.rtde_c.moveL(newPose, speed, accel, asynchronous=True)

        if not success:
            self.rtde_c.forceModeStop()
            self.movement_progress.emit("Movement command returned False")
            return

        self.movement_progress.emit("Asynchronous operation started with force compliance...")

        lastWaypointTime = time.time()
        waypointInterval = 0.05

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            currentPose = list(self.rtde_r.getActualTCPPose())
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            tcpWrenchInBase = self.rtde_r.getActualTCPForce()
            tcpForce = transform_wrench(currentPose, tcpWrenchInBase)
            forceXTcp = tcpForce[0]
            forceZTcp = tcpForce[2]

            if direction == 'left':
                limitDisplay = f"limit: < {forceLimitX:.2f} N"
            elif direction == 'right':
                limitDisplay = f"limit: > -{forceLimitX:.2f} N"
            else:
                limitDisplay = f"limit: ±{forceLimitX:.2f} N"

            forceExceeded = False
            if direction == 'left':
                if forceXTcp > forceLimitX:
                    forceExceeded = True
            elif direction == 'right':
                if forceXTcp < -forceLimitX:
                    forceExceeded = True
            else:
                if abs(forceXTcp) > forceLimitX:
                    forceExceeded = True

            if forceExceeded:
                self.movement_progress.emit(
                    f"Force threshold exceeded! Fx: {forceXTcp:.3f} N ({limitDisplay}). Stopping movement."
                )
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                self.rtde_c.forceModeStop()
                time.sleep(0.5)
                return

            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progress = self.rtde_c.getAsyncOperationProgress()
                self.movement_progress.emit(
                    f"Progress: {progress} | Fx: {forceXTcp:.2f} N ({limitDisplay}) | "
                    f"Fz: {forceZTcp:.3f} N (force-controlled)"
                )

            time.sleep(0.02)

        self.rtde_c.forceModeStop()

        if self._stop_requested:
            try:
                self.rtde_c.stopL()
            except Exception:
                pass
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Movement completed with force compliance.")

        finalPose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(finalPose)
        self.waypointCollector.collect(finalPose)
        self.movement_progress.emit(f"Final TCP pose: [{finalPose[0]:.4f}, {finalPose[1]:.4f}, {finalPose[2]:.4f}]")

    except Exception as e:
        try:
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during new-y hybrid movement: {str(e)}")
        raise
