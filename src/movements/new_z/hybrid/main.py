"""
New Z Hybrid - Force mode wrapper around moveL.

Uses force mode for Fx=0, Fy=0, Fz=0 compliance (all three translational axes)
while executing a standard moveL for the orbital motion around the Ref frame's
z-axis. The safety limit is Mz (moment around TCP z-axis).
"""

import time
import math
import os
import numpy as np
from src.utils import axis_angle_to_rotation_matrix
from config import defaults as CONFIG


def execute(self):
    """Execute new-z movement using hybrid approach (force mode + moveL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.new_z.speed)
    accel = self.kwargs.get('accel', CONFIG.new_z.acceleration)
    momentLimitZ = self.kwargs.get('moment_limit_z', CONFIG.new_z.max_moment)
    pathFile = self.kwargs.get('path_file', None)

    z_cfg = CONFIG.new_z_hybrid.z_limit
    xy_cfg = CONFIG.new_z_hybrid.xy_limit
    forceModeXYLimit = self.kwargs.get('force_mode_xy_limit', xy_cfg.force_mode_xy_limit)
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', z_cfg.force_mode_z_limit)
    forceModeZDamping = self.kwargs.get('force_mode_z_damping', z_cfg.force_mode_damping)
    forceModeXYDamping = self.kwargs.get('force_mode_xy_damping', xy_cfg.force_mode_damping)
    forceModeDamping = self.kwargs.get('force_mode_damping', (forceModeZDamping + forceModeXYDamping) / 2)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', (z_cfg.force_mode_gain_scaling + xy_cfg.force_mode_gain_scaling) / 2)

    direction = None
    if pathFile:
        filename = os.path.basename(pathFile)
        if filename.startswith('left.'):
            direction = 'left'
        elif filename.startswith('right.'):
            direction = 'right'

    self.movement_started.emit()
    self.movement_progress.emit("Starting new-z movement (hybrid: force mode + moveL)...")

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

        self.movement_progress.emit(
            f"Force mode enabled: Fx/Fy/Fz compliance active (Z limit: {forceModeZLimit} m/s, XY limit: {forceModeXYLimit} m/s)"
        )

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

            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
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
                    f"Moment threshold exceeded! Mz: {momentZTcp:.3f} Nm ({limitDisplay}). Stopping movement."
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
                    f"Progress: {progress} | Mz: {momentZTcp:.3f} Nm ({limitDisplay}) | "
                    f"Fx: {forceXTcp:.3f} N, Fy: {forceYTcp:.3f} N (force-controlled) | "
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
        self.movement_progress.emit(f"Error during new-z hybrid movement: {str(e)}")
        raise
