"""
Axial Rotation Hybrid - Force mode wrapper around moveL.

Uses force mode for full translational compliance (Fx=Fy=Fz=0) while executing
a standard moveL for the rotation around TCP z-axis. All three positional axes
float freely so the tool maintains zero contact force in every direction.
"""

import time
from config import defaults as CONFIG


def execute(self):
    """Execute axial rotation with full translational compliance (force mode + moveL)."""
    startPosition = self.kwargs.get('start_position')
    newPose = self.kwargs.get('new_pose')
    speed = self.kwargs.get('speed', CONFIG.rotation.speed)
    accel = self.kwargs.get('accel', CONFIG.rotation.acceleration)
    maxMoment = self.kwargs.get('max_moment', CONFIG.rotation.max_moment)

    # Force mode parameters
    forceModeXYLimit = self.kwargs.get('force_mode_xy_limit', CONFIG.rotation_hybrid.force_mode_xy_limit)
    forceModeZLimit = self.kwargs.get('force_mode_z_limit', CONFIG.rotation_hybrid.force_mode_z_limit)
    forceModeDamping = self.kwargs.get('force_mode_damping', CONFIG.rotation_hybrid.force_mode_damping)
    forceModeGainScaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.rotation_hybrid.force_mode_gain_scaling)

    self.movement_started.emit()
    self.movement_progress.emit("Starting axial rotation (hybrid: full compliance + moveL)...")

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

        initialPose = list(self.rtde_r.getActualTCPPose())
        self.rtde_c.forceMode(initialPose, selection_vector, target_wrench, force_type, limits)

        # =====================================================
        # EXECUTE MOVEL WITH FORCE MODE ACTIVE
        # =====================================================
        self.movement_progress.emit(f"Moving to new pose (speed: {speed} m/s, accel: {accel} m/s²)...")
        success = self.rtde_c.moveL(newPose, speed, accel, asynchronous=True)

        if not success:
            self.rtde_c.forceModeStop()
            self.movement_progress.emit("Movement command returned False")
            return

        self.movement_progress.emit("Asynchronous operation started with full translational compliance...")

        lastWaypointTime = time.time()
        waypointInterval = 0.05

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            currentPose = list(self.rtde_r.getActualTCPPose())

            # Update task frame so compliance stays in TCP frame
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)

            tcpForce = self.robot.getTcpForceInTcpFrame()
            if tcpForce is None:
                tcpForce = [0.0] * 6
            torqueMagnitudeZ = abs(tcpForce[5])
            forceXTcp = tcpForce[0]
            forceYTcp = tcpForce[1]
            forceZTcp = tcpForce[2]

            # Check torque limit (Mz in TCP frame)
            if torqueMagnitudeZ > maxMoment:
                self.movement_progress.emit(
                    f"Contact detected! Torque {torqueMagnitudeZ:.2f} Nm exceeds limit {maxMoment:.2f} Nm. Stopping."
                )
                try:
                    self.rtde_c.stopL()
                except Exception:
                    pass
                self.rtde_c.forceModeStop()
                time.sleep(0.5)
                contactPose = list(self.rtde_r.getActualTCPPose())
                self.waypointCollector.collect(contactPose)
                self.pose_updated.emit(contactPose)
                self.movement_progress.emit(
                    f"Contact pose: [{contactPose[0]:.4f}, {contactPose[1]:.4f}, {contactPose[2]:.4f}]"
                )
                return

            # Collect waypoint at intervals
            currentTime = time.time()
            if (currentTime - lastWaypointTime) >= waypointInterval:
                self.pose_updated.emit(currentPose)
                self.waypointCollector.collect(currentPose)
                lastWaypointTime = currentTime

                progress = self.rtde_c.getAsyncOperationProgress()
                self.movement_progress.emit(
                    f"Progress: {progress} | Mz: {torqueMagnitudeZ:.2f} Nm (limit: {maxMoment:.2f}) | "
                    f"Fx: {forceXTcp:.3f} Fy: {forceYTcp:.3f} Fz: {forceZTcp:.3f} N (all force-controlled)"
                )

            time.sleep(0.02)

        # =====================================================
        # CLEANUP
        # =====================================================
        self.rtde_c.forceModeStop()

        if self._stop_requested:
            try:
                self.rtde_c.stopL()
            except Exception:
                pass
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
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during rotation hybrid movement: {str(e)}")
        raise
