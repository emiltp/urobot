"""
Arc (z) - Force mode + moveL path around ref Z axis.
Uses Fx/Fy/Fz compliance (like New z) with Z and XY limits.
Stop criterion: Mz only.
"""

import time
import numpy as np

from src.utils import orbit_tcp_around_ref, transform_wrench
from src.movements.waypoint_collector import interpolateWaypoints
from config import defaults as CONFIG

_arc_z_cfg = getattr(CONFIG, 'arc_z', CONFIG.arc_force)


def execute(self):
    """Execute arc (z) movement: moveL around ref Z axis, Fx/Fy/Fz compliant, Mz limit only."""
    start_position = self.kwargs.get('start_position')
    ref_pose = self.kwargs.get('ref_pose')
    end_pose = self.kwargs.get('end_pose')
    angle_rad = self.kwargs.get('angle_rad')
    axis = self.kwargs.get('axis', 'z')
    direction_multiplier = self.kwargs.get('direction_multiplier', 1)
    speed = self.kwargs.get('speed', _arc_z_cfg.speed)
    accel = self.kwargs.get('accel', _arc_z_cfg.acceleration)
    max_moment = self.kwargs.get('max_moment', _arc_z_cfg.max_moment)
    force_mode_z_limit = self.kwargs.get('force_mode_z_limit', _arc_z_cfg.force_mode_z_limit)
    force_mode_xy_limit = self.kwargs.get('force_mode_xy_limit', getattr(_arc_z_cfg, 'force_mode_xy_limit', 0.05))
    z_damping = self.kwargs.get('force_mode_z_damping', _arc_z_cfg.force_mode_damping)
    xy_damping = self.kwargs.get('force_mode_xy_damping', _arc_z_cfg.force_mode_damping)
    force_mode_damping = (z_damping + xy_damping) / 2
    z_gain = self.kwargs.get('force_mode_z_gain_scaling', _arc_z_cfg.force_mode_gain_scaling)
    xy_gain = self.kwargs.get('force_mode_xy_gain_scaling', _arc_z_cfg.force_mode_gain_scaling)
    force_mode_gain_scaling = (z_gain + xy_gain) / 2
    waypoint_count = self.kwargs.get('waypoint_count', _arc_z_cfg.waypoint_count)
    target_distance = self.kwargs.get('target_distance', _arc_z_cfg.target_distance)
    force_baseline = self.kwargs.get('force_baseline', 'null')
    blend = 0.01

    self.movement_started.emit()
    self.movement_progress.emit("Starting Arc (z) moveL, Fx/Fy/Fz compliant, Mz limit...")

    try:
        self.movement_progress.emit("Nulling force sensor...")
        self.rtde_c.zeroFtSensor()
        time.sleep(0.3)

        initial_wrench_base = None
        if force_baseline == 'subtract_initial':
            self.movement_progress.emit("Capturing baseline...")
            time.sleep(0.2)
            initial_wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(initial_wrench_base) < 6:
                initial_wrench_base = list(initial_wrench_base) + [0.0] * (6 - len(initial_wrench_base))

        self.waypointCollector.collect(start_position)

        angles = np.linspace(0, angle_rad, max(2, waypoint_count))
        waypoints = np.array([orbit_tcp_around_ref(start_position, ref_pose, a, axis) for a in angles])

        timestamps = np.linspace(0, 1, len(waypoints))
        waypoints, _ = interpolateWaypoints(waypoints, timestamps, targetDistance=target_distance)

        # Fx, Fy, Fz compliant (like New z) - freedom in XY plane and Z
        selection_vector = [1, 1, 1, 0, 0, 0]
        force_type = 2
        limits = [force_mode_xy_limit, force_mode_xy_limit, force_mode_z_limit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(force_mode_damping)
        self.rtde_c.forceModeSetGainScaling(force_mode_gain_scaling)

        initial_pose = list(self.rtde_r.getActualTCPPose())
        target_wrench = (
            transform_wrench(initial_pose, initial_wrench_base)
            if initial_wrench_base is not None
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.rtde_c.forceMode(initial_pose, selection_vector, target_wrench, force_type, limits)

        path = []
        for i, wp in enumerate(waypoints):
            pose = wp.tolist()
            wp_blend = 0.0 if i == 0 or i == len(waypoints) - 1 else blend
            path.append(pose + [speed, accel, wp_blend])

        success = self.rtde_c.moveL(path, asynchronous=True)

        if not success:
            self.rtde_c.forceModeStop()
            self.movement_progress.emit("moveL path failed to start")
            return

        last_waypoint_time = time.time()
        waypoint_interval = 0.05

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            current_pose = list(self.rtde_r.getActualTCPPose())

            target_wrench = (
                transform_wrench(current_pose, initial_wrench_base)
                if initial_wrench_base is not None
                else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.rtde_c.forceMode(current_pose, selection_vector, target_wrench, force_type, limits)

            # Check Mz only (Arc z limit)
            wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(wrench_base) < 6:
                wrench_base = wrench_base + [0.0] * (6 - len(wrench_base))
            if initial_wrench_base is not None:
                wrench_base = [c - i for c, i in zip(wrench_base, initial_wrench_base)]
            wrench_tcp = transform_wrench(current_pose, wrench_base)
            mz = wrench_tcp[5]

            mz_exceeded = (direction_multiplier > 0 and mz < -max_moment) or (
                direction_multiplier < 0 and mz > max_moment
            )
            if mz_exceeded:
                self.movement_progress.emit(f"Mz {mz:.2f} Nm exceeds limit. Stopping.")
                self.rtde_c.stopL()
                break

            now = time.time()
            if now - last_waypoint_time >= waypoint_interval:
                self.pose_updated.emit(current_pose)
                self.waypointCollector.collect(current_pose)
                last_waypoint_time = now
                progress = self.rtde_c.getAsyncOperationProgress()
                self.movement_progress.emit(f"Progress: {progress} | Mz: {mz:.2f} Nm")

            time.sleep(0.02)

        self.rtde_c.forceModeStop()
        time.sleep(0.1)

        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Arc (z) moveL completed.")
        final_pose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(final_pose)
        self.waypointCollector.collect(final_pose)

    except Exception as e:
        try:
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during Arc (z) moveL: {str(e)}")
        raise
