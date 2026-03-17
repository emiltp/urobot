"""
Arc (Fz-constrained) - Force mode + moveL path.

Uses force mode with Z-only compliance and a pre-computed moveL path.
Simplest approach; no mid-flight adaptation.
"""

import time
import numpy as np

from src.utils import orbit_tcp_around_ref, transform_wrench
from src.movements.waypoint_collector import interpolateWaypoints
from config import defaults as CONFIG


def execute(self):
    """Execute arc movement with force mode + moveL path."""
    start_position = self.kwargs.get('start_position')
    ref_pose = self.kwargs.get('ref_pose')
    end_pose = self.kwargs.get('end_pose')
    angle_rad = self.kwargs.get('angle_rad')
    axis = self.kwargs.get('axis', 'z')
    direction_multiplier = self.kwargs.get('direction_multiplier', 1)
    speed = self.kwargs.get('speed', CONFIG.arc_force.speed)
    accel = self.kwargs.get('accel', CONFIG.arc_force.acceleration)
    max_moment = self.kwargs.get('max_moment', CONFIG.arc_force.max_moment)
    force_mode_z_limit = self.kwargs.get('force_mode_z_limit', CONFIG.arc_force.force_mode_z_limit)
    force_mode_damping = self.kwargs.get('force_mode_damping', CONFIG.arc_force.force_mode_damping)
    force_mode_gain_scaling = self.kwargs.get('force_mode_gain_scaling', CONFIG.arc_force.force_mode_gain_scaling)
    fy_tolerance = self.kwargs.get('fy_tolerance', CONFIG.arc_force.fy_tolerance)
    waypoint_count = self.kwargs.get('waypoint_count', CONFIG.arc_force.waypoint_count)
    target_distance = self.kwargs.get('target_distance', CONFIG.arc_force.target_distance)
    force_baseline = self.kwargs.get('force_baseline', 'null')
    blend = 0.01

    self.movement_started.emit()
    self.movement_progress.emit("Starting arc (moveL, Fz-constrained)...")

    try:
        # Always zero - UR force mode requires it; without zero the TCP drops
        self.movement_progress.emit("Nulling force sensor...")
        self.rtde_c.zeroFtSensor()
        time.sleep(0.3)

        initial_wrench_base = None
        if force_baseline == 'subtract_initial':
            # Capture post-zero residual (gravity, drift) as baseline for target and stop criteria
            self.movement_progress.emit("Capturing baseline for subtract-initial...")
            time.sleep(0.2)
            initial_wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(initial_wrench_base) < 6:
                initial_wrench_base = list(initial_wrench_base) + [0.0] * (6 - len(initial_wrench_base))

        self.waypointCollector.collect(start_position)

        # Generate arc waypoints
        angles = np.linspace(0, angle_rad, max(2, waypoint_count))
        waypoints = np.array([orbit_tcp_around_ref(start_position, ref_pose, a, axis) for a in angles])

        timestamps = np.linspace(0, 1, len(waypoints))
        waypoints, _ = interpolateWaypoints(waypoints, timestamps, targetDistance=target_distance)

        selection_vector = [0, 0, 1, 0, 0, 0]
        force_type = 2
        limits = [0.1, 0.1, force_mode_z_limit, 0.5, 0.5, 0.5]

        self.rtde_c.forceModeSetDamping(force_mode_damping)
        self.rtde_c.forceModeSetGainScaling(force_mode_gain_scaling)

        initial_pose = list(self.rtde_r.getActualTCPPose())
        # Target wrench: zero for null, or initial transformed to current pose (sensor in base after zero)
        target_wrench = (
            transform_wrench(initial_pose, initial_wrench_base)
            if initial_wrench_base is not None
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.rtde_c.forceMode(initial_pose, selection_vector, target_wrench, force_type, limits)

        self.movement_progress.emit(f"Force mode enabled, executing {len(waypoints)} waypoints")

        # Build path: each waypoint + [speed, acceleration, blend]
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

        self.movement_progress.emit("Asynchronous moveL started with force compliance...")

        last_waypoint_time = time.time()
        waypoint_interval = 0.05

        while self.rtde_c.getAsyncOperationProgress() >= 0 and not self._stop_requested:
            current_pose = list(self.rtde_r.getActualTCPPose())

            # Target wrench: initial (base frame) transformed to current TCP frame
            target_wrench = (
                transform_wrench(current_pose, initial_wrench_base)
                if initial_wrench_base is not None
                else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            self.rtde_c.forceMode(current_pose, selection_vector, target_wrench, force_type, limits)

            # Check Fy (tangential) and Mx - directional based on motion direction
            wrench_base = list(self.rtde_r.getActualTCPForce())
            if len(wrench_base) < 6:
                wrench_base = wrench_base + [0.0] * (6 - len(wrench_base))
            if initial_wrench_base is not None:
                wrench_base = [c - i for c, i in zip(wrench_base, initial_wrench_base)]
            wrench_tcp = transform_wrench(current_pose, wrench_base)
            fy = wrench_tcp[1]
            mx = wrench_tcp[3]

            fy_exceeded = (direction_multiplier > 0 and fy > fy_tolerance) or (
                direction_multiplier < 0 and fy < -fy_tolerance
            )
            mx_exceeded = (direction_multiplier > 0 and mx < -max_moment) or (
                direction_multiplier < 0 and mx > max_moment
            )
            if fy_exceeded:
                self.movement_progress.emit(f"Fy (tangential) {fy:.2f} N exceeds tolerance. Stopping.")
                self.rtde_c.stopL()
                break
            if mx_exceeded:
                self.movement_progress.emit(f"Mx {mx:.2f} Nm exceeds limit. Stopping.")
                self.rtde_c.stopL()
                break

            now = time.time()
            if now - last_waypoint_time >= waypoint_interval:
                self.pose_updated.emit(current_pose)
                self.waypointCollector.collect(current_pose)
                last_waypoint_time = now
                progress = self.rtde_c.getAsyncOperationProgress()
                self.movement_progress.emit(f"Progress: {progress} | Fy: {fy:.3f} N | Mx: {mx:.2f} Nm")

            time.sleep(0.02)

        self.rtde_c.forceModeStop()
        time.sleep(0.1)

        if self._stop_requested:
            self.movement_progress.emit("Movement stopped by user")
            return

        self.movement_progress.emit("Arc (moveL) completed.")
        final_pose = list(self.rtde_r.getActualTCPPose())
        self.pose_updated.emit(final_pose)
        self.waypointCollector.collect(final_pose)

    except Exception as e:
        try:
            self.rtde_c.forceModeStop()
        except Exception:
            pass
        self.movement_progress.emit(f"Error during arc moveL: {str(e)}")
        raise
