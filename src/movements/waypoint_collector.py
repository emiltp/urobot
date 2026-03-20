"""
Waypoint collector for collecting and traversing robot trajectories.
Uses NumPy .npz format for fast, compact storage.
"""
import logging
import os
import time
import numpy as np

_log = logging.getLogger(__name__)
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from typing import Optional, List, Tuple, Callable

from config import defaults as CONFIG, runtime_tcp_offset, get_hybrid_config_for_path, _get_hybrid_force_params
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


def _orientation_diff_rad(pose_a: List[float], pose_b: List[float]) -> float:
    """Geodesic rotation angle (rad) between two poses. Use instead of axis-angle Euclidean difference."""
    R_a = axis_angle_to_rotation_matrix(pose_a[3], pose_a[4], pose_a[5])
    R_b = axis_angle_to_rotation_matrix(pose_b[3], pose_b[4], pose_b[5])
    R_err = R_b @ R_a.T
    return np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec())


def _orientation_direction(pose_a: List[float], pose_b: List[float]) -> np.ndarray:
    """Unit rotvec direction from pose_a to pose_b (geodesic). Returns zeros if already aligned."""
    R_a = axis_angle_to_rotation_matrix(pose_a[3], pose_a[4], pose_a[5])
    R_b = axis_angle_to_rotation_matrix(pose_b[3], pose_b[4], pose_b[5])
    R_err = R_b @ R_a.T
    rotvec = Rotation.from_matrix(R_err).as_rotvec()
    norm = np.linalg.norm(rotvec)
    if norm < 1e-6:
        return np.zeros(3)
    return rotvec / norm

# Default paths derived from config
DEFAULT_WAYPOINTS_FILE = os.path.join(CONFIG.paths.data_dir, CONFIG.paths.path_filename)
DATA_DIR = CONFIG.paths.data_dir

# Minimum rotation weight to avoid division issues
MIN_ROTATION_WEIGHT = 0.01  # 1cm minimum


def getDefaultRotationWeight() -> float:
    """Get the default rotation weight from TCP offset magnitude.
    
    Uses the distance from flange to TCP as the rotation weight.
    This makes the rotation component represent actual arc length traveled by TCP.
    
    Returns:
        TCP offset magnitude in meters (minimum 0.01m)
    """
    tcp_offset = runtime_tcp_offset
    if tcp_offset is None or len(tcp_offset) < 3:
        return MIN_ROTATION_WEIGHT
    
    magnitude = np.sqrt(tcp_offset[0]**2 + tcp_offset[1]**2 + tcp_offset[2]**2)
    return max(magnitude, MIN_ROTATION_WEIGHT)


class WaypointCollector:
    """Collects and traverses TCP poses for robot motion.
    
    Usage for collecting (with runner):
        collector = WaypointCollector(async_motion_runner)
        collector.start()
        collector.collect(pose)
        collector.stop()
        collector.save('path.npz')
    
    Usage for traversing:
        collector = WaypointCollector.load(async_motion_runner, 'path.npz')
        collector.forwardTraverse(speed=0.01)
    
    Usage for retracing:
        collector = WaypointCollector.load(async_motion_runner, 'path.npz')
        collector.backwardTraverse(speed=0.02)
    
    The async_motion_runner provides rtde_c, rtde_r, and callbacks for progress/pose/stop.
    """
    
    def __init__(self, async_motion_runner=None):
        self._waypointsList: List[List[float]] = []
        self._timestampsList: List[float] = []
        self.startTime: Optional[float] = None
        self.waypoints: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self.collecting = False
        self.traverseStopIndex = None
        self.filename = ""

        """Setting the async motion runner references for traversal."""
        if async_motion_runner is not None:
            self.robot = async_motion_runner.robot
            self.rtde_c = async_motion_runner.rtde_c
            self.rtde_r = async_motion_runner.rtde_r
            self.progressCallback = lambda msg: async_motion_runner.movement_progress.emit(msg)
            self.poseCallback = lambda pose: async_motion_runner.pose_updated.emit(pose)
            self.stopCheck = lambda: async_motion_runner._stop_requested
        else:
            self.robot = None
            self.rtde_c = None
            self.rtde_r = None
            self.progressCallback = lambda msg: None
            self.poseCallback = lambda pose: None
            self.stopCheck = lambda: False
    
    
    @classmethod
    def load(cls, async_motion_runner, filepath: Optional[str] = None) -> Optional['WaypointCollector']:
        """Load waypoints from file.
        
        Args:
            filepath: Path to the .npz file
            async_motion_runner: Optional runner for traversal callbacks
        """
        filepath = filepath or DEFAULT_WAYPOINTS_FILE
        if not os.path.exists(filepath):
            return None

        with np.load(filepath) as data:
            collector = cls(async_motion_runner)
            collector.waypoints = np.array(data['poses'])
            collector.timestamps = np.array(data['timestamps'])
            collector.filename = filepath
        return collector
    
    # ==================== Collection Methods ====================
    
    def start(self):
        """Start collecting a new waypoint sequence."""
        self._waypointsList = []
        self._timestampsList = []
        self.waypoints = None
        self.timestamps = None
        self.startTime = time.time()
        self.collecting = True
    
    def collect(self, pose: List[float]):
        """Collect a pose with timestamp."""
        if not self.collecting:
            return
        self._waypointsList.append(list(pose))
        self._timestampsList.append(time.time() - self.startTime)
    
    def stop(self):
        """Stop collecting and convert to numpy arrays."""
        self.collecting = False
        if self._waypointsList:
            self.waypoints = np.array(self._waypointsList, dtype=np.float64)
            self.timestamps = np.array(self._timestampsList, dtype=np.float64)
            # Clear lists to free memory - numpy arrays are now authoritative
            self._waypointsList = []
            self._timestampsList = []

            self.eliminateWaypointJumps()
    
    def save(self, filepath: Optional[str] = None) -> bool:
        """Save waypoints to .npz file with jump elimination."""
        # Use numpy arrays if available, otherwise convert from lists
        if self.waypoints is None:
            if not self._waypointsList:
                return False
            self.waypoints = np.array(self._waypointsList, dtype=np.float64)
            self.timestamps = np.array(self._timestampsList, dtype=np.float64)
            self._waypointsList = []
            self._timestampsList = []
        
        filepath = filepath or DEFAULT_WAYPOINTS_FILE
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filename = filepath
        np.savez_compressed(filepath, poses=self.waypoints, timestamps=self.timestamps)
        return True
    
    # ==================== Accessors ====================

    def getWaypointsDisplayName(self) -> str:
        """Get display name for the waypoints file (without .path.npz extension)."""
        if self.filename.endswith('.path.npz'):
            return self.filename[:-9]
        elif self.filename.endswith('.npz'):
            return self.filename[:-4]
        return self.filename
    
    def getWaypointCount(self) -> int:
        """Return number of waypoints."""
        if self.waypoints is not None:
            return len(self.waypoints)
        return len(self._waypointsList)
    
    def getWaypoints(self) -> Optional[np.ndarray]:
        if self.waypoints is None and self._waypointsList:
            self.waypoints = np.array(self._waypointsList, dtype=np.float64)
        return self.waypoints
    
    def getTimestamps(self) -> Optional[np.ndarray]:
        if self.timestamps is None and self._timestampsList:
            self.timestamps = np.array(self._timestampsList, dtype=np.float64)
        return self.timestamps
    
    def getDistance(self) -> float:
        waypoints = self.getWaypoints()
        return calculateWaypointsDistance(waypoints) if waypoints is not None else 0.0
    
    # ==================== Traversal Methods ====================
    
    def forwardTraverse(self, speed: Optional[float] = None,
                 acceleration: Optional[float] = None,
                 blend: Optional[float] = None,
                 traverseMethod: Optional[str] = None,
                 enableForceControl: Optional[bool] = False,
                 forceLimit: Optional[float] = None,
                 forceAxis: Optional[str] = None,
                 direction: Optional[str] = None,
                 forceFrame: str = 'tcp',
                 momentLimit: Optional[float] = None,
                 momentAxis: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray], int]:
        """Traverse waypoints with optional force control.
        
        Args:
            speed: Speed in m/s (user-determined, independent of collection)
            acceleration: Acceleration in m/s² (user-determined)
            blend: Blend radius in meters for smooth transitions
            traverseMethod: 'moveLPath', 'servoPath', or 'movePath'
            enableForceControl: Enable force control
            forceLimit: Force or moment limit (N for x/y, Nm for mz)
            forceAxis: 'x', 'y', or 'mz'
            direction: Direction (left or right)
            forceFrame: 'tcp' or 'ref' for translational x/y checks (flexion uses 'ref')
            momentLimit: Optional TCP moment limit in Nm (flexion Mx/My; 0 or None disables)
            momentAxis: 'mx' or 'my' when momentLimit is used

        Returns: (success, traveledWaypoints, stopIndex)
        """
        
        waypoints = self.getWaypoints()
        timestamps = self.getTimestamps()
        
        if waypoints is None or len(waypoints) == 0:
            if self.progressCallback:
                self.progressCallback("No waypoints available for traversal")
            return False, None, -1
        
        # Default to moveLPath
        traverseMethod = traverseMethod or 'moveLPath'
        
        # Log traverse method and settings
        print(f"\n=== Forward Traverse ===")
        print(f"Method: {traverseMethod}")
        print(f"Waypoints: {len(waypoints)}")
        
        # Select execution method based on traverseMethod
        if traverseMethod == 'forceHybrid':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movelpath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movelpath.acceleration
            effectiveBlend = blend if blend is not None else CONFIG.traverse_movelpath.blend
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"Blend radius: {effectiveBlend} m")
            print(f"Force mode: Fz compliance enabled")
            return self._executeForceHybridPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                blend=effectiveBlend,
                isBackwardTraverse=False
            )
        elif traverseMethod == 'forceSpeedL':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_servopath.speed
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Force mode: Fz compliance enabled (speedL control)")
            return self._executeForceSpeedLPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                isBackwardTraverse=False
            )
        elif traverseMethod == 'servoPath':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_servopath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_servopath.acceleration
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"  dt: {CONFIG.traverse_servopath.dt} s")
            print(f"  lookahead_time: {CONFIG.traverse_servopath.lookahead_time} s")
            print(f"  gain: {CONFIG.traverse_servopath.gain}")
            print(f"  ramp_up_time: {CONFIG.traverse_servopath.ramp_up_time} s")
            if enableForceControl:
                print(f"Force control: {forceAxis} axis, limit={forceLimit}, frame={forceFrame}, direction={direction}")
            return self._executeServoPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
                forceFrame=forceFrame,
                momentLimit=momentLimit,
                momentAxis=momentAxis,
                isBackwardTraverse=False
            )
        elif traverseMethod == 'movePath':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movepath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movepath.acceleration
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            if enableForceControl:
                print(f"Force control: {forceAxis} axis, limit={forceLimit}, frame={forceFrame}, direction={direction}")
            return self._executeMovePath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
                forceFrame=forceFrame,
                momentLimit=momentLimit,
                momentAxis=momentAxis,
                isBackwardTraverse=False
            )
        else:  # moveLPath (default)
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movelpath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movelpath.acceleration
            effectiveBlend = blend if blend is not None else CONFIG.traverse_movelpath.blend
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"Blend radius: {effectiveBlend} m")
            if enableForceControl:
                print(f"Force control: {forceAxis} axis, limit={forceLimit}, frame={forceFrame}, direction={direction}")
            return self._executeMoveLPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                blend=effectiveBlend,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
                forceFrame=forceFrame,
                momentLimit=momentLimit,
                momentAxis=momentAxis,
                isBackwardTraverse=False
            )
    
    def backwardTraverse(self, speed: Optional[float] = None,
                         acceleration: Optional[float] = None,
                         blend: Optional[float] = None,
                         traverseMethod: Optional[str] = None) -> bool:
        """Retrace waypoints in reverse (return path).
        
        If traverseStopIndex is set (e.g., from force limit stop during forward traverse),
        only retraces the portion of the path that was actually traversed.
        
        Args:
            speed: Speed in m/s (user-determined)
            acceleration: Acceleration in m/s²
            blend: Blend radius in meters for smooth transitions
            traverseMethod: 'moveLPath', 'servoPath', or 'movePath'
        """
        waypoints = self.getWaypoints()
        timestamps = self.getTimestamps()
        
        if waypoints is None or len(waypoints) == 0:
            if self.progressCallback:
                self.progressCallback("No waypoints available for retrace")
            return False
        
        speed = speed or CONFIG.traverse.retrace_speed
        traverseMethod = traverseMethod or 'moveLPath'
        
        # If forward traverse stopped early (force limit), only retrace the traveled portion
        if self.traverseStopIndex is not None and 0 <= self.traverseStopIndex < len(waypoints) - 1:
            stopIdx = self.traverseStopIndex + 1  # Include the stop index
            waypoints = waypoints[:stopIdx]
            timestamps = timestamps[:stopIdx]
            if self.progressCallback:
                self.progressCallback(f"Backward traverse from stop index {self.traverseStopIndex} ({len(waypoints)} waypoints)")
        
        # Reverse waypoints and timestamps
        reversedWaypoints = waypoints[::-1].copy()
        if len(timestamps) > 1:
            deltas = np.diff(timestamps)[::-1]
            reversedTimestamps = np.zeros(len(timestamps))
            reversedTimestamps[1:] = np.cumsum(deltas)
        else:
            reversedTimestamps = np.array([0.0])
        
        # Log backward traverse settings
        print(f"\n=== Backward Traverse ===")
        print(f"Method: {traverseMethod}")
        print(f"Waypoints: {len(reversedWaypoints)}")
        
        # Select execution method based on traverseMethod
        if traverseMethod == 'forceHybrid':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movelpath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movelpath.acceleration
            effectiveBlend = blend if blend is not None else CONFIG.traverse_movelpath.blend
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"Blend radius: {effectiveBlend} m")
            print(f"Force mode: Fz compliance enabled")
            success = self._executeForceHybridPath(
                reversedWaypoints, reversedTimestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                blend=effectiveBlend,
                isBackwardTraverse=True
            )
        elif traverseMethod == 'forceSpeedL':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_servopath.speed
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Force mode: Fz compliance enabled (speedL control)")
            success = self._executeForceSpeedLPath(
                reversedWaypoints, reversedTimestamps,
                speed=effectiveSpeed,
                isBackwardTraverse=True
            )
        elif traverseMethod == 'servoPath':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_servopath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_servopath.acceleration
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"  dt: {CONFIG.traverse_servopath.dt} s")
            print(f"  lookahead_time: {CONFIG.traverse_servopath.lookahead_time} s")
            print(f"  gain: {CONFIG.traverse_servopath.gain}")
            print(f"  ramp_up_time: {CONFIG.traverse_servopath.ramp_up_time} s")
            success = self._executeServoPath(
                reversedWaypoints, reversedTimestamps,
                speed=effectiveSpeed,
                isBackwardTraverse=True
            )
        elif traverseMethod == 'movePath':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movepath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movepath.acceleration
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            success = self._executeMovePath(
                reversedWaypoints, reversedTimestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                isBackwardTraverse=True
            )
        else:  # moveLPath (default)
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movelpath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movelpath.acceleration
            effectiveBlend = blend if blend is not None else CONFIG.traverse_movelpath.blend
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            print(f"Blend radius: {effectiveBlend} m")
            success = self._executeMoveLPath(
                reversedWaypoints, reversedTimestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                blend=effectiveBlend,
                isBackwardTraverse=True
            )
        return success
    
    def forceCompliantBackwardTraverse(self, speed: Optional[float] = None,
                                        acceleration: Optional[float] = None) -> bool:
        """Retrace waypoints in reverse with force mode compliance (Fz=0).
        
        Uses hardware force mode to maintain Fz=0 while executing the return path.
        This is used when the collection method was hybrid or force.
        
        Args:
            speed: Traversal speed in m/s
            acceleration: Acceleration in m/s²
            
        Returns:
            True if successful, False otherwise
        """
        waypoints = self.getWaypoints()
        timestamps = self.getTimestamps()
        
        if waypoints is None or len(waypoints) == 0:
            if self.progressCallback:
                self.progressCallback("No waypoints available for force-compliant retrace")
            return False
        
        speed = speed or CONFIG.traverse.retrace_speed
        acceleration = acceleration or CONFIG.traverse_movelpath.acceleration
        
        # If forward traverse stopped early (force limit), only retrace the traveled portion
        if self.traverseStopIndex is not None and 0 <= self.traverseStopIndex < len(waypoints) - 1:
            stopIdx = self.traverseStopIndex + 1
            waypoints = waypoints[:stopIdx]
            timestamps = timestamps[:stopIdx]
            if self.progressCallback:
                self.progressCallback(f"Force-compliant backward traverse from stop index {self.traverseStopIndex} ({len(waypoints)} waypoints)")
        
        # Reverse waypoints
        reversedWaypoints = waypoints[::-1].copy()
        
        # Calculate path info
        waypointsDistance = calculateWaypointsDistance(reversedWaypoints)
        
        print(f"\n=== Force-Compliant Backward Traverse ===")
        print(f"Waypoints: {len(reversedWaypoints)}")
        print(f"Speed: {speed} m/s")
        print(f"Acceleration: {acceleration} m/s²")
        print(f"Force mode: Fz compliance enabled")
        
        if self.progressCallback:
            self.progressCallback(f"Force-compliant backward traverse: {len(reversedWaypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s")
        
        # Stats tracking
        trackingTcpPoses = []
        trackingFlangePoses = []
        trackingTimestamps = []
        
        try:
            # Get current pose
            currentPose = self.robot.getTcpPose() if self.robot else None
            if currentPose is None or len(currentPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Error: No valid TCP pose available for backward traverse.")
                return False

            # Force mode parameters (motion-specific hybrid config from path filename)
            hybrid_cfg = get_hybrid_config_for_path(self.filename)
            z_limit, damping, gain = _get_hybrid_force_params(hybrid_cfg)
            selection_vector = [0, 0, 1, 0, 0, 0]  # Only Z is force-controlled
            target_wrench = [0, 0, 0, 0, 0, 0]  # Fz = 0
            force_type = 2  # Force frame not transformed
            limits = [0.1, 0.1, z_limit, 0.5, 0.5, 0.5]  # Z velocity limit
            
            # Configure force mode
            self.rtde_c.forceModeSetDamping(damping)
            self.rtde_c.forceModeSetGainScaling(gain)
            
            if self.progressCallback:
                self.progressCallback("Force mode enabled for return path")
            
            # Enter force mode
            self.rtde_c.forceMode(currentPose, selection_vector, target_wrench, force_type, limits)
            
            # Build path with parameters
            path = []
            blend = CONFIG.traverse_movelpath.blend
            for i, wp in enumerate(reversedWaypoints):
                pose = wp.tolist()
                wp_blend = 0.0 if i == 0 or i == len(reversedWaypoints) - 1 else blend
                path.append(pose + [speed, acceleration, wp_blend])
            
            # Execute path asynchronously
            success = self.rtde_c.moveL(path, asynchronous=True)
            
            if not success:
                self.rtde_c.forceModeStop()
                if self.progressCallback:
                    self.progressCallback("Force-compliant backward traverse failed to start")
                return False
            
            # Monitor progress
            poseUpdateInterval = CONFIG.traverse.pose_update_interval
            lastPoseUpdate = 0
            statsStartTime = time.time()
            
            while self.rtde_c.getAsyncOperationProgress() >= 0:
                currentTime = time.time()
                
                # Update task frame to current pose (keeps Z-compliance in TCP frame)
                actualPose = self.robot.getTcpPose()
                if actualPose:
                    self.rtde_c.forceMode(actualPose, selection_vector, target_wrench, force_type, limits)
                    
                    # Track for stats
                    trackingTcpPoses.append(actualPose)
                    trackingTimestamps.append(currentTime - statsStartTime)
                    
                    flangePose = self.robot.getFlangePose()
                    if flangePose:
                        trackingFlangePoses.append(flangePose)
                    
                    # Pose callback
                    if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                        try:
                            self.poseCallback(actualPose)
                            lastPoseUpdate = currentTime
                        except Exception as e:
                            _log.warning("Pose callback failed: %s", e)
                
                # Check for stop request
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.stopL()
                    self.rtde_c.forceModeStop()
                    if self.progressCallback:
                        self.progressCallback("Force-compliant backward traverse stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, "Force-compliant backward traverse (stopped)")
                    return False
                
                time.sleep(0.02)  # 50Hz
            
            # Exit force mode
            self.rtde_c.forceModeStop()
            time.sleep(0.1)
            
            # Final pose callback
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback("Force-compliant backward traverse completed")
            
            # Print stats
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, "Force-compliant backward traverse")
            
            return True
            
        except Exception as e:
            try:
                self.rtde_c.forceModeStop()
            except Exception as stop_err:
                _log.debug("forceModeStop during cleanup: %s", stop_err)
            if self.progressCallback:
                self.progressCallback(f"Error during force-compliant backward traverse: {str(e)}")
            return False
    
    # ==================== Private Move Execution ====================
    
    def _executeMovePath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                         speed: float,
                         acceleration: float = 0.1,
                         enableForceControl: bool = False,
                         forceLimit: Optional[float] = 10.0,
                         forceAxis: str = 'y',
                         direction: Optional[str] = None,
                         forceFrame: str = 'tcp',
                         momentLimit: Optional[float] = None,
                         momentAxis: Optional[str] = None,
                         isBackwardTraverse: bool = False) -> bool:
        """Execute path traversal using moveL with async monitoring.
        
        Uses moveL which provides smooth acceleration/deceleration handled by
        the robot controller. Simpler than servoL but follows the same path.
        
        Args:
            waypoints: Array of poses to traverse
            timestamps: Timestamps for each waypoint (used for progress tracking)
            speed: Movement speed in m/s
            acceleration: Movement acceleration in m/s²
            enableForceControl: Enable force limit checking
            forceLimit: Force limit in N
            forceAxis: Force axis ('x' or 'y')
            direction: Direction for force limit sign ('left' or 'right')
            isBackwardTraverse: True if this is a return path
            
        Returns:
            True if completed successfully, False if stopped early
        """
        if waypoints is None or len(waypoints) == 0:
            return False
        
        # Calculate path info
        waypointsDistance = calculateWaypointsDistance(waypoints)
        action = "Backward traversal" if isBackwardTraverse else "Forward traversal"
        
        if self.progressCallback:
            self.progressCallback(f"{action} of {len(waypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s")
        
        # Stats tracking
        trackingTcpPoses = []
        trackingFlangePoses = []
        trackingTimestamps = []
        
        try:
            currentPose = self.robot.getTcpPose() if self.robot else None

            # For forward traverse: move to start if needed
            # For backward traverse: skip "move to start" - just go from current position
            if not isBackwardTraverse and currentPose is not None and len(currentPose) >= 6:
                startPose = waypoints[0].tolist()
                
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = _orientation_diff_rad(currentPose, startPose)
                
                if posDiff > 0.001 or orientDiff > 0.01:
                    if self.progressCallback:
                        self.progressCallback(f"Moving to start (pos: {posDiff*1000:.1f}mm, orient: {np.degrees(orientDiff):.1f}°)...")
                    self.rtde_c.moveL(startPose, speed * 0.5, acceleration)
                    time.sleep(0.1)
                
                # Zero force sensor for forward traverse
                self.rtde_c.zeroFtSensor()
                time.sleep(0.1)
            
            # Target: final waypoint (end for forward, original start for backward)
            targetPose = waypoints[-1].tolist()
            
            if self.progressCallback:
                self.progressCallback(f"Moving to target pose...")
            
            success = self.rtde_c.moveL(targetPose, speed, acceleration, asynchronous=True)
            
            if not success:
                if self.progressCallback:
                    self.progressCallback(f"{action} failed to start")
                return False
            
            # Monitor progress
            poseUpdateInterval = CONFIG.traverse.pose_update_interval
            forceCheckInterval = CONFIG.traverse.force_check_interval
            lastPoseUpdate = 0
            lastForceCheck = 0
            statsStartTime = time.time()
            
            while self.rtde_c.getAsyncOperationProgress() >= 0:
                currentTime = time.time()
                
                # Check for stop request
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.stopL()
                    if self.progressCallback:
                        self.progressCallback(f"{action} stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (stopped)")
                    return False
                
                # Get current pose using robot instance
                actualPose = self.robot.getTcpPose()
                flangePose = self.robot.getFlangePose()
                
                # Track for stats
                if actualPose:
                    trackingTcpPoses.append(actualPose)
                    trackingTimestamps.append(currentTime - statsStartTime)
                if flangePose:
                    trackingFlangePoses.append(flangePose)
                
                # Find closest waypoint index for stop tracking
                if actualPose:
                    minDist = float('inf')
                    for i, wp in enumerate(waypoints):
                        dist = np.sqrt(sum((actualPose[j] - wp[j])**2 for j in range(3)))
                        if dist < minDist:
                            minDist = dist
                            self.traverseStopIndex = i
                    
                    # Pose callback
                    if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                        try:
                            self.poseCallback(actualPose)
                            lastPoseUpdate = currentTime
                        except Exception as e:
                            _log.warning("Pose callback failed: %s", e)
                
                # Force control (only during forward traversal)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(
                        forceLimit, forceAxis, direction,
                        force_frame=forceFrame, moment_limit=momentLimit, moment_axis=momentAxis)
                    if self.progressCallback and msg:
                        self.progressCallback(msg)
                    if exceeded:
                        self.rtde_c.stopL()
                        time.sleep(0.3)  # Allow robot to settle after stop
                        
                        # Update stop index based on actual stopped position
                        stoppedPose = self.robot.getTcpPose()
                        if stoppedPose:
                            minDist = float('inf')
                            for i, wp in enumerate(waypoints):
                                dist = np.sqrt(sum((stoppedPose[j] - wp[j])**2 for j in range(3)))
                                if dist < minDist:
                                    minDist = dist
                                    self.traverseStopIndex = i
                        
                        if self.progressCallback:
                            self.progressCallback(f"Force threshold exceeded! Stopped at index {self.traverseStopIndex}.")
                        self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (force stop)")
                        return False
                    lastForceCheck = currentTime
                
                time.sleep(0.05)  # 50ms poll interval
            
            # Final pose callback
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            # Print traverse statistics
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            
            return True
            
        except Exception as e:
            if self.progressCallback:
                self.progressCallback(f"Error during {action.lower()}: {str(e)}")
            return False
    
    # ==================== Private MoveLPath Execution ====================
    
    def _executeMoveLPath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                          speed: float,
                          acceleration: float = 0.5,
                          blend: float = 0.01,
                          enableForceControl: bool = False,
                          forceLimit: Optional[float] = 10.0,
                          forceAxis: str = 'y',
                          direction: Optional[str] = None,
                          forceFrame: str = 'tcp',
                          momentLimit: Optional[float] = None,
                          momentAxis: Optional[str] = None,
                          isBackwardTraverse: bool = False) -> bool:
        """Execute path traversal using moveL with path parameter.
        
        Uses moveL(path) which the robot controller handles as a smooth path.
        Each waypoint includes [x, y, z, rx, ry, rz, speed, accel, blend].
        
        Args:
            waypoints: Array of poses to traverse
            timestamps: Timestamps for each waypoint (used for progress tracking)
            speed: Movement speed in m/s
            acceleration: Movement acceleration in m/s²
            blend: Blend radius in meters (smoothing between waypoints)
            enableForceControl: Enable force limit checking
            forceLimit: Force limit in N
            forceAxis: Force axis ('x' or 'y')
            direction: Direction for force limit sign ('left' or 'right')
            isBackwardTraverse: True if this is a return path
            
        Returns:
            True if completed successfully, False if stopped early
        """
        if waypoints is None or len(waypoints) == 0:
            return False
        
        # Calculate path info
        waypointsDistance = calculateWaypointsDistance(waypoints)
        action = "Backward traversal" if isBackwardTraverse else "Forward traversal"
        
        if self.progressCallback:
            self.progressCallback(f"{action} of {len(waypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s (moveLPath)")
        
        try:
            currentPose = list(self.rtde_r.getActualTCPPose()) if self.rtde_r else None
            if currentPose is None or len(currentPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Error: No valid TCP pose for traverse.")
                return False

            # For forward traverse: move to start if needed
            if not isBackwardTraverse:
                startPose = waypoints[0].tolist()
                
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = _orientation_diff_rad(currentPose, startPose)
                
                if posDiff > 0.001 or orientDiff > 0.01:
                    if self.progressCallback:
                        self.progressCallback(f"Moving to start (pos: {posDiff*1000:.1f}mm, orient: {np.degrees(orientDiff):.1f}°)...")
                    self.rtde_c.moveL(startPose, speed * 0.5, acceleration)
                    time.sleep(0.1)
                
                # Zero force sensor for forward traverse
                self.rtde_c.zeroFtSensor()
                time.sleep(0.1)
            
            # Build path with parameters for each waypoint
            # Format: [x, y, z, rx, ry, rz, speed, acceleration, blend]
            path = []
            for i, wp in enumerate(waypoints):
                pose = wp.tolist()
                # Use smaller blend for first and last waypoints
                wp_blend = 0.0 if i == 0 or i == len(waypoints) - 1 else blend
                path.append(pose + [speed, acceleration, wp_blend])
            
            if self.progressCallback:
                self.progressCallback(f"Executing path with {len(path)} waypoints...")
            
            # Execute path asynchronously so we can monitor progress
            success = self.rtde_c.moveL(path, asynchronous=True)
            
            if not success:
                if self.progressCallback:
                    self.progressCallback(f"{action} failed to start")
                return False
            
            # Monitor progress
            poseUpdateInterval = CONFIG.traverse.pose_update_interval
            forceCheckInterval = CONFIG.traverse.force_check_interval
            lastPoseUpdate = 0
            lastForceCheck = 0
            
            # Stats tracking
            trackingTcpPoses = []
            trackingFlangePoses = []
            trackingTimestamps = []
            statsStartTime = time.time()
            
            while self.rtde_c.getAsyncOperationProgress() >= 0:
                currentTime = time.time()
                
                # Check for stop request
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.stopL()
                    if self.progressCallback:
                        self.progressCallback(f"{action} stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (stopped)")
                    return False
                
                # Get current pose using robot instance
                actualPose = self.robot.getTcpPose()
                flangePose = self.robot.getFlangePose()
                
                # Track for stats
                if actualPose:
                    trackingTcpPoses.append(actualPose)
                    trackingTimestamps.append(currentTime - statsStartTime)
                if flangePose:
                    trackingFlangePoses.append(flangePose)
                
                # Find closest waypoint index for stop tracking
                if actualPose:
                    minDist = float('inf')
                    for i, wp in enumerate(waypoints):
                        dist = np.sqrt(sum((actualPose[j] - wp[j])**2 for j in range(3)))
                        if dist < minDist:
                            minDist = dist
                            self.traverseStopIndex = i
                    
                    # Pose callback
                    if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                        try:
                            self.poseCallback(actualPose)
                            lastPoseUpdate = currentTime
                        except Exception as e:
                            _log.warning("Pose callback failed: %s", e)
                
                # Force control (only during forward traversal)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(
                        forceLimit, forceAxis, direction,
                        force_frame=forceFrame, moment_limit=momentLimit, moment_axis=momentAxis)
                    if self.progressCallback and msg:
                        self.progressCallback(msg)
                    if exceeded:
                        self.rtde_c.stopL()
                        time.sleep(0.3)  # Allow robot to settle after stop
                        
                        # Update stop index based on actual stopped position
                        stoppedPose = self.robot.getTcpPose()
                        if stoppedPose:
                            minDist = float('inf')
                            for i, wp in enumerate(waypoints):
                                dist = np.sqrt(sum((stoppedPose[j] - wp[j])**2 for j in range(3)))
                                if dist < minDist:
                                    minDist = dist
                                    self.traverseStopIndex = i
                        
                        if self.progressCallback:
                            self.progressCallback(f"Force threshold exceeded! Stopped at index {self.traverseStopIndex}.")
                        self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (force stop)")
                        return False
                    lastForceCheck = currentTime
                
                time.sleep(0.05)  # 50ms poll interval
            
            # Final pose callback
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            # Print traverse statistics
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            
            return True
            
        except Exception as e:
            if self.progressCallback:
                self.progressCallback(f"Error during {action.lower()}: {str(e)}")
            return False
    
    # ==================== Private Servo Execution ====================
    
    def _executeServoPath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                          speed: float,
                          enableForceControl: bool = False,
                          forceLimit: Optional[float] = 10.0,
                          forceAxis: str = 'y',
                          direction: Optional[str] = None,
                          forceFrame: str = 'tcp',
                          momentLimit: Optional[float] = None,
                          momentAxis: Optional[str] = None,
                          isBackwardTraverse: bool = False) -> Tuple[bool]:
        """Execute servo path traversal with ramp-up and optional force control.
        
        Returns: (success, traveledWaypoints, stopIndex)
        """
        if waypoints is None or len(waypoints) == 0: return False
        
        # Get config values
        dt = CONFIG.traverse_servopath.dt
        lookaheadTime = CONFIG.traverse_servopath.lookahead_time
        gain = CONFIG.traverse_servopath.gain
        rampUpTime = CONFIG.traverse_servopath.ramp_up_time
        poseUpdateInterval = CONFIG.traverse.pose_update_interval
        forceCheckInterval = CONFIG.traverse.force_check_interval
        
        # Calculate timing
        waypointsDistance = calculateWaypointsDistance(waypoints)
        originalDuration = timestamps[-1] if len(timestamps) > 0 and timestamps[-1] > 0 else 1.0
        targetDuration = waypointsDistance / speed if waypointsDistance > 0 and speed > 0 else originalDuration
        targetDurationWithRamp = targetDuration + rampUpTime
        timeScale = targetDuration / originalDuration if originalDuration > 0 else 1.0
        scaledTimestamps = timestamps * timeScale
        
        action = "Backward traversal" if isBackwardTraverse else "Forward traversal"
        if self.progressCallback:
            self.progressCallback(f"{action} of {len(waypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s")
        
        traveledPoses = []
        
        # Stats tracking
        trackingTcpPoses = []
        trackingFlangePoses = []
        trackingTimestamps = []
        
        try:
            # Move to start if needed (check both position AND orientation)
            currentPose = self.robot.getTcpPose() if self.robot else None
            if currentPose is None or len(currentPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Error: No valid TCP pose for servo path start.")
                return False
            startPose = waypoints[0].tolist()

            # Position difference
            posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
            
            # Orientation difference (axis-angle magnitude)
            orientDiff = _orientation_diff_rad(currentPose, startPose)
            
            if posDiff > 0.001 or orientDiff > 0.01:  # 1mm position or ~0.5° orientation
                if self.progressCallback:
                    self.progressCallback(f"Moving to start (pos: {posDiff*1000:.1f}mm, orient: {np.degrees(orientDiff):.1f}°)...")
                self.rtde_c.moveL(startPose, speed * 0.5, 0.1)
                time.sleep(0.2)  # Allow settling time after moveL
            
            if not isBackwardTraverse:
                self.rtde_c.zeroFtSensor()
                time.sleep(0.1)
            
            # Re-read current pose after moveL for accurate starting point
            # This is used for blending during ramp-up to prevent initial jerk
            initialPose = np.array(self.rtde_r.getActualTCPPose())
            
            # Servo loop
            servoSpeed = speed * 2
            acceleration = 0.1
            
            startTime = time.time()
            lastPoseUpdate = 0
            lastForceCheck = 0
            currentIdx = 0
            
            while True:
                loopStart = time.time()
                wallElapsed = loopStart - startTime
                
                if wallElapsed >= targetDurationWithRamp:
                    break
                
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.servoStop()
                    if self.progressCallback:
                        self.progressCallback(f"{action} stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (stopped)")
                    return False
                
                # Ramp-up: sinusoidal easing for smooth start
                if wallElapsed < rampUpTime:
                    rampProgress = wallElapsed / rampUpTime
                    effectiveElapsed = rampUpTime * (rampProgress - (1/np.pi) * np.sin(np.pi * rampProgress)) * 0.5
                else:
                    effectiveElapsed = wallElapsed - rampUpTime * 0.5
                
                effectiveElapsed = max(0, min(effectiveElapsed, targetDuration))
                
                # Find current waypoint index
                while currentIdx < len(scaledTimestamps) - 1 and scaledTimestamps[currentIdx + 1] < effectiveElapsed:
                    currentIdx += 1
                
                # Interpolate target pose from waypoints
                if currentIdx >= len(waypoints) - 1:
                    waypointTarget = waypoints[-1]
                else:
                    t0, t1 = scaledTimestamps[currentIdx], scaledTimestamps[currentIdx + 1]
                    if t1 > t0:
                        alpha = np.clip((effectiveElapsed - t0) / (t1 - t0), 0.0, 1.0)
                        waypointTarget = waypoints[currentIdx] * (1 - alpha) + waypoints[currentIdx + 1] * alpha
                    else:
                        waypointTarget = waypoints[currentIdx]
                
                # Blend from initial pose to waypoint path during ramp-up
                # This prevents a sudden jump if the robot isn't exactly at waypoints[0]
                if wallElapsed < rampUpTime:
                    blendFactor = wallElapsed / rampUpTime  # 0 to 1 during ramp
                    # Smooth blending using sinusoidal easing
                    blendFactor = 0.5 * (1 - np.cos(np.pi * blendFactor))
                    targetPose = (initialPose * (1 - blendFactor) + waypointTarget * blendFactor).tolist()
                else:
                    targetPose = waypointTarget.tolist()
                
                self.traverseStopIndex = currentIdx
                self.rtde_c.servoL(targetPose, servoSpeed, acceleration, dt, lookaheadTime, gain)
                
                # Collect traveled pose and stats
                currentTime = time.time()
                try:
                    actualPose = self.robot.getTcpPose()
                    flangePose = self.robot.getFlangePose()
                    if actualPose:
                        if len(traveledPoses) == 0 or (currentTime - startTime - (len(traveledPoses) * 0.05)) >= 0.05:
                            traveledPoses.append(actualPose)
                        # Track for stats
                        trackingTcpPoses.append(actualPose)
                        trackingTimestamps.append(currentTime - startTime)
                    if flangePose:
                        trackingFlangePoses.append(flangePose)
                except Exception as e:
                    _log.warning("Traverse pose/state read failed: %s", e)

                # Pose callback
                if self.poseCallback and actualPose and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                    try:
                        self.poseCallback(actualPose)
                        lastPoseUpdate = currentTime
                    except Exception as e:
                        _log.warning("Pose callback failed: %s", e)
                
                # Force control (only during traversal, not retrace)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(
                        forceLimit, forceAxis, direction,
                        force_frame=forceFrame, moment_limit=momentLimit, moment_axis=momentAxis)
                    if self.progressCallback and msg:
                        self.progressCallback(msg)
                    if exceeded:
                        self.rtde_c.servoStop()
                        time.sleep(0.2)
                        # Snap stop index to actual TCP (like moveLPath). Time-based currentIdx can
                        # lag/ahead of pose; without this, backward retrace starts too far along the path
                        # and the first return commands look like moving further forward.
                        stoppedPose = self.robot.getTcpPose()
                        if stoppedPose:
                            closest_i = 0
                            min_dist = float("inf")
                            for i, wp in enumerate(waypoints):
                                dist = np.sqrt(
                                    sum((stoppedPose[j] - wp[j]) ** 2 for j in range(3))
                                )
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_i = i
                            self.traverseStopIndex = min(closest_i, currentIdx)
                        if self.progressCallback:
                            self.progressCallback(
                                f"Force threshold exceeded! Stopping at path index {self.traverseStopIndex}."
                            )
                        self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (force stop)")
                        return False
                    lastForceCheck = currentTime
                
                # Sleep to maintain timing
                elapsed = time.time() - loopStart
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            
            self.rtde_c.servoStop(2.0)
            time.sleep(0.1)
            
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            # Print traverse statistics
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            
            return True
            
        except Exception as e:
            try:
                self.rtde_c.servoStop()
            except Exception as stop_err:
                _log.debug("servoStop during cleanup: %s", stop_err)
            if self.progressCallback:
                self.progressCallback(f"Error during {action.lower()}: {e}")
            self.traverseStopIndex = -1
            return False
    
    # ==================== Force Mode Traverse Methods ====================
    
    def _executeForceHybridPath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                                 speed: float,
                                 acceleration: float = 0.5,
                                 blend: float = 0.01,
                                 isBackwardTraverse: bool = False) -> bool:
        """Execute path traversal with force mode Fz compliance + moveLPath.
        
        Uses hardware force mode to maintain Fz=0 while executing moveLPath.
        Good for paths that need to maintain contact with a surface.
        """
        if waypoints is None or len(waypoints) == 0:
            return False
        
        waypointsDistance = calculateWaypointsDistance(waypoints)
        action = "Backward traversal (forceHybrid)" if isBackwardTraverse else "Forward traversal (forceHybrid)"
        
        if self.progressCallback:
            self.progressCallback(f"{action}: {len(waypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s")
        
        # Stats tracking
        trackingTcpPoses = []
        trackingFlangePoses = []
        trackingTimestamps = []
        
        try:
            currentPose = self.robot.getTcpPose() if self.robot else None
            if currentPose is None or len(currentPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Error: No valid TCP pose for force hybrid traverse.")
                return False

            # Move to start if needed (forward traverse only)
            if not isBackwardTraverse:
                startPose = waypoints[0].tolist()
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = _orientation_diff_rad(currentPose, startPose)

                if posDiff > 0.001 or orientDiff > 0.01:
                    if self.progressCallback:
                        self.progressCallback(f"Moving to start...")
                    self.rtde_c.moveL(startPose, speed * 0.5, acceleration)
                    time.sleep(0.1)
                
                self.rtde_c.zeroFtSensor()
                time.sleep(0.1)
            
            # Force mode parameters (motion-specific hybrid config from path filename)
            hybrid_cfg = get_hybrid_config_for_path(self.filename)
            z_limit, damping, gain = _get_hybrid_force_params(hybrid_cfg)
            selection_vector = [0, 0, 1, 0, 0, 0]  # Only Z is force-controlled
            target_wrench = [0, 0, 0, 0, 0, 0]  # Fz = 0
            force_type = 2  # Force frame not transformed
            limits = [0.1, 0.1, z_limit, 0.5, 0.5, 0.5]
            
            # Configure force mode
            self.rtde_c.forceModeSetDamping(damping)
            self.rtde_c.forceModeSetGainScaling(gain)
            
            # Enter force mode
            initialPose = self.robot.getTcpPose()
            if initialPose is None or len(initialPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Could not get TCP pose for force mode")
                return False
            self.rtde_c.forceMode(initialPose, selection_vector, target_wrench, force_type, limits)
            
            if self.progressCallback:
                self.progressCallback("Force mode enabled (Fz compliance)")
            
            # Build path with parameters
            path = []
            for i, wp in enumerate(waypoints):
                pose = wp.tolist()
                wp_blend = 0.0 if i == 0 or i == len(waypoints) - 1 else blend
                path.append(pose + [speed, acceleration, wp_blend])
            
            # Execute path asynchronously
            success = self.rtde_c.moveL(path, asynchronous=True)
            
            if not success:
                self.rtde_c.forceModeStop()
                if self.progressCallback:
                    self.progressCallback(f"{action} failed to start")
                return False
            
            # Monitor progress
            poseUpdateInterval = CONFIG.traverse.pose_update_interval
            lastPoseUpdate = 0
            statsStartTime = time.time()
            
            while self.rtde_c.getAsyncOperationProgress() >= 0:
                currentTime = time.time()
                
                # Update task frame for force mode
                actualPose = self.robot.getTcpPose()
                if actualPose:
                    self.rtde_c.forceMode(actualPose, selection_vector, target_wrench, force_type, limits)
                    
                    # Track for stats
                    trackingTcpPoses.append(actualPose)
                    trackingTimestamps.append(currentTime - statsStartTime)
                    
                    flangePose = self.robot.getFlangePose()
                    if flangePose:
                        trackingFlangePoses.append(flangePose)
                    
                    # Find closest waypoint index
                    minDist = float('inf')
                    for i, wp in enumerate(waypoints):
                        dist = np.sqrt(sum((actualPose[j] - wp[j])**2 for j in range(3)))
                        if dist < minDist:
                            minDist = dist
                            self.traverseStopIndex = i
                    
                    # Pose callback
                    if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                        try:
                            self.poseCallback(actualPose)
                            lastPoseUpdate = currentTime
                        except Exception as e:
                            _log.warning("Pose callback failed: %s", e)
                
                # Check for stop request
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.stopL()
                    self.rtde_c.forceModeStop()
                    if self.progressCallback:
                        self.progressCallback(f"{action} stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (stopped)")
                    return False
                
                time.sleep(0.02)
            
            # Exit force mode
            self.rtde_c.forceModeStop()
            time.sleep(0.1)
            
            # Final pose callback
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            return True
            
        except Exception as e:
            try:
                self.rtde_c.forceModeStop()
            except Exception as stop_err:
                _log.debug("forceModeStop during cleanup: %s", stop_err)
            if self.progressCallback:
                self.progressCallback(f"Error during {action.lower()}: {str(e)}")
            return False
    
    def _executeForceSpeedLPath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                                 speed: float,
                                 isBackwardTraverse: bool = False) -> bool:
        """Execute path traversal with force mode Fz compliance + speedL control.
        
        Uses hardware force mode to maintain Fz=0 while using speedL for velocity control.
        Provides continuous velocity-based path following.
        """
        if waypoints is None or len(waypoints) == 0:
            return False
        
        waypointsDistance = calculateWaypointsDistance(waypoints)
        action = "Backward traversal (forceSpeedL)" if isBackwardTraverse else "Forward traversal (forceSpeedL)"
        
        if self.progressCallback:
            self.progressCallback(f"{action}: {len(waypoints)} waypoints, {waypointsDistance*1000:.1f}mm at {speed*1000:.1f}mm/s")
        
        # Stats tracking
        trackingTcpPoses = []
        trackingFlangePoses = []
        trackingTimestamps = []
        
        try:
            currentPose = self.robot.getTcpPose() if self.robot else None
            if currentPose is None or len(currentPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Error: No valid TCP pose for force servo traverse.")
                return False

            # Move to start if needed (forward traverse only)
            if not isBackwardTraverse:
                startPose = waypoints[0].tolist()
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = _orientation_diff_rad(currentPose, startPose)

                if posDiff > 0.001 or orientDiff > 0.01:
                    if self.progressCallback:
                        self.progressCallback(f"Moving to start...")
                    self.rtde_c.moveL(startPose, speed * 0.5, 0.5)
                    time.sleep(0.1)
                
                self.rtde_c.zeroFtSensor()
                time.sleep(0.1)
            
            # Force mode parameters (motion-specific hybrid config from path filename)
            hybrid_cfg = get_hybrid_config_for_path(self.filename)
            z_limit, damping, gain = _get_hybrid_force_params(hybrid_cfg)
            selection_vector = [0, 0, 1, 0, 0, 0]  # Only Z is force-controlled
            target_wrench = [0, 0, 0, 0, 0, 0]  # Fz = 0
            force_type = 2
            limits = [0.1, 0.1, z_limit, 0.5, 0.5, 0.5]
            
            # Configure force mode
            self.rtde_c.forceModeSetDamping(damping)
            self.rtde_c.forceModeSetGainScaling(gain)
            
            # Enter force mode
            initialPose = self.robot.getTcpPose()
            if initialPose is None or len(initialPose) < 6:
                if self.progressCallback:
                    self.progressCallback("Could not get TCP pose for force mode")
                return False
            self.rtde_c.forceMode(initialPose, selection_vector, target_wrench, force_type, limits)
            
            if self.progressCallback:
                self.progressCallback("Force mode enabled (Fz compliance + speedL)")
            
            # SpeedL control loop
            dt = CONFIG.traverse_servopath.dt
            acceleration = 0.5
            statsStartTime = time.time()
            currentIdx = 0
            poseUpdateInterval = CONFIG.traverse.pose_update_interval
            lastPoseUpdate = 0
            
            while currentIdx < len(waypoints) - 1:
                loopStart = time.time()
                
                # Check for stop request
                if self.stopCheck and self.stopCheck():
                    self.rtde_c.speedStop()
                    self.rtde_c.forceModeStop()
                    if self.progressCallback:
                        self.progressCallback(f"{action} stopped by user")
                    self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, f"{action} (stopped)")
                    return False
                
                # Get current pose
                actualPose = self.robot.getTcpPose()
                if not actualPose:
                    time.sleep(dt)
                    continue
                
                # Update task frame for force mode
                self.rtde_c.forceMode(actualPose, selection_vector, target_wrench, force_type, limits)
                
                # Track for stats
                currentTime = time.time()
                trackingTcpPoses.append(actualPose)
                trackingTimestamps.append(currentTime - statsStartTime)
                
                flangePose = self.robot.getFlangePose()
                if flangePose:
                    trackingFlangePoses.append(flangePose)
                
                # Calculate velocity toward next waypoint
                targetPose = waypoints[currentIdx + 1]
                diff = np.array(targetPose) - np.array(actualPose)
                
                # Position and orientation distances
                posDist = np.linalg.norm(diff[:3])
                
                # Check if we've reached the next waypoint
                if posDist < 0.002:  # 2mm threshold
                    currentIdx += 1
                    self.traverseStopIndex = currentIdx
                    continue
                
                # Calculate velocity (normalized direction * speed)
                posVel = (diff[:3] / posDist) * speed if posDist > 0 else np.zeros(3)
                
                # Orientation velocity: geodesic direction * angular speed (avoids axis-angle subtraction near π)
                orientAngle = _orientation_diff_rad(actualPose, targetPose)
                orientDir = _orientation_direction(actualPose, targetPose)
                orientVel = orientDir * min(orientAngle * 2.0, 0.5)
                
                # Construct speedL command [vx, vy, vz, ωx, ωy, ωz]
                speedLCmd = list(posVel) + list(orientVel)
                
                self.rtde_c.speedL(speedLCmd, acceleration, dt * 2)
                
                # Pose callback
                if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                    try:
                        self.poseCallback(actualPose)
                        lastPoseUpdate = currentTime
                    except Exception as e:
                        _log.warning("Pose callback failed: %s", e)

                # Sleep to maintain timing
                elapsed = time.time() - loopStart
                if elapsed < dt:
                    time.sleep(dt - elapsed)
            
            # Stop and exit force mode
            self.rtde_c.speedStop()
            self.rtde_c.forceModeStop()
            time.sleep(0.1)
            
            # Final pose callback
            if self.poseCallback:
                try:
                    finalPose = self.robot.getTcpPose()
                    if finalPose:
                        self.poseCallback(finalPose)
                except Exception as e:
                    _log.warning("Final pose callback failed: %s", e)

            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            return True
            
        except Exception as e:
            try:
                self.rtde_c.speedStop()
                self.rtde_c.forceModeStop()
            except Exception as stop_err:
                _log.debug("speedStop/forceModeStop during cleanup: %s", stop_err)
            if self.progressCallback:
                self.progressCallback(f"Error during {action.lower()}: {str(e)}")
            return False
    
    def _printTraverseStats(self, tcpPoses: List[List[float]], flangePoses: List[List[float]], 
                             timestamps: List[float], action: str) -> None:
        """Print traverse statistics for TCP and flange.
        
        Args:
            tcpPoses: List of TCP poses [x, y, z, rx, ry, rz]
            flangePoses: List of flange poses [x, y, z, rx, ry, rz]
            timestamps: List of timestamps for each pose
            action: Description of the action (e.g., "Forward traversal")
        """
        if len(tcpPoses) < 2 or len(timestamps) < 2:
            return
        
        def computeStats(poses: List[List[float]], timestamps: List[float], name: str) -> None:
            """Compute and print statistics for a set of poses."""
            poses_arr = np.array(poses)
            times_arr = np.array(timestamps)
            
            # Calculate distances between consecutive poses (position only)
            diffs = np.diff(poses_arr[:, :3], axis=0)
            segment_distances = np.linalg.norm(diffs, axis=1)
            total_distance = np.sum(segment_distances)
            
            # Calculate time differences
            dt = np.diff(times_arr)
            dt = np.maximum(dt, 1e-6)  # Avoid division by zero
            
            # Calculate velocities (m/s)
            velocities = segment_distances / dt
            
            # Filter out extreme outliers (> 10x median)
            median_vel = np.median(velocities)
            valid_mask = velocities < (median_vel * 10) if median_vel > 0 else np.ones(len(velocities), dtype=bool)
            valid_velocities = velocities[valid_mask]
            
            if len(valid_velocities) > 0:
                vel_min = np.min(valid_velocities)
                vel_max = np.max(valid_velocities)
                vel_mean = np.mean(valid_velocities)
            else:
                vel_min = vel_max = vel_mean = 0.0
            
            total_time = times_arr[-1] - times_arr[0]
            
            print(f"  {name}:")
            print(f"    Distance: {total_distance*1000:.2f} mm")
            print(f"    Time: {total_time:.2f} s")
            print(f"    Velocity: min={vel_min*1000:.2f}, max={vel_max*1000:.2f}, mean={vel_mean*1000:.2f} mm/s")
        
        print(f"\n=== {action} Statistics ===")
        print(f"Samples: {len(tcpPoses)}")
        computeStats(tcpPoses, timestamps, "TCP")
        if len(flangePoses) >= 2:
            computeStats(flangePoses, timestamps, "Flange")
        print()
    
    def _checkForceLimit(
        self,
        force_limit: Optional[float],
        force_axis: Optional[str],
        direction: Optional[str],
        *,
        force_frame: str = 'tcp',
        moment_limit: Optional[float] = None,
        moment_axis: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check traverse force/moment limits (aligned with movement code). Returns (exceeded, message).
        
        Args:
            force_limit: N for Fx/Fy, Nm for Mz (None skips that axis's check)
            force_axis: 'x', 'y', or 'mz'
            direction: 'left' or 'right' for direction-dependent checks
            force_frame: 'tcp' or 'ref' for translational x/y (flexion uses ref)
            moment_limit: TCP Mx/My limit (Nm); 0 or None disables
            moment_axis: 'mx' or 'my' (flexion)
        """
        try:
            tcp_force = self.robot.getTcpForceInTcpFrame()
            if tcp_force is None:
                tcp_force = [0.0] * 6

            fa = (force_axis or 'y').lower()
            ff = (force_frame or 'tcp').lower()
            ref_wrench: Optional[List[float]] = None

            def _ref_wrench() -> List[float]:
                nonlocal ref_wrench
                if ref_wrench is None:
                    rel = self.robot.getRefFrameRelativeTo()
                    ref_wrench = self.robot.getRefFrameForceInRefFrame(rel)
                    if ref_wrench is None:
                        ref_wrench = [0.0] * 6
                return ref_wrench

            status_parts: List[str] = []
            exceed_parts: List[str] = []
            exceeded_any = False

            # --- Mz (rotation / new_z): same directional rules as movement ---
            if fa == 'mz' and force_limit is not None and force_limit > 0:
                mz = tcp_force[5]
                if direction == 'left':
                    lim_disp = f"limit: > -{force_limit:.2f} Nm"
                    t_ex = mz < -force_limit
                elif direction == 'right':
                    lim_disp = f"limit: < {force_limit:.2f} Nm"
                    t_ex = mz > force_limit
                else:
                    lim_disp = f"limit: ±{force_limit:.2f} Nm"
                    t_ex = abs(mz) > force_limit
                status_parts.append(f"Mz: {mz:.2f} Nm ({lim_disp})")
                if t_ex:
                    exceeded_any = True
                    exceed_parts.append(f"Mz: {mz:.3f} Nm ({lim_disp})")

            # --- Translational Fx / Fy ---
            elif fa in ('x', 'y') and force_limit is not None:
                if ff == 'ref':
                    w = _ref_wrench()
                    f_idx = 0 if fa == 'x' else 1
                    force_value = w[f_idx]
                    tag = '(ref)'
                else:
                    f_idx = 0 if fa == 'x' else 1
                    force_value = tcp_force[f_idx]
                    tag = '(tcp)'
                flab = 'Fx' if fa == 'x' else 'Fy'
                t_ex = False
                if fa == 'x':
                    if direction == 'left' and force_value > force_limit:
                        t_ex = True
                    elif direction == 'right' and force_value < -force_limit:
                        t_ex = True
                    elif direction is None and abs(force_value) > force_limit:
                        t_ex = True
                    lim_disp = (
                        f"limit: < {force_limit:.2f} N" if direction == 'left'
                        else f"limit: > -{force_limit:.2f} N" if direction == 'right'
                        else f"limit: ±{force_limit:.2f} N")
                else:
                    if direction == 'left' and force_value < -force_limit:
                        t_ex = True
                    elif direction == 'right' and force_value > force_limit:
                        t_ex = True
                    elif direction is None and abs(force_value) > force_limit:
                        t_ex = True
                    lim_disp = (
                        f"limit: > -{force_limit:.2f} N" if direction == 'left'
                        else f"limit: < {force_limit:.2f} N" if direction == 'right'
                        else f"limit: ±{force_limit:.2f} N")
                status_parts.append(f"{flab}{tag}: {force_value:.2f} N ({lim_disp})")
                if t_ex:
                    exceeded_any = True
                    exceed_parts.append(f"{flab}{tag}: {force_value:.3f} N ({lim_disp})")

            # --- Flexion TCP moment Mx / My (alongside x/y force) ---
            ma = (moment_axis or '').lower()
            if ma in ('mx', 'my') and moment_limit is not None and moment_limit > 0:
                mi = 3 if ma == 'mx' else 4
                mval = tcp_force[mi]
                mlab = 'Mx' if ma == 'mx' else 'My'
                if direction == 'left':
                    lim_disp = f"limit: < {moment_limit:.2f} Nm"
                    m_ex = mval > moment_limit
                elif direction == 'right':
                    lim_disp = f"limit: > -{moment_limit:.2f} Nm"
                    m_ex = mval < -moment_limit
                else:
                    lim_disp = f"limit: ±{moment_limit:.2f} Nm"
                    m_ex = abs(mval) > moment_limit
                status_parts.append(f"{mlab}: {mval:.2f} Nm ({lim_disp})")
                if m_ex:
                    exceeded_any = True
                    exceed_parts.append(f"{mlab}: {mval:.3f} Nm ({lim_disp})")

            if exceeded_any:
                return True, "Limit exceeded! " + "; ".join(exceed_parts)
            if status_parts:
                return False, " | ".join(status_parts)
            return False, None
        except Exception as e:
            return False, f"Force check error: {e}"

    def eliminateWaypointJumps(self, jumpThresholdMultiplier: Optional[float] = None,
                                rotationWeight: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Eliminate large jumps in waypoints by interpolating additional points.
        
        Uses combined distance metric that includes both translation and rotation.
        
        Args:
            jumpThresholdMultiplier: Multiplier for average step size to determine jump threshold
            rotationWeight: Weight for rotation component (default: TCP offset magnitude)
        """
        if len(self.waypoints) < 2:
            return self.waypoints, self.timestamps
        
        jumpThresholdMultiplier = jumpThresholdMultiplier or CONFIG.traverse.jump_threshold_multiplier
        rotationWeight = rotationWeight if rotationWeight is not None else getDefaultRotationWeight()
        
        # Calculate combined step sizes (translation + weighted rotation)
        diffs = np.diff(self.waypoints, axis=0)
        trans_dist_sq = np.sum(diffs[:, :3]**2, axis=1)
        rot_dist_sq = np.sum(diffs[:, 3:]**2, axis=1) * (rotationWeight**2)
        stepSizes = np.sqrt(trans_dist_sq + rot_dist_sq)
        
        validSteps = stepSizes[stepSizes > 1e-6]
        
        if len(validSteps) == 0:
            return self.waypoints, self.timestamps
        
        avgStepSize = np.mean(validSteps)
        jumpThreshold = avgStepSize * jumpThresholdMultiplier
        
        smoothedPoses = [self.waypoints[0]]
        smoothedTimestamps = [self.timestamps[0]]
        
        for i in range(1, len(self.waypoints)):
            stepSize = stepSizes[i - 1]
            
            if stepSize > jumpThreshold:
                numIntermediate = max(2, int(np.ceil(stepSize / avgStepSize)))
                for j in range(1, numIntermediate):
                    alpha = j / numIntermediate
                    # Interpolate all 6 DOF (position and rotation)
                    smoothedPoses.append(self.waypoints[i - 1] * (1 - alpha) + self.waypoints[i] * alpha)
                    smoothedTimestamps.append(self.timestamps[i - 1] * (1 - alpha) + self.timestamps[i] * alpha)
            
            smoothedPoses.append(self.waypoints[i])
            smoothedTimestamps.append(self.timestamps[i])

        self.waypoints = np.array(smoothedPoses)
        self.timestamps = np.array(smoothedTimestamps)
        


# ==================== Utility Functions ====================

def getWaypointsDisplayName(filepath: str) -> str:
    """Get display name for a waypoints file (without .path.npz extension)."""
    filename = os.path.basename(filepath)
    if filename.endswith('.path.npz'):
        return filename[:-9]
    elif filename.endswith('.npz'):
        return filename[:-4]
    return filename


def deleteWaypointsFile(filepath: str) -> bool:
    """Delete a waypoints file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception as e:
        _log.warning("Could not delete waypoints file %s: %s", filepath, e)
    return False


def loadWaypoints(filepath: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load waypoints from .npz file. Returns (poses, timestamps) or (None, None)."""
    filepath = filepath or DEFAULT_WAYPOINTS_FILE
    if not os.path.exists(filepath):
        return None, None
    with np.load(filepath) as data:
        return np.array(data['poses']), np.array(data['timestamps'])





def smoothWaypoints(poses: np.ndarray, timestamps: np.ndarray,
                    smoothingWindow: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smoothing filter to waypoints. Smooths position only; orientation preserved.

    Linear smoothing of axis-angle is invalid (e.g. [0,0,0] and [0,0,2π] would average
    incorrectly). Position is smoothed; orientation comes from the original poses.
    """
    if len(poses) < smoothingWindow:
        return poses, timestamps

    smoothed = poses.copy()
    for i in range(3):
        smoothed[:, i] = uniform_filter1d(poses[:, i], size=smoothingWindow, mode='nearest')
    smoothed[0], smoothed[-1] = poses[0], poses[-1]
    return smoothed, timestamps


def interpolateWaypoints(poses: np.ndarray, timestamps: np.ndarray,
                         targetDistance: Optional[float] = None,
                         rotationWeight: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate waypoints for uniform spatial distance using cubic splines.
    
    This function resamples waypoints to achieve consistent inter-waypoint distances,
    accounting for both translational and rotational components of the pose.
    
    Args:
        poses: Array of poses (N x 6) with [tx, ty, tz, rx, ry, rz]
        timestamps: Array of timestamps (N,)
        targetDistance: Target distance between waypoints in meters (default: 0.001m = 1mm)
        rotationWeight: Characteristic length to convert rotation to equivalent distance.
                       A rotation of 1 radian is treated as rotationWeight meters.
                       Default: TCP offset magnitude (distance from flange to TCP).
    
    Returns:
        (interpolated_poses, interpolated_timestamps)
    """
    if len(poses) < 4:
        return poses, timestamps
    
    targetDistance = targetDistance or 0.001  # 1mm default spacing
    rotationWeight = rotationWeight if rotationWeight is not None else getDefaultRotationWeight()
    
    # Compute combined distance metric for each segment (geodesic rotation, not axis-angle Euclidean)
    # distance = sqrt(dx² + dy² + dz² + (L * geodesic_angle)²)
    segment_distances = np.zeros(len(poses) - 1)
    for j in range(len(poses) - 1):
        trans = np.linalg.norm(poses[j + 1, :3] - poses[j, :3])
        rot_rad = _orientation_diff_rad(poses[j].tolist(), poses[j + 1].tolist())
        segment_distances[j] = np.sqrt(trans**2 + (rot_rad * rotationWeight)**2)
    
    # Cumulative distance along path
    cumulative_dist = np.zeros(len(poses))
    cumulative_dist[1:] = np.cumsum(segment_distances)
    total_distance = cumulative_dist[-1]
    
    if total_distance < targetDistance:
        return poses, timestamps
    
    # Create new sample points at uniform distance intervals
    num_samples = max(4, int(np.ceil(total_distance / targetDistance)) + 1)
    new_distances = np.linspace(0, total_distance, num_samples)
    
    # Interpolate position with cubic spline (axis-angle cannot use linear/cubic spline)
    interpolated = np.zeros((num_samples, 6))
    for i in range(3):
        cs = CubicSpline(cumulative_dist, poses[:, i])
        interpolated[:, i] = cs(new_distances)
    
    # Interpolate orientation with SLERP (geodesic, avoids CubicSpline on axis-angle)
    key_rots = Rotation.from_rotvec(poses[:, 3:6])
    slerp = Slerp(cumulative_dist, key_rots)
    interp_rots = slerp(new_distances)
    interpolated[:, 3:6] = interp_rots.as_rotvec()
    
    # Interpolate timestamps based on distance (preserves velocity profile)
    ts_spline = CubicSpline(cumulative_dist, timestamps)
    new_timestamps = ts_spline(new_distances)
    
    # Ensure start and end points are exact
    interpolated[0] = poses[0]
    interpolated[-1] = poses[-1]
    new_timestamps[0] = timestamps[0]
    new_timestamps[-1] = timestamps[-1]
    
    return interpolated, new_timestamps


def calculateWaypointsDistance(poses: np.ndarray, rotationWeight: Optional[float] = None) -> float:
    """Calculate total combined distance of waypoints (translation + rotation).
    
    Args:
        poses: Array of poses (N x 6) with [tx, ty, tz, rx, ry, rz]
        rotationWeight: Characteristic length to convert rotation to equivalent distance.
                       Default: TCP offset magnitude (distance from flange to TCP).
    
    Returns:
        Total combined distance in meters
    """
    if poses is None or len(poses) < 2:
        return 0.0
    rotationWeight = rotationWeight if rotationWeight is not None else getDefaultRotationWeight()
    total = 0.0
    for j in range(len(poses) - 1):
        trans = np.linalg.norm(poses[j + 1, :3] - poses[j, :3])
        rot_rad = _orientation_diff_rad(poses[j].tolist(), poses[j + 1].tolist())
        total += np.sqrt(trans**2 + (rot_rad * rotationWeight)**2)
    return float(total)


def calculateWaypointsAngularVelocity(poses: np.ndarray, timestamps: np.ndarray) -> float:
    """Calculate mean angular velocity in rad/s."""
    if len(poses) < 2:
        return 0.0
    angles = np.sqrt(poses[:, 3]**2 + poses[:, 4]**2 + poses[:, 5]**2)
    totalTime = timestamps[-1] - timestamps[0]
    return abs(angles[-1] - angles[0]) / totalTime if totalTime > 0 else 0.0


def getReversedWaypoints(filepath: Optional[str] = None,
                         targetAngularVelocity: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load and reverse waypoints for return journey."""
    collector = WaypointCollector.load(None, filepath)
    if collector is None:
        return None, None
    
    waypoints = collector.getWaypoints()
    timestamps = collector.getTimestamps()
    if waypoints is None or timestamps is None or len(waypoints) == 0:
        return None, None

    reversedWaypoints = waypoints[::-1].copy()
    if len(timestamps) > 1:
        deltas = np.diff(timestamps)[::-1]
        reversedTimestamps = np.zeros(len(timestamps))
        reversedTimestamps[1:] = np.cumsum(deltas)
    else:
        reversedTimestamps = np.array([0.0])
    
    # Scale for target angular velocity
    if targetAngularVelocity is None:
        targetAngularVelocity = CONFIG.movement.return_angular_velocity
    
    targetRadS = np.radians(targetAngularVelocity)
    origVel = calculateWaypointsAngularVelocity(waypoints, timestamps)
    if origVel > 0:
        speedFactor = targetRadS / origVel
        reversedTimestamps = reversedTimestamps / speedFactor
    
    return reversedWaypoints, reversedTimestamps


def _tcpPoseToFlangePose(tcpPose: np.ndarray, tcpOffset: List[float]) -> np.ndarray:
    """Convert a single TCP pose to flange pose.
    
    Args:
        tcpPose: TCP pose [x, y, z, rx, ry, rz]
        tcpOffset: TCP offset [x, y, z, rx, ry, rz]
    
    Returns:
        Flange pose [x, y, z, rx, ry, rz]
    """
    tcpPos = tcpPose[:3]
    tcpRot = axis_angle_to_rotation_matrix(tcpPose[3], tcpPose[4], tcpPose[5])
    
    offsetPos = np.array(tcpOffset[:3])
    offsetRot = axis_angle_to_rotation_matrix(tcpOffset[3], tcpOffset[4], tcpOffset[5])
    
    # R_flange = R_tcp @ R_offset^T
    RFlangeToBase = tcpRot @ offsetRot.T
    
    # Flange_pos = TCP_pos - R_flange @ offset_pos
    flangePos = tcpPos - RFlangeToBase @ offsetPos
    
    # Convert rotation matrix back to axis-angle
    rx, ry, rz = rotation_matrix_to_axis_angle(RFlangeToBase)
    
    return np.array([flangePos[0], flangePos[1], flangePos[2], rx, ry, rz])


def _tcpPosesToFlangePoses(tcpPoses: np.ndarray, tcpOffset: Optional[List[float]] = None) -> np.ndarray:
    """Convert array of TCP poses to flange poses.
    
    Args:
        tcpPoses: Array of TCP poses (N x 6)
        tcpOffset: TCP offset [x, y, z, rx, ry, rz], defaults to runtime_tcp_offset
    
    Returns:
        Array of flange poses (N x 6)
    """
    tcpOffset = tcpOffset if tcpOffset is not None else runtime_tcp_offset
    flangePoses = np.zeros_like(tcpPoses)
    for i in range(len(tcpPoses)):
        flangePoses[i] = _tcpPoseToFlangePose(tcpPoses[i], tcpOffset)
    return flangePoses


def smoothWaypointsByFlange(poses: np.ndarray, timestamps: np.ndarray,
                            smoothingWindow: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smoothing filter to waypoints based on flange positions.
    
    Computes flange positions, applies smoothing, then uses the smoothed
    flange trajectory to resample the original TCP poses.
    
    Args:
        poses: Array of TCP poses (N x 6)
        timestamps: Array of timestamps (N,)
        smoothingWindow: Size of smoothing window
    
    Returns:
        (smoothed_poses, timestamps)
    """
    if len(poses) < smoothingWindow:
        return poses, timestamps
    
    # Convert to flange poses
    flangePoses = _tcpPosesToFlangePoses(poses)
    
    # Smooth position only; linear smoothing of axis-angle is invalid (e.g. [0,0,0] and [0,0,2π]
    # would average to ~π instead of ~0). Orientation is preserved from original flange.
    smoothedFlange = flangePoses.copy()
    for i in range(3):
        smoothedFlange[:, i] = uniform_filter1d(flangePoses[:, i], size=smoothingWindow, mode='nearest')
    smoothedFlange[0] = flangePoses[0]
    smoothedFlange[-1] = flangePoses[-1]
    
    # Apply position delta to TCP poses; orientation unchanged (no axis-angle smoothing)
    smoothedTcp = poses.copy()
    for i in range(len(poses)):
        positionDelta = smoothedFlange[i, :3] - flangePoses[i, :3]
        smoothedTcp[i, :3] = poses[i, :3] + positionDelta
    
    # Preserve exact endpoints
    smoothedTcp[0] = poses[0]
    smoothedTcp[-1] = poses[-1]
    
    return smoothedTcp, timestamps


def interpolateWaypointsByFlange(poses: np.ndarray, timestamps: np.ndarray,
                                  targetDistance: Optional[float] = None,
                                  rotationWeight: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate waypoints for uniform flange spatial distance.
    
    Computes flange positions and uses those for distance calculations,
    ensuring uniform spacing in flange space rather than TCP space.
    
    Args:
        poses: Array of TCP poses (N x 6)
        timestamps: Array of timestamps (N,)
        targetDistance: Target distance between waypoints in meters (default: 0.001m)
        rotationWeight: Weight for rotation component (default: TCP offset magnitude)
    
    Returns:
        (interpolated_poses, interpolated_timestamps)
    """
    if len(poses) < 4:
        return poses, timestamps
    
    targetDistance = targetDistance or 0.001
    rotationWeight = rotationWeight if rotationWeight is not None else getDefaultRotationWeight()
    
    # Convert to flange poses
    flangePoses = _tcpPosesToFlangePoses(poses)
    
    # Compute combined distance based on flange movement (geodesic rotation, not axis-angle Euclidean)
    segment_distances = np.zeros(len(flangePoses) - 1)
    for j in range(len(flangePoses) - 1):
        trans = np.linalg.norm(flangePoses[j + 1, :3] - flangePoses[j, :3])
        rot_rad = _orientation_diff_rad(flangePoses[j].tolist(), flangePoses[j + 1].tolist())
        segment_distances[j] = np.sqrt(trans**2 + (rot_rad * rotationWeight)**2)
    
    # Cumulative distance along path
    cumulative_dist = np.zeros(len(poses))
    cumulative_dist[1:] = np.cumsum(segment_distances)
    total_distance = cumulative_dist[-1]
    
    if total_distance < targetDistance:
        return poses, timestamps
    
    # Create new sample points at uniform distance intervals
    num_samples = max(4, int(np.ceil(total_distance / targetDistance)) + 1)
    new_distances = np.linspace(0, total_distance, num_samples)
    
    # Interpolate TCP position with cubic spline; orientation with SLERP (geodesic)
    interpolated = np.zeros((num_samples, 6))
    for i in range(3):
        cs = CubicSpline(cumulative_dist, poses[:, i])
        interpolated[:, i] = cs(new_distances)
    key_rots = Rotation.from_rotvec(poses[:, 3:6])
    slerp = Slerp(cumulative_dist, key_rots)
    interpolated[:, 3:6] = slerp(new_distances).as_rotvec()
    
    # Interpolate timestamps
    ts_spline = CubicSpline(cumulative_dist, timestamps)
    new_timestamps = ts_spline(new_distances)
    
    # Ensure start and end points are exact
    interpolated[0] = poses[0]
    interpolated[-1] = poses[-1]
    new_timestamps[0] = timestamps[0]
    new_timestamps[-1] = timestamps[-1]
    
    return interpolated, new_timestamps


def calculateFlangeDistance(poses: np.ndarray, rotationWeight: Optional[float] = None) -> float:
    """Calculate total combined distance based on flange movement.
    
    Args:
        poses: Array of TCP poses (N x 6)
        rotationWeight: Weight for rotation component
    
    Returns:
        Total flange distance in meters
    """
    if poses is None or len(poses) < 2:
        return 0.0

    rotationWeight = rotationWeight if rotationWeight is not None else getDefaultRotationWeight()
    flangePoses = _tcpPosesToFlangePoses(poses)
    total = 0.0
    for j in range(len(flangePoses) - 1):
        trans = np.linalg.norm(flangePoses[j + 1, :3] - flangePoses[j, :3])
        rot_rad = _orientation_diff_rad(flangePoses[j].tolist(), flangePoses[j + 1].tolist())
        total += np.sqrt(trans**2 + (rot_rad * rotationWeight)**2)
    return float(total)
