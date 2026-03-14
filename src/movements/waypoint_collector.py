"""
Waypoint collector for collecting and traversing robot trajectories.
Uses NumPy .npz format for fast, compact storage.
"""
import os
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline
from typing import Optional, List, Tuple, Callable

from config import defaults as CONFIG, runtime_tcp_offset, get_hybrid_config_for_path, _get_hybrid_force_params
from src.utils import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

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
        self.robot = async_motion_runner.robot  # UniversalRobot instance
        self.rtde_c = async_motion_runner.rtde_c
        self.rtde_r = async_motion_runner.rtde_r
        self.progressCallback = lambda msg: async_motion_runner.movement_progress.emit(msg)
        self.poseCallback = lambda pose: async_motion_runner.pose_updated.emit(pose)
        self.stopCheck = lambda: async_motion_runner._stop_requested
    
    
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
        
        data = np.load(filepath)
        collector = cls(async_motion_runner)
        collector.waypoints = data['poses']
        collector.timestamps = data['timestamps']
        collector.filename = filepath
        # Lists stay empty - use numpy arrays directly
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
                 direction: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray], int]:
        """Traverse waypoints with optional force control.
        
        Args:
            speed: Speed in m/s (user-determined, independent of collection)
            acceleration: Acceleration in m/s² (user-determined)
            blend: Blend radius in meters for smooth transitions
            traverseMethod: 'moveLPath', 'servoPath', or 'movePath'
            enableForceControl: Enable force control
            forceLimit: Force limit in N
            forceAxis: Force axis (x or y)
            direction: Direction (left or right)

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
                print(f"Force control: {forceAxis} axis, limit={forceLimit} N, direction={direction}")
            return self._executeServoPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
                isBackwardTraverse=False
            )
        elif traverseMethod == 'movePath':
            effectiveSpeed = speed if speed is not None else CONFIG.traverse_movepath.speed
            effectiveAccel = acceleration if acceleration is not None else CONFIG.traverse_movepath.acceleration
            print(f"Speed: {effectiveSpeed} m/s")
            print(f"Acceleration: {effectiveAccel} m/s²")
            if enableForceControl:
                print(f"Force control: {forceAxis} axis, limit={forceLimit} N, direction={direction}")
            return self._executeMovePath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
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
                print(f"Force control: {forceAxis} axis, limit={forceLimit} N, direction={direction}")
            return self._executeMoveLPath(
                waypoints, timestamps,
                speed=effectiveSpeed,
                acceleration=effectiveAccel,
                blend=effectiveBlend,
                enableForceControl=enableForceControl,
                forceLimit=forceLimit,
                forceAxis=forceAxis,
                direction=direction,
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
            currentPose = self.robot.getTcpPose()
            
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
                        except Exception:
                            pass
                
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
                except Exception:
                    pass
            
            if self.progressCallback:
                self.progressCallback("Force-compliant backward traverse completed")
            
            # Print stats
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, "Force-compliant backward traverse")
            
            return True
            
        except Exception as e:
            try:
                self.rtde_c.forceModeStop()
            except Exception:
                pass
            if self.progressCallback:
                self.progressCallback(f"Error during force-compliant backward traverse: {str(e)}")
            return False
    
    # ==================== Private Move Execution ====================
    
    def _executeMovePath(self, waypoints: np.ndarray, timestamps: np.ndarray,
                         speed: float,
                         acceleration: float = 0.1,
                         enableForceControl: bool = False,
                         forceLimit: float = 10.0,
                         forceAxis: str = 'y',
                         direction: Optional[str] = None,
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
            currentPose = self.robot.getTcpPose()
            
            # For forward traverse: move to start if needed
            # For backward traverse: skip "move to start" - just go from current position
            if not isBackwardTraverse and currentPose:
                startPose = waypoints[0].tolist()
                
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3, 6)))
                
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
                        except Exception:
                            pass
                
                # Force control (only during forward traversal)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(forceLimit, forceAxis, direction)
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
                except Exception:
                    pass
            
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
                          forceLimit: float = 10.0,
                          forceAxis: str = 'y',
                          direction: Optional[str] = None,
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
            currentPose = list(self.rtde_r.getActualTCPPose())
            
            # For forward traverse: move to start if needed
            if not isBackwardTraverse:
                startPose = waypoints[0].tolist()
                
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3, 6)))
                
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
                        except Exception:
                            pass
                
                # Force control (only during forward traversal)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(forceLimit, forceAxis, direction)
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
                except Exception:
                    pass
            
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
                          forceLimit: float = 10.0,
                          forceAxis: str = 'y',
                          direction: Optional[str] = None,
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
            currentPose = self.robot.getTcpPose()
            startPose = waypoints[0].tolist()
            
            # Position difference
            posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
            
            # Orientation difference (axis-angle magnitude)
            orientDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3, 6)))
            
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
                except Exception:
                    pass
                
                # Pose callback
                if self.poseCallback and actualPose and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                    try:
                        self.poseCallback(actualPose)
                        lastPoseUpdate = currentTime
                    except Exception:
                        pass
                
                # Force control (only during traversal, not retrace)
                if enableForceControl and not isBackwardTraverse and (currentTime - lastForceCheck) >= forceCheckInterval:
                    exceeded, msg = self._checkForceLimit(forceLimit, forceAxis, direction)
                    if self.progressCallback and msg:
                        self.progressCallback(msg)
                    if exceeded:
                        self.rtde_c.servoStop()
                        if self.progressCallback:
                            self.progressCallback("Force threshold exceeded! Stopping.")
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
                except Exception:
                    pass
            
            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            # Print traverse statistics
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            
            return True
            
        except Exception as e:
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass
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
            currentPose = self.robot.getTcpPose()
            
            # Move to start if needed (forward traverse only)
            if not isBackwardTraverse and currentPose:
                startPose = waypoints[0].tolist()
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3, 6)))
                
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
                        except Exception:
                            pass
                
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
                except Exception:
                    pass
            
            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            return True
            
        except Exception as e:
            try:
                self.rtde_c.forceModeStop()
            except Exception:
                pass
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
            currentPose = self.robot.getTcpPose()
            
            # Move to start if needed (forward traverse only)
            if not isBackwardTraverse and currentPose:
                startPose = waypoints[0].tolist()
                posDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3)))
                orientDiff = np.sqrt(sum((currentPose[i] - startPose[i])**2 for i in range(3, 6)))
                
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
                
                # Orientation velocity (simplified - proportional to difference)
                orientVel = diff[3:6] * 2.0  # Simple proportional control
                orientVel = np.clip(orientVel, -0.5, 0.5)
                
                # Construct speedL command [vx, vy, vz, ωx, ωy, ωz]
                speedLCmd = list(posVel) + list(orientVel)
                
                self.rtde_c.speedL(speedLCmd, acceleration, dt * 2)
                
                # Pose callback
                if self.poseCallback and (currentTime - lastPoseUpdate) >= poseUpdateInterval:
                    try:
                        self.poseCallback(actualPose)
                        lastPoseUpdate = currentTime
                    except Exception:
                        pass
                
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
                except Exception:
                    pass
            
            if self.progressCallback:
                self.progressCallback(f"{action} completed")
            
            self._printTraverseStats(trackingTcpPoses, trackingFlangePoses, trackingTimestamps, action)
            return True
            
        except Exception as e:
            try:
                self.rtde_c.speedStop()
                self.rtde_c.forceModeStop()
            except Exception:
                pass
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
    
    def _checkForceLimit(self, forceLimit: float, forceAxis: str,
                         direction: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Check if force limit is exceeded. Returns (exceeded, message).
        
        Args:
            forceLimit: Limit value (N for forces, Nm for torques)
            forceAxis: 'x' (Fx), 'y' (Fy), or 'mz' (Mz torque about TCP z)
            direction: 'left' or 'right' for direction-dependent sign checks
        """
        try:
            from src.utils import transform_wrench
            
            tcpWrench = self.rtde_r.getActualTCPForce()
            tcpPose = list(self.rtde_r.getActualTCPPose())
            tcpForce = transform_wrench(tcpPose, tcpWrench)
            
            if forceAxis == 'mz':
                forceValue = tcpForce[5]
                exceeded = abs(forceValue) > forceLimit
                return exceeded, f"Mz: {forceValue:.2f} Nm (limit: ±{forceLimit:.1f} Nm)"
            
            forceIndex = 0 if forceAxis == 'x' else 1
            forceLabel = 'Fx' if forceAxis == 'x' else 'Fy'
            forceValue = tcpForce[forceIndex]
            
            exceeded = False
            if forceAxis == 'x':
                if direction == 'left' and forceValue > forceLimit:
                    exceeded = True
                elif direction == 'right' and forceValue < -forceLimit:
                    exceeded = True
                elif direction is None and abs(forceValue) > forceLimit:
                    exceeded = True
            else:  # y-axis
                if direction == 'left' and forceValue < -forceLimit:
                    exceeded = True
                elif direction == 'right' and forceValue > forceLimit:
                    exceeded = True
                elif direction is None and abs(forceValue) > forceLimit:
                    exceeded = True
            
            return exceeded, f"{forceLabel}: {forceValue:.2f} N (limit: ±{forceLimit:.1f} N)"
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
    except Exception:
        pass
    return False


def loadWaypoints(filepath: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load waypoints from .npz file. Returns (poses, timestamps) or (None, None)."""
    filepath = filepath or DEFAULT_WAYPOINTS_FILE
    if not os.path.exists(filepath):
        return None, None
    data = np.load(filepath)
    return data['poses'], data['timestamps']





def smoothWaypoints(poses: np.ndarray, timestamps: np.ndarray,
                    smoothingWindow: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply smoothing filter to waypoints."""
    if len(poses) < smoothingWindow:
        return poses, timestamps
    
    smoothed = np.zeros_like(poses)
    for i in range(6):
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
    
    # Compute combined distance metric for each segment
    # distance = sqrt(dx² + dy² + dz² + L²*(drx² + dry² + drz²))
    # where L = rotationWeight converts radians to equivalent meters
    diffs = np.diff(poses, axis=0)
    trans_dist_sq = np.sum(diffs[:, :3]**2, axis=1)  # translation component
    rot_dist_sq = np.sum(diffs[:, 3:]**2, axis=1) * (rotationWeight**2)  # rotation component
    segment_distances = np.sqrt(trans_dist_sq + rot_dist_sq)
    
    # Cumulative distance along path
    cumulative_dist = np.zeros(len(poses))
    cumulative_dist[1:] = np.cumsum(segment_distances)
    total_distance = cumulative_dist[-1]
    
    if total_distance < targetDistance:
        return poses, timestamps
    
    # Create new sample points at uniform distance intervals
    num_samples = max(4, int(np.ceil(total_distance / targetDistance)) + 1)
    new_distances = np.linspace(0, total_distance, num_samples)
    
    # Interpolate poses at new distance points
    interpolated = np.zeros((num_samples, 6))
    for i in range(6):
        cs = CubicSpline(cumulative_dist, poses[:, i])
        interpolated[:, i] = cs(new_distances)
    
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
    diffs = np.diff(poses, axis=0)
    trans_dist_sq = np.sum(diffs[:, :3]**2, axis=1)
    rot_dist_sq = np.sum(diffs[:, 3:]**2, axis=1) * (rotationWeight**2)
    return float(np.sum(np.sqrt(trans_dist_sq + rot_dist_sq)))


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
    
    # Smooth the flange poses
    smoothedFlange = np.zeros_like(flangePoses)
    for i in range(6):
        smoothedFlange[:, i] = uniform_filter1d(flangePoses[:, i], size=smoothingWindow, mode='nearest')
    
    # Preserve endpoints
    smoothedFlange[0] = flangePoses[0]
    smoothedFlange[-1] = flangePoses[-1]
    
    # Compute how much each flange pose changed, use that to adjust TCP poses
    # Delta = smoothed_flange - original_flange
    # Apply same delta to TCP poses (for position, orientation handled separately)
    smoothedTcp = poses.copy()
    for i in range(len(poses)):
        # Position delta
        positionDelta = smoothedFlange[i, :3] - flangePoses[i, :3]
        smoothedTcp[i, :3] = poses[i, :3] + positionDelta
        
        # For orientation, we apply the smoothed flange orientation change
        # by computing the relative rotation and applying it to TCP
        origFlangeRot = axis_angle_to_rotation_matrix(flangePoses[i, 3], flangePoses[i, 4], flangePoses[i, 5])
        smoothFlangeRot = axis_angle_to_rotation_matrix(smoothedFlange[i, 3], smoothedFlange[i, 4], smoothedFlange[i, 5])
        
        # Relative rotation: R_smooth @ R_orig^T
        relRot = smoothFlangeRot @ origFlangeRot.T
        
        # Apply to TCP orientation
        origTcpRot = axis_angle_to_rotation_matrix(poses[i, 3], poses[i, 4], poses[i, 5])
        newTcpRot = relRot @ origTcpRot
        rx, ry, rz = rotation_matrix_to_axis_angle(newTcpRot)
        smoothedTcp[i, 3:] = [rx, ry, rz]
    
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
    
    # Compute combined distance based on flange movement
    diffs = np.diff(flangePoses, axis=0)
    trans_dist_sq = np.sum(diffs[:, :3]**2, axis=1)
    rot_dist_sq = np.sum(diffs[:, 3:]**2, axis=1) * (rotationWeight**2)
    segment_distances = np.sqrt(trans_dist_sq + rot_dist_sq)
    
    # Cumulative distance along path
    cumulative_dist = np.zeros(len(poses))
    cumulative_dist[1:] = np.cumsum(segment_distances)
    total_distance = cumulative_dist[-1]
    
    if total_distance < targetDistance:
        return poses, timestamps
    
    # Create new sample points at uniform distance intervals
    num_samples = max(4, int(np.ceil(total_distance / targetDistance)) + 1)
    new_distances = np.linspace(0, total_distance, num_samples)
    
    # Interpolate TCP poses at new distance points (using flange-based distances)
    interpolated = np.zeros((num_samples, 6))
    for i in range(6):
        cs = CubicSpline(cumulative_dist, poses[:, i])
        interpolated[:, i] = cs(new_distances)
    
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
    
    diffs = np.diff(flangePoses, axis=0)
    trans_dist_sq = np.sum(diffs[:, :3]**2, axis=1)
    rot_dist_sq = np.sum(diffs[:, 3:]**2, axis=1) * (rotationWeight**2)
    return float(np.sum(np.sqrt(trans_dist_sq + rot_dist_sq)))
