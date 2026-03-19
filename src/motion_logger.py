"""
Motion logger for recording robot pose and force data during motion execution.
Records flange pose, TCP pose, and forces at 10ms intervals.
"""
import logging
import os
import time
import threading
import csv
from datetime import datetime
from typing import Optional

from config import defaults as CONFIG

_log = logging.getLogger(__name__)



class MotionLogger:
    """
    Background logger that records robot motion data at 10ms intervals.
    
    Logs:
    - Elapsed time (s)
    - Absolute timestamp
    - Flange pose (x, y, z, rx, ry, rz)
    - TCP pose (x, y, z, rx, ry, rz)
    - TCP forces in base frame (fx, fy, fz, mx, my, mz)
    - TCP forces in local TCP frame (fx, fy, fz, mx, my, mz)
    """
    
    def __init__(self, robot, log_file_path: str):
        """
        Initialize the motion logger.
        
        Args:
            robot: UniversalRobot instance
            log_file_path: Path to the log file (e.g., "logs/mylog.log")
        """
        self.robot = robot
        self.log_file_path = log_file_path
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._file = None
        self._writer = None
        self._file_lock = threading.Lock()
    
    def start(self):
        """Start the background logging thread."""
        if self._thread is not None and self._thread.is_alive():
            print("Motion logger already running")
            return
        
        # Ensure logs directory exists
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self._stop_event.clear()
        self._start_time = time.time()
        
        # Open file and write header
        try:
            self._file = open(self.log_file_path, 'w', newline='')
            self._writer = csv.writer(self._file)

            # Write header
            header = [
                'elapsed_time_s',
                'absolute_timestamp',
                'flange_x', 'flange_y', 'flange_z', 'flange_rx', 'flange_ry', 'flange_rz',
                'tcp_x', 'tcp_y', 'tcp_z', 'tcp_rx', 'tcp_ry', 'tcp_rz',
                'force_base_fx', 'force_base_fy', 'force_base_fz',
                'torque_base_mx', 'torque_base_my', 'torque_base_mz',
                'force_tcp_fx', 'force_tcp_fy', 'force_tcp_fz',
                'torque_tcp_mx', 'torque_tcp_my', 'torque_tcp_mz'
            ]
            self._writer.writerow(header)
            self._file.flush()

            print(f"Motion logging started: {self.log_file_path}")
        except Exception as e:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception as e:
                    _log.debug("Error closing log file during start() error handling: %s", e)
                self._file = None
                self._writer = None
            print(f"Error opening log file: {e}")
            return

        # Start background thread
        self._thread = threading.Thread(target=self._logging_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the background logging thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        thread_still_alive = self._thread.is_alive()
        self._thread = None
        if thread_still_alive:
            print(f"Warning: Motion logging thread did not exit in time")

        # Close file only after acquiring lock (ensures no write in progress)
        with self._file_lock:
            if self._file is not None:
                try:
                    self._file.close()
                    print(f"Motion logging stopped: {self.log_file_path}")
                except Exception as e:
                    print(f"Error closing log file: {e}")
                self._file = None
                self._writer = None
    
    def _logging_loop(self):
        """Main logging loop running at 10ms intervals."""
        interval = 0.010  # 10ms
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Get current time
                elapsed_time = loop_start - self._start_time
                absolute_timestamp = datetime.now().isoformat(timespec='milliseconds')
                
                # Get TCP pose
                try:
                    tcp_pose_raw = self.robot.getTcpPose()
                    tcp_pose = list(tcp_pose_raw) if tcp_pose_raw else [0.0] * 6
                except Exception as e:
                    _log.debug("getTcpPose failed in motion logger: %s", e)
                    tcp_pose = [0.0] * 6

                # Calculate flange pose from TCP pose and offset
                try:
                    flange_pose_raw = self.robot.getFlangePose()
                    flange_pose = list(flange_pose_raw) if flange_pose_raw else [0.0] * 6
                except Exception as e:
                    _log.debug("getFlangePose failed in motion logger: %s", e)
                    flange_pose = [0.0] * 6
                
                # Get TCP forces (in base frame)
                try:
                    tcp_force_base = list(self.robot.getTcpForce())
                except Exception as e:
                    _log.debug("getTcpForce failed in motion logger: %s", e)
                    tcp_force_base = [0.0] * 6
                
                # Get forces in TCP local frame (using robot's method)
                try:
                    tcp_force_local = self.robot.getTcpForceInTcpFrame()
                    if tcp_force_local is None:
                        tcp_force_local = [0.0] * 6
                except Exception as e:
                    _log.debug("getTcpForceInTcpFrame failed in motion logger: %s", e)
                    tcp_force_local = [0.0] * 6
                
                # Write row
                row = [
                    f"{elapsed_time:.4f}",
                    absolute_timestamp,
                    *[f"{v:.6f}" for v in flange_pose],
                    *[f"{v:.6f}" for v in tcp_pose],
                    *[f"{v:.4f}" for v in tcp_force_base],
                    *[f"{v:.4f}" for v in tcp_force_local]
                ]
                
                if self._writer is not None:
                    with self._file_lock:
                        self._writer.writerow(row)
                        self._file.flush()

            except Exception as e:
                print(f"Error in motion logging: {e}")
            
            # Sleep for remaining time to maintain 10ms interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def is_running(self) -> bool:
        """Check if the logger is currently running."""
        return self._thread is not None and self._thread.is_alive()



def getLogfilePath(filename: str) -> str:
    """Get the path to the motion log file."""
    if filename == 'test':
        filename = "test.log"
    else:
        filename = f"{filename}.{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    return os.path.join(CONFIG.paths.logs_dir, filename)