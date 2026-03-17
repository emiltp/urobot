"""Pytest fixtures for UniversalRobot tests."""

import sys
import os

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Sample poses and wrenches (deterministic fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture
def pose_identity():
    """Identity pose: [0, 0, 0, 0, 0, 0] - origin, no rotation."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def pose_offset_z():
    """TCP pose with 0.1m offset along Z and identity rotation."""
    return [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]


@pytest.fixture
def tcp_offset_simple():
    """Simple TCP offset: 0.05m along flange Z, no rotation."""
    return [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]


@pytest.fixture
def tcp_offset_with_rotation():
    """TCP offset with rotation: 0.05m along Z + 90 deg around Z."""
    # 90 deg around Z in axis-angle: [0, 0, pi/2]
    return [0.0, 0.0, 0.05, 0.0, 0.0, np.pi / 2]


@pytest.fixture
def tcp_pose_full():
    """Arbitrary TCP pose: position + orientation."""
    return [0.34, -0.12, 0.08, 0.1, -0.2, 0.05]


@pytest.fixture
def flange_pose_full():
    """Arbitrary flange pose."""
    return [0.32, -0.11, 0.05, 0.08, -0.15, 0.03]


@pytest.fixture
def wrench_base():
    """Sample wrench in base frame: [Fx, Fy, Fz, Mx, My, Mz]."""
    return [1.0, -2.0, 5.0, 0.1, 0.2, -0.3]


@pytest.fixture
def ref_frame_offset_simple():
    """Simple ref frame offset: 0.02m along X."""
    return [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Mock config (avoids loading real config.yaml)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_config():
    """Patch config.defaults.tolerance for _isEqualPoses."""
    with patch("src.objects.universal_robot.CONFIG") as mock_cfg:
        mock_cfg.tolerance.position = 0.001
        mock_cfg.tolerance.orientation = 0.01
        yield mock_cfg


# ---------------------------------------------------------------------------
# UniversalRobot with mocked RTDE
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rtde_receive():
    """Mock RTDE receive interface."""
    m = MagicMock()
    m.isConnected.return_value = True
    m.getActualTCPPose.return_value = [0.35, -0.10, 0.08, 0.1, -0.2, 0.05]
    m.getActualQ.return_value = [0.0] * 6
    m.getActualTCPForce.return_value = [1.0, -2.0, 5.0, 0.1, 0.2, -0.3]
    m.isProtectiveStopped.return_value = False
    m.isEmergencyStopped.return_value = False
    return m


@pytest.fixture
def mock_rtde_control():
    """Mock RTDE control interface."""
    m = MagicMock()
    m.moveL.return_value = True
    m.moveJ.return_value = True
    m.moveL.side_effect = lambda *a, **kw: True
    m.moveJ.side_effect = lambda *a, **kw: True
    m.getTCPOffset.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return m


@pytest.fixture
def robot_with_mocked_rtde(mock_rtde_receive, mock_rtde_control):
    """UniversalRobot instance with RTDE interfaces mocked (no real connection)."""
    with patch("src.objects.universal_robot.RTDE_AVAILABLE", True):
        with patch("src.objects.universal_robot.rtde_receive") as mock_r_mod:
            with patch("src.objects.universal_robot.rtde_control") as mock_c_mod:
                mock_r_mod.RTDEReceiveInterface.return_value = mock_rtde_receive
                mock_c_mod.RTDEControlInterface.return_value = mock_rtde_control

                from src.objects.universal_robot import UniversalRobot

                robot = UniversalRobot(ip="192.168.1.100", tcpOffset=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
                success = robot.connect("192.168.1.100")

                if success:
                    yield robot
                else:
                    # If connect fails (e.g. isConnected returns False), inject mocks directly
                    robot._rtdeR = mock_rtde_receive
                    robot._rtdeC = mock_rtde_control
                    robot._connected = True
                    robot._tcpOffset = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
                    robot._tcpPose = list(mock_rtde_receive.getActualTCPPose())
                    robot._flangePose = robot._calculateFlangePoseFromTcp(
                        robot._tcpPose, robot._tcpOffset
                    )
                    yield robot


@pytest.fixture
def robot_disconnected():
    """UniversalRobot instance without connection (no RTDE)."""
    with patch("src.objects.universal_robot.RTDE_AVAILABLE", True):
        from src.objects.universal_robot import UniversalRobot

        return UniversalRobot(ip=None)
