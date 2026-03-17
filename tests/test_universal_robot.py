"""UniversalRobot tests with mocked RTDE (connection, state, control, wrench)."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.objects.universal_robot import UniversalRobot
from src.utils import transform_wrench


# ---------------------------------------------------------------------------
# Connection and State
# ---------------------------------------------------------------------------

class TestConnection:
    """Connection tests with mocked RTDE."""

    def test_connect_with_mocked_rtde_succeeds(self, robot_with_mocked_rtde):
        """Connect succeeds when RTDE is available and isConnected returns True."""
        assert robot_with_mocked_rtde.isConnected() is True

    def test_connect_without_ip_returns_false(self):
        """Connect without IP returns False."""
        with patch("src.objects.universal_robot.RTDE_AVAILABLE", True):
            robot = UniversalRobot(ip=None)
            assert robot.connect() is False

    def test_connect_when_rtde_unavailable_returns_false(self):
        """Connect when RTDE not installed returns False."""
        with patch("src.objects.universal_robot.RTDE_AVAILABLE", False):
            robot = UniversalRobot(ip="192.168.1.100")
            assert robot.connect() is False

    def test_disconnect_during_freedrive_calls_end_freedrive(
        self, robot_with_mocked_rtde, mock_rtde_control
    ):
        """Disconnect when freedrive active calls endFreedriveMode."""
        robot_with_mocked_rtde._freedriveActive = True
        robot_with_mocked_rtde.disconnect()
        mock_rtde_control.endFreedriveMode.assert_called()

    def test_reconnect_disconnects_then_connects(
        self, robot_with_mocked_rtde, mock_rtde_receive, mock_rtde_control
    ):
        """Reconnect calls disconnect then connect."""
        with patch("src.objects.universal_robot.rtde_receive") as mock_r:
            with patch("src.objects.universal_robot.rtde_control") as mock_c:
                mock_r.RTDEReceiveInterface.return_value = mock_rtde_receive
                mock_c.RTDEControlInterface.return_value = mock_rtde_control
                result = robot_with_mocked_rtde.reconnect()
        assert result is True


class TestStateReading:
    """State reading with mocked RTDE."""

    def test_get_tcp_pose_propagates_from_rtde(self, robot_with_mocked_rtde, mock_rtde_receive):
        """getTcpPose returns value from RTDE."""
        expected = [0.35, -0.10, 0.08, 0.1, -0.2, 0.05]
        mock_rtde_receive.getActualTCPPose.return_value = expected
        result = robot_with_mocked_rtde.getTcpPose()
        assert result == expected

    def test_get_flange_pose_computed_from_tcp(
        self, robot_with_mocked_rtde, mock_rtde_receive
    ):
        """getFlangePose computes from TCP pose and offset."""
        tcp = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._tcpOffset = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        result = robot_with_mocked_rtde.getFlangePose()
        assert result is not None
        np.testing.assert_allclose(result[2], 0.05, rtol=1e-9)

    def test_get_joint_positions_propagates(self, robot_with_mocked_rtde, mock_rtde_receive):
        """getJointPositions returns value from RTDE."""
        joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        mock_rtde_receive.getActualQ.return_value = joints
        result = robot_with_mocked_rtde.getJointPositions()
        assert result == joints

    def test_get_tcp_force_propagates(self, robot_with_mocked_rtde, mock_rtde_receive):
        """getTcpForce returns value from RTDE."""
        wrench = [1.0, -2.0, 5.0, 0.1, 0.2, -0.3]
        mock_rtde_receive.getActualTCPForce.return_value = wrench
        result = robot_with_mocked_rtde.getTcpForce()
        assert result == wrench

    def test_is_protective_stopped(self, robot_with_mocked_rtde, mock_rtde_receive):
        """isProtectiveStopped returns RTDE value."""
        mock_rtde_receive.isProtectiveStopped.return_value = True
        assert robot_with_mocked_rtde.isProtectiveStopped() is True

    def test_is_emergency_stopped(self, robot_with_mocked_rtde, mock_rtde_receive):
        """isEmergencyStopped returns RTDE value."""
        mock_rtde_receive.isEmergencyStopped.return_value = True
        assert robot_with_mocked_rtde.isEmergencyStopped() is True


# ---------------------------------------------------------------------------
# Offset and Ref Frame
# ---------------------------------------------------------------------------

class TestTcpOffset:
    """TCP offset get/set."""

    def test_set_tcp_offset_invalid_length_returns_false(self, robot_disconnected):
        """setTcpOffset with wrong length returns False."""
        assert robot_disconnected.setTcpOffset([0, 0, 0]) is False
        assert robot_disconnected.setTcpOffset([0] * 7) is False

    def test_set_tcp_offset_valid_stores(self, robot_disconnected):
        """setTcpOffset with valid data stores offset."""
        offset = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        assert robot_disconnected.setTcpOffset(offset) is True
        result = robot_disconnected.getTcpOffset()
        assert result == offset

    def test_get_tcp_offset_none_when_never_set(self, robot_disconnected):
        """getTcpOffset returns None when never set."""
        # Robot with no tcpOffset in init
        robot = UniversalRobot(ip=None)
        result = robot.getTcpOffset()
        assert result is None


class TestRefFrame:
    """Ref frame offset and relative-to."""

    def test_set_ref_frame_offset_invalid_returns_false(self, robot_disconnected):
        """setRefFrameOffset with wrong length returns False."""
        assert robot_disconnected.setRefFrameOffset([0, 0, 0]) is False

    def test_set_ref_frame_offset_valid(self, robot_disconnected):
        """setRefFrameOffset with valid data stores."""
        offset = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert robot_disconnected.setRefFrameOffset(offset) is True
        assert robot_disconnected.getRefFrameOffset() == offset

    def test_set_ref_frame_relative_to_tcp_or_flange(self, robot_disconnected):
        """setRefFrameRelativeTo accepts tcp and flange."""
        robot_disconnected.setRefFrameRelativeTo("tcp")
        assert robot_disconnected.getRefFrameRelativeTo() == "tcp"
        robot_disconnected.setRefFrameRelativeTo("flange")
        assert robot_disconnected.getRefFrameRelativeTo() == "flange"

    def test_clear_ref_frame_offset(self, robot_disconnected):
        """clearRefFrameOffset resets to zero."""
        robot_disconnected.setRefFrameOffset([0.1, 0.1, 0.1, 0, 0, 0])
        robot_disconnected.clearRefFrameOffset()
        np.testing.assert_allclose(
            robot_disconnected.getRefFrameOffset(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


# ---------------------------------------------------------------------------
# Movement Control
# ---------------------------------------------------------------------------

class TestMovementControl:
    """Movement commands with mocked RTDE."""

    def test_move_l_calls_rtde(self, robot_with_mocked_rtde, mock_rtde_control):
        """moveL delegates to RTDE moveL."""
        pose = [0.3, -0.1, 0.05, 0.0, 0.0, 0.0]
        robot_with_mocked_rtde.moveL(pose, 0.1, 0.2)
        mock_rtde_control.moveL.assert_called_with(pose, 0.1, 0.2)

    def test_move_j_calls_rtde(self, robot_with_mocked_rtde, mock_rtde_control):
        """moveJ delegates to RTDE moveJ."""
        joints = [0.0] * 6
        robot_with_mocked_rtde.moveJ(joints, 1.0, 1.0)
        mock_rtde_control.moveJ.assert_called_with(joints, 1.0, 1.0)

    def test_move_l_path_builds_path_format(
        self, robot_with_mocked_rtde, mock_rtde_control
    ):
        """moveLPath appends speed, accel, blend to each pose."""
        path = [[0.3, 0, 0.05, 0, 0, 0], [0.35, 0, 0.05, 0, 0, 0]]
        robot_with_mocked_rtde.moveLPath(path, 0.1, 0.2, 0.01)
        call_args = mock_rtde_control.moveL.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == path[0] + [0.1, 0.2, 0.01]

    def test_stop_calls_stop_l(self, robot_with_mocked_rtde, mock_rtde_control):
        """stop delegates to stopL."""
        robot_with_mocked_rtde.stop(5.0)
        mock_rtde_control.stopL.assert_called_with(5.0)

    def test_stop_script_calls_rtde(self, robot_with_mocked_rtde, mock_rtde_control):
        """stopScript delegates to RTDE."""
        robot_with_mocked_rtde.stopScript()
        mock_rtde_control.stopScript.assert_called()

    def test_start_freedrive_calls_rtde(
        self, robot_with_mocked_rtde, mock_rtde_control
    ):
        """startFreedrive calls freedriveMode."""
        assert robot_with_mocked_rtde.startFreedrive() is True
        mock_rtde_control.freedriveMode.assert_called()

    def test_stop_freedrive_calls_rtde(
        self, robot_with_mocked_rtde, mock_rtde_control
    ):
        """stopFreedrive calls endFreedriveMode."""
        robot_with_mocked_rtde._freedriveActive = True
        assert robot_with_mocked_rtde.stopFreedrive() is True
        mock_rtde_control.endFreedriveMode.assert_called()

    def test_toggle_freedrive(self, robot_with_mocked_rtde, mock_rtde_control):
        """toggleFreedrive starts when off, stops when on."""
        robot_with_mocked_rtde._freedriveActive = False
        robot_with_mocked_rtde.toggleFreedrive()
        mock_rtde_control.freedriveMode.assert_called()
        robot_with_mocked_rtde._freedriveActive = True
        robot_with_mocked_rtde.toggleFreedrive()
        mock_rtde_control.endFreedriveMode.assert_called()

    def test_zero_ft_sensor_calls_rtde(
        self, robot_with_mocked_rtde, mock_rtde_control
    ):
        """zeroFtSensor delegates to RTDE."""
        assert robot_with_mocked_rtde.zeroFtSensor() is True
        mock_rtde_control.zeroFtSensor.assert_called()


# ---------------------------------------------------------------------------
# Wrench Transformations
# ---------------------------------------------------------------------------

# Parametrized wrench test data
WRENCH_TCP_FRAME_CASES = [
    ("identity_tcp_flange", [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
    ("offset_z_only", [0.0, 0.0, 0.15, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
    ("offset_x_y_z", [0.05, -0.03, 0.12, 0.0, 0.0, 0.0], [0.02, -0.01, 0.05, 0.0, 0.0, 0.0]),
]

WRENCH_FLANGE_ROTATION_CASES = [
    ("identity", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("rot_z_90", [0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]),
    ("rot_x_45", [0.0, 0.0, 0.0, np.pi / 4, 0.0, 0.0]),
]

WRENCH_REF_RELATIVE_TO = ["tcp", "flange"]


class TestWrenchTransformations:
    """Wrench transformation methods."""

    def test_get_tcp_force_in_tcp_frame_matches_transform_wrench(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base
    ):
        """getTcpForceInTcpFrame matches transform_wrench for same inputs."""
        tcp = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        flange = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._tcpOffset = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
        robot_with_mocked_rtde._flangePose = flange

        result = robot_with_mocked_rtde.getTcpForceInTcpFrame()
        expected = transform_wrench(tcp, wrench_base, flange)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("case_name,tcp,flange", WRENCH_TCP_FRAME_CASES)
    def test_get_tcp_force_in_tcp_frame_parametrized(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base, case_name, tcp, flange
    ):
        """getTcpForceInTcpFrame matches transform_wrench for multiple pose combos."""
        offset = [tcp[0] - flange[0], tcp[1] - flange[1], tcp[2] - flange[2], 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._tcpOffset = offset
        robot_with_mocked_rtde._flangePose = flange

        result = robot_with_mocked_rtde.getTcpForceInTcpFrame()
        expected = transform_wrench(tcp, wrench_base, flange)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)

    def test_get_flange_force_in_flange_frame_identity(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base
    ):
        """getFlangeForceInFlangeFrame with identity flange: base == flange frame."""
        flange = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        mock_rtde_receive.getActualTCPPose.return_value = flange
        robot_with_mocked_rtde._tcpPose = flange
        robot_with_mocked_rtde._tcpOffset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        robot_with_mocked_rtde._flangePose = flange

        result = robot_with_mocked_rtde.getFlangeForceInFlangeFrame()
        assert result is not None
        np.testing.assert_allclose(result[:3], wrench_base[:3], rtol=1e-9)
        np.testing.assert_allclose(result[3:6], wrench_base[3:6], rtol=1e-9)

    @pytest.mark.parametrize("rot_name,flange", WRENCH_FLANGE_ROTATION_CASES)
    def test_get_flange_force_in_flange_frame_rotated(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base, rot_name, flange
    ):
        """getFlangeForceInFlangeFrame rotates wrench to flange frame."""
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        mock_rtde_receive.getActualTCPPose.return_value = flange
        robot_with_mocked_rtde._tcpPose = flange
        robot_with_mocked_rtde._tcpOffset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        robot_with_mocked_rtde._flangePose = flange

        result = robot_with_mocked_rtde.getFlangeForceInFlangeFrame()
        assert result is not None
        assert len(result) == 6
        assert all(np.isfinite(r) for r in result)

    def test_get_ref_frame_force_relative_to_tcp(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base, ref_frame_offset_simple
    ):
        """getRefFrameForceInRefFrame with relative_to tcp."""
        tcp = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        flange = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._flangePose = flange
        robot_with_mocked_rtde.setRefFrameOffset(ref_frame_offset_simple)
        robot_with_mocked_rtde.setRefFrameRelativeTo("tcp")

        result = robot_with_mocked_rtde.getRefFrameForceInRefFrame("tcp")
        assert result is not None
        assert len(result) == 6

    def test_get_ref_frame_force_relative_to_flange(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base, ref_frame_offset_simple
    ):
        """getRefFrameForceInRefFrame with relative_to flange."""
        tcp = [0.15, 0.0, 0.05, 0.0, 0.0, 0.0]
        flange = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._flangePose = flange
        robot_with_mocked_rtde.setRefFrameOffset(ref_frame_offset_simple)
        robot_with_mocked_rtde.setRefFrameRelativeTo("flange")

        result = robot_with_mocked_rtde.getRefFrameForceInRefFrame("flange")
        assert result is not None
        assert len(result) == 6

    @pytest.mark.parametrize("relative_to", WRENCH_REF_RELATIVE_TO)
    def test_get_ref_frame_force_parametrized(
        self, robot_with_mocked_rtde, mock_rtde_receive, wrench_base,
        ref_frame_offset_simple, relative_to
    ):
        """getRefFrameForceInRefFrame for both tcp and flange parents."""
        tcp = [0.12, -0.03, 0.08, 0.0, 0.0, 0.0]
        flange = [0.08, -0.02, 0.05, 0.0, 0.0, 0.0]
        mock_rtde_receive.getActualTCPPose.return_value = tcp
        mock_rtde_receive.getActualTCPForce.return_value = wrench_base
        robot_with_mocked_rtde._tcpPose = tcp
        robot_with_mocked_rtde._flangePose = flange
        robot_with_mocked_rtde.setRefFrameOffset(ref_frame_offset_simple)
        robot_with_mocked_rtde.setRefFrameRelativeTo(relative_to)

        result = robot_with_mocked_rtde.getRefFrameForceInRefFrame(relative_to)
        assert result is not None
        assert len(result) == 6
        assert all(np.isfinite(r) for r in result)


# ---------------------------------------------------------------------------
# Edge Cases and Properties
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and properties."""

    def test_is_equal_to_tcp_pose_none_returns_false(self, robot_disconnected):
        """isEqualToTcpPose returns False when tcpPose is None."""
        assert robot_disconnected.isEqualToTcpPose([0, 0, 0, 0, 0, 0]) is False

    def test_is_equal_to_tcp_pose_custom_tolerance(self, robot_with_mocked_rtde):
        """isEqualToTcpPose respects custom tolerances."""
        pose = [0.35, -0.10, 0.08, 0.1, -0.2, 0.05]
        robot_with_mocked_rtde._tcpPose = pose
        assert robot_with_mocked_rtde.isEqualToTcpPose(
            pose, position_tolerance=0.001, orientation_tolerance=0.01
        ) is True

    def test_properties_return_copies(self, robot_with_mocked_rtde):
        """tcpPose, flangePose etc return copies."""
        robot_with_mocked_rtde._tcpPose = [0.1, 0, 0, 0, 0, 0]
        p1 = robot_with_mocked_rtde.tcpPose
        p2 = robot_with_mocked_rtde.tcpPose
        assert p1 is not p2
        assert p1 == p2

    def test_properties_return_none_when_empty(self, robot_disconnected):
        """Properties return None when state is None."""
        assert robot_disconnected.tcpPose is None
        assert robot_disconnected.flangePose is None
        assert robot_disconnected.jointPositions is None
        assert robot_disconnected.tcpForce is None

    def test_update_when_connected_returns_poses(self, robot_with_mocked_rtde):
        """update() returns TCP and flange poses when connected."""
        robot_with_mocked_rtde._tcpPose = [0.1, 0, 0, 0, 0, 0]
        robot_with_mocked_rtde._tcpOffset = [0, 0, 0.05, 0, 0, 0]
        tcp, flange = robot_with_mocked_rtde.update()
        assert tcp is not None
        assert flange is not None
