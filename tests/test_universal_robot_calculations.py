"""Pure calculation tests for UniversalRobot (no RTDE, no robot)."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
from unittest.mock import patch

from src.objects.universal_robot import UniversalRobot, _isEqualPoses


# ---------------------------------------------------------------------------
# TCP/Flange offset parametrized data
# ---------------------------------------------------------------------------
TCP_OFFSET_TRANSLATION_CASES = [
    ("X_only", [0.05, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0.05),
    ("Y_only", [0.0, 0.05, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0, 0.0, 0.0], 1, 0.05),
    ("Z_only", [0.0, 0.0, 0.05, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0], 2, 0.05),
]

TCP_OFFSET_ROTATION_ONLY_CASES = [
    ("rot_Z_90", [0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]),
    ("rot_X_45", [0.0, 0.0, 0.0, np.pi / 4, 0.0, 0.0]),
    ("rot_Y_45", [0.0, 0.0, 0.0, 0.0, np.pi / 4, 0.0]),
    ("rot_combined", [0.0, 0.0, 0.0, 0.1, 0.05, -0.1]),
]


class TestCalculateFlangePoseFromTcp:
    """Tests for _calculateFlangePoseFromTcp."""

    def test_zero_offset_returns_tcp_as_flange(self, pose_identity, tcp_pose_full):
        """When offset is zero, flange pose equals TCP pose."""
        robot = UniversalRobot(tcpOffset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for tcp in [pose_identity, tcp_pose_full]:
            flange = robot._calculateFlangePoseFromTcp(tcp, [0.0] * 6)
            np.testing.assert_allclose(flange, tcp, rtol=1e-10, atol=1e-12)

    def test_simple_translation_offset(self, tcp_offset_simple):
        """Offset along Z only: flange = tcp - offset_in_base."""
        robot = UniversalRobot(tcpOffset=tcp_offset_simple)
        tcp = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        flange = robot._calculateFlangePoseFromTcp(tcp, tcp_offset_simple)
        np.testing.assert_allclose(flange[:3], [0, 0, 0.05], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(flange[3:6], [0, 0, 0], atol=1e-10)

    @pytest.mark.parametrize("axis_name,offset,tcp,axis_idx,expected_flange_pos", TCP_OFFSET_TRANSLATION_CASES)
    def test_translation_offset_along_axis(self, axis_name, offset, tcp, axis_idx, expected_flange_pos):
        """Offset along X, Y, or Z: flange position along that axis."""
        robot = UniversalRobot()
        flange = robot._calculateFlangePoseFromTcp(tcp, offset)
        # tcp[axis] - offset[axis] in base (identity flange) = expected
        np.testing.assert_allclose(flange[axis_idx], expected_flange_pos, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("rot_name,offset", TCP_OFFSET_ROTATION_ONLY_CASES)
    def test_rotation_only_offset_roundtrip(self, rot_name, offset):
        """Offset with rotation only (no translation): round-trip recovers offset."""
        robot = UniversalRobot()
        tcp = [0.3, -0.1, 0.05, 0.1, -0.2, 0.05]  # arbitrary pose
        flange = robot._calculateFlangePoseFromTcp(tcp, offset)
        recovered = robot._calculateTcpOffsetFromPoses(tcp, flange)
        np.testing.assert_allclose(recovered, offset, rtol=1e-9, atol=1e-10)

    def test_roundtrip_tcp_flange_offset(self, tcp_offset_with_rotation):
        """Round-trip: tcp + offset -> flange -> recover offset."""
        robot = UniversalRobot(tcpOffset=tcp_offset_with_rotation)
        tcp = [0.35, -0.12, 0.08, 0.1, -0.2, 0.05]
        flange = robot._calculateFlangePoseFromTcp(tcp, tcp_offset_with_rotation)
        recovered = robot._calculateTcpOffsetFromPoses(tcp, flange)
        np.testing.assert_allclose(recovered, tcp_offset_with_rotation, rtol=1e-9, atol=1e-10)

    def test_roundtrip_with_complex_poses(self, tcp_pose_full, flange_pose_full):
        """Round-trip with arbitrary poses and offsets."""
        robot = UniversalRobot()
        offset = [0.02, -0.01, 0.05, 0.1, 0.05, -0.1]
        flange = robot._calculateFlangePoseFromTcp(tcp_pose_full, offset)
        recovered = robot._calculateTcpOffsetFromPoses(tcp_pose_full, flange)
        np.testing.assert_allclose(recovered, offset, rtol=1e-9, atol=1e-10)

    def test_roundtrip_using_fixtures(self, tcp_pose_full, tcp_offset_simple, tcp_offset_with_rotation):
        """Round-trip with multiple offset types."""
        robot = UniversalRobot()
        for offset in [tcp_offset_simple, tcp_offset_with_rotation]:
            flange = robot._calculateFlangePoseFromTcp(tcp_pose_full, offset)
            recovered = robot._calculateTcpOffsetFromPoses(tcp_pose_full, flange)
            np.testing.assert_allclose(recovered, offset, rtol=1e-9, atol=1e-10)


class TestCalculateTcpOffsetFromPoses:
    """Tests for _calculateTcpOffsetFromPoses."""

    def test_identical_poses_zero_offset(self, pose_identity, tcp_pose_full):
        """When TCP = flange, offset is zero."""
        robot = UniversalRobot()
        for pose in [pose_identity, tcp_pose_full]:
            offset = robot._calculateTcpOffsetFromPoses(pose, pose)
            np.testing.assert_allclose(offset[:3], [0, 0, 0], atol=1e-10)
            np.testing.assert_allclose(offset[3:6], [0, 0, 0], atol=1e-10)

    @pytest.mark.parametrize("tcp_pos,flange_pos,expected_offset_pos", [
        ([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]),
        ([0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.1, 0.0]),
        ([0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ([0.05, -0.03, 0.02], [0.0, 0.0, 0.0], [0.05, -0.03, 0.02]),
    ])
    def test_offset_position_only(self, tcp_pos, flange_pos, expected_offset_pos):
        """Offset with position difference along X, Y, Z or combined."""
        robot = UniversalRobot()
        tcp = tcp_pos + [0.0, 0.0, 0.0]
        flange = flange_pos + [0.0, 0.0, 0.0]
        offset = robot._calculateTcpOffsetFromPoses(tcp, flange)
        np.testing.assert_allclose(offset[:3], expected_offset_pos, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(offset[3:6], [0, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Ref frame parametrized data
# ---------------------------------------------------------------------------
REF_PARENT_POSE_CASES = [
    ("tcp_parent", [0.3, -0.1, 0.05, 0.1, -0.2, 0.05]),
    ("flange_parent", [0.28, -0.09, 0.03, 0.08, -0.15, 0.04]),
    ("origin_identity", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]

REF_OFFSET_WITH_ROTATION_CASES = [
    ("translation_only", [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("translation_and_rot_Z", [0.05, 0.0, 0.0, 0.0, 0.0, np.pi / 2]),
    ("translation_and_rot_combined", [0.02, 0.01, 0.0, 0.1, 0.0, 0.0]),
]


class TestCalculateRefFramePose:
    """Tests for _calculateRefFramePose."""

    @pytest.mark.parametrize("parent_name,parent", REF_PARENT_POSE_CASES)
    def test_zero_offset_returns_parent(self, parent_name, parent):
        """Zero ref frame offset -> ref pose = parent pose."""
        robot = UniversalRobot()
        ref = robot._calculateRefFramePose(parent, [0.0] * 6)
        np.testing.assert_allclose(ref, parent, rtol=1e-10, atol=1e-12)

    def test_translation_offset_only(self, ref_frame_offset_simple):
        """Ref frame offset with translation only (parent at origin)."""
        robot = UniversalRobot()
        parent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ref = robot._calculateRefFramePose(parent, ref_frame_offset_simple)
        np.testing.assert_allclose(ref[:3], [0.02, 0.0, 0.0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ref[3:6], [0, 0, 0], atol=1e-10)

    @pytest.mark.parametrize("axis,offset,expected_ref_pos", [
        ("X", [0.05, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0]),
        ("Y", [0.0, 0.05, 0.0, 0.0, 0.0, 0.0], [0.0, 0.05, 0.0]),
        ("Z", [0.0, 0.0, 0.05, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05]),
    ])
    def test_translation_along_axis_parent_at_origin(self, axis, offset, expected_ref_pos):
        """Ref offset translation along X, Y, Z with parent at origin."""
        robot = UniversalRobot()
        parent = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ref = robot._calculateRefFramePose(parent, offset)
        np.testing.assert_allclose(ref[:3], expected_ref_pos, rtol=1e-10, atol=1e-12)

    def test_translation_with_parent_rotation(self):
        """Ref frame offset in rotated parent frame (parent X -> base +Y)."""
        robot = UniversalRobot()
        parent = [0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]
        ref_offset = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        ref = robot._calculateRefFramePose(parent, ref_offset)
        np.testing.assert_allclose(ref[:3], [0.0, 0.1, 0.0], rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("offset_name,ref_offset", REF_OFFSET_WITH_ROTATION_CASES)
    def test_ref_offset_translation_and_rotation(self, offset_name, ref_offset):
        """Ref frame with offset having translation and/or rotation."""
        robot = UniversalRobot()
        parent = [0.2, -0.05, 0.1, 0.1, 0.0, 0.0]
        ref = robot._calculateRefFramePose(parent, ref_offset)
        assert len(ref) == 6
        assert all(np.isfinite(r) for r in ref)

    def test_ref_from_tcp_vs_flange_same_offset(self):
        """Ref from TCP vs flange: different parents, same offset composition."""
        robot = UniversalRobot()
        offset = [0.02, 0.01, 0.0, 0.0, 0.0, 0.0]
        tcp = [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]
        flange = [0.28, 0.0, 0.08, 0.0, 0.0, 0.0]
        ref_from_tcp = robot._calculateRefFramePose(tcp, offset)
        ref_from_flange = robot._calculateRefFramePose(flange, offset)
        np.testing.assert_allclose(ref_from_tcp[:3], [0.32, 0.01, 0.1], rtol=1e-9)
        np.testing.assert_allclose(ref_from_flange[:3], [0.30, 0.01, 0.08], rtol=1e-9)


class TestIsEqualPoses:
    """Tests for _isEqualPoses (used by isEqualToTcpPose)."""

    @pytest.fixture(autouse=True)
    def _mock_config(self, mock_config):
        """Ensure config is mocked for these tests."""
        pass

    @pytest.mark.parametrize("pose", [
        [0.3, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.2, -0.1, 0.3, -0.4, 0.2],
    ])
    def test_same_pose_returns_true(self, pose):
        """Identical poses are equal."""
        assert _isEqualPoses(pose, pose) is True

    def test_none_returns_false(self):
        """None in either argument returns False."""
        pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _isEqualPoses(None, pose) is False
        assert _isEqualPoses(pose, None) is False
        assert _isEqualPoses(None, None) is False

    def test_within_position_tolerance(self):
        """Poses within position tolerance are equal."""
        p1 = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Default position tolerance 0.001, distance = 0.001, so equal
        assert _isEqualPoses(p1, p2, position_tolerance=0.002) is True

    def test_outside_position_tolerance(self):
        """Poses outside position tolerance are not equal."""
        p1 = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _isEqualPoses(p1, p2, position_tolerance=0.001) is False

    def test_within_orientation_tolerance(self):
        """Poses within orientation tolerance are equal."""
        p1 = [0.0, 0.0, 0.0, 0.01, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _isEqualPoses(p1, p2, orientation_tolerance=0.02) is True

    def test_outside_orientation_tolerance(self):
        """Poses outside orientation tolerance are not equal."""
        p1 = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert _isEqualPoses(p1, p2, orientation_tolerance=0.01) is False
