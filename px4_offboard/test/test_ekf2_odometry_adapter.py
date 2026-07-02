"""ROS-free unit tests for the PX4-01 EKF2 odometry adapter conversions.

These exercise the adapter's pure helper functions (no node init/spin), the
same way EVAL-01 tests ned_to_enu directly.
"""

from px4_offboard.ekf2_odometry_adapter import (
    enu_pose_covariance_diagonal,
    enu_twist_from_ned,
    px4_timestamp_to_ros_ns,
)

TOL = 1e-6


def _close(a, b):
    return all(abs(x - y) < TOL for x, y in zip(a, b))


def test_ned_velocity_north_maps_to_enu_y():
    # NED linear velocity (1,0,0) -> ENU (0,1,0): North becomes ENU +y.
    linear, _ = enu_twist_from_ned((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert _close(linear, (0.0, 1.0, 0.0))


def test_ned_down_position_velocity_maps_to_up():
    # NED (0,0,-1) -> ENU (0,0,1): Down becomes up, for the twist swap too.
    linear, _ = enu_twist_from_ned((0.0, 0.0, -1.0), (0.0, 0.0, 0.0))
    assert _close(linear, (0.0, 0.0, 1.0))


def test_px4_microseconds_to_ros_nanoseconds():
    # PX4 timestamps are microseconds; ROS Time wants nanoseconds (x1000).
    assert px4_timestamp_to_ros_ns(0) == 0
    assert px4_timestamp_to_ros_ns(1) == 1000
    assert px4_timestamp_to_ros_ns(1_781_100_454_333_690) == 1_781_100_454_333_690_000


def test_covariance_diagonal_swaps_xy_variance():
    # Position variance x/y are swapped by the NED->ENU axis relabel; the
    # diagonal lands in the nav order x, y, z, rx, ry, rz.
    cov = enu_pose_covariance_diagonal(
        position_variance=(0.1, 0.2, 0.3),
        orientation_variance=(0.4, 0.5, 0.6))
    assert cov[0] == 0.2    # x <- ned y
    assert cov[7] == 0.1    # y <- ned x
    assert cov[14] == 0.3   # z
    assert cov[21] == 0.4
    assert cov[28] == 0.5
    assert cov[35] == 0.6
    # Everything off the diagonal stays zero.
    assert sum(1 for i, v in enumerate(cov)
               if v != 0.0 and i not in (0, 7, 14, 21, 28, 35)) == 0
