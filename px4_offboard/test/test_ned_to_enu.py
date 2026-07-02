"""ROS-free unit tests for the NED<->ENU conversion helpers (EVAL-01)."""

import math

from px4_offboard.ned_to_enu import (
    ned_to_enu_position,
    ned_to_enu_quaternion,
    rotate_vector,
)

TOL = 1e-6


def _close(a, b):
    return all(abs(x - y) < TOL for x, y in zip(a, b))


def test_north_maps_to_enu_y():
    # NED (1,0,0) -> ENU (0,1,0): North becomes ENU +y.
    assert _close(ned_to_enu_position(1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_down_maps_to_up():
    # NED (0,0,-1) -> ENU (0,0,1): one metre up.
    assert _close(ned_to_enu_position(0.0, 0.0, -1.0), (0.0, 0.0, 1.0))


def test_east_maps_to_enu_x():
    # NED (0,1,0) -> ENU (1,0,0): East becomes ENU +x.
    assert _close(ned_to_enu_position(0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def test_quaternion_norm_preserved():
    # Conversion is multiplication by unit quaternions, so a unit input
    # (here a 45 deg yaw in NED) stays unit after conversion.
    s = math.sqrt(0.5)
    q = ned_to_enu_quaternion(s, 0.0, 0.0, s)
    norm = math.sqrt(sum(c * c for c in q))
    assert abs(norm - 1.0) < TOL


def test_identity_orientation_points_forward_north():
    # PX4 identity (body-FRD aligned with NED). In ENU the body forward axis
    # must point North (ENU +y) and the body up axis must point up (ENU +z).
    q = ned_to_enu_quaternion(1.0, 0.0, 0.0, 0.0)
    assert _close(rotate_vector(q, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
    assert _close(rotate_vector(q, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))
