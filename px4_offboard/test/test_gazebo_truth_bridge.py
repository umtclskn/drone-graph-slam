"""ROS-free unit tests for the EVAL-06 Gazebo ground-truth composition.

Exercises the pure helpers only (no node init/spin), the same way EVAL-01
tests ned_to_enu directly. The numbers come from a real
/world/my_slam_world/pose/info sample captured during SITL.
"""

import math

from gz.msgs10.pose_v_pb2 import Pose_V

from px4_offboard.gazebo_truth_bridge import _compose, _pose_parts, _unique_entry

TOL = 1e-9


def _close(a, b, tol=TOL):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def _add(msg, name, position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)):
    """Append a gz pose entry; orientation is (w, x, y, z)."""
    p = msg.pose.add()
    p.name = name
    p.position.x, p.position.y, p.position.z = position
    p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = orientation
    return p


def test_identity_model_pose_gives_link_offset():
    # Model at the origin: world truth is just the link's model-relative pose.
    position, orientation = _compose(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.24), (1.0, 0.0, 0.0, 0.0)))
    assert _close(position, (0.0, 0.0, 0.24))
    assert _close(orientation, (1.0, 0.0, 0.0, 0.0))


def test_link_offset_is_rotated_into_the_world():
    # Model yawed 90 deg at (1,2,0); a 1 m forward link offset must come out
    # along world +y, not +x. A plain addition would wrongly give (2,2,0).
    s = math.sqrt(0.5)
    position, orientation = _compose(
        ((1.0, 2.0, 0.0), (s, 0.0, 0.0, s)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    assert _close(position, (1.0, 3.0, 0.0), tol=1e-12)
    assert _close(orientation, (s, 0.0, 0.0, s))


def test_spawn_sample_reproduces_measured_altitude():
    # Real capture: model z = -0.013, base_link z = +0.24 relative to it.
    position, _ = _compose(
        ((0.0, 0.0, -0.013000183290526351), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.24), (1.0, 0.0, 0.0, 0.0)))
    assert abs(position[2] - 0.226999816709) < 1e-9


def test_unique_entry_selects_by_name():
    msg = Pose_V()
    _add(msg, 'wall_west', (-10.0, 0.0, 2.0))
    _add(msg, 'x500_lidar_down_0', (1.0, 2.0, 3.0))
    entry = _unique_entry(msg, 'x500_lidar_down_0')
    assert entry is not None
    assert _close(_pose_parts(entry)[0], (1.0, 2.0, 3.0))


def test_unique_entry_refuses_ambiguous_names():
    # my_slam_world really does contain several links called 'link'; picking
    # the first match would silently track a wall instead of the drone.
    msg = Pose_V()
    _add(msg, 'link', (1.0, 0.0, 0.0))
    _add(msg, 'link', (2.0, 0.0, 0.0))
    assert _unique_entry(msg, 'link') is None


def test_unique_entry_missing_name_is_none():
    msg = Pose_V()
    _add(msg, 'wall_west', (-10.0, 0.0, 2.0))
    assert _unique_entry(msg, 'x500_lidar_down_0') is None
