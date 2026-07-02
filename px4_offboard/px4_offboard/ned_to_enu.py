"""NED <-> ENU frame conversion helpers (ROS-free).

PX4 reports pose in a North-East-Down (NED) world frame with a
Forward-Right-Down (FRD) body frame; ROS / REP-103 uses an East-North-Up
(ENU) world with a Forward-Left-Up (FLU) body. These free functions are the
single place that conversion happens for the px4_offboard ground-truth bridge.

Position:    x_enu = y_ned,  y_enu = x_ned,  z_enu = -z_ned
Orientation: q_enu = Q_NED_ENU * q_px4 * Q_FRD_FLU, using the two static
rotations below. The body rotation (180 deg about forward x) is its own
inverse, so the same quaternion converts FRD->FLU and FLU->FRD.

Quaternions are (w, x, y, z) tuples, matching px4_msgs/VehicleOdometry.q.
"""

import math

# 180 deg rotation about the (1, 1, 0)/sqrt(2) axis: swaps x/y, negates z.
# As a rotation matrix this is exactly [[0,1,0],[1,0,0],[0,0,-1]], i.e. the
# position rule above. Applied to a vector expressed in NED it yields ENU.
_Q_NED_ENU = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)  # (w, x, y, z)
# 180 deg rotation about body forward (x): FRD <-> FLU. Self-inverse.
_Q_FRD_FLU = (0.0, 1.0, 0.0, 0.0)


def ned_to_enu_position(x, y, z):
    """Convert an NED position/vector to ENU. Returns (x, y, z)."""
    return (y, x, -z)


def quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q):
    """Conjugate (inverse rotation for a unit quaternion) of (w, x, y, z)."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def ned_to_enu_quaternion(w, x, y, z):
    """Convert an NED/FRD body orientation to ENU/FLU. Returns (w, x, y, z)."""
    return quat_mul(quat_mul(_Q_NED_ENU, (w, x, y, z)), _Q_FRD_FLU)


def rotate_vector(q, v):
    """Rotate 3-vector v by unit quaternion q (w, x, y, z). Returns (x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
