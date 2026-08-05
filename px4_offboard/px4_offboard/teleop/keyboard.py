#!/usr/bin/env python3
"""Keyboard → /offboard_velocity_cmd (+ /arm_message).

Adapted from ARK-Electronics ROS2_PX4_Offboard_Example control.py.
Mode-2-ish mapping; rates are absolute (not ARK's accumulating stick), so
releasing keys zeroes motion (better for indoor SLAM flights).
"""

from __future__ import annotations

import sys
import termios
import tty

import geometry_msgs.msg
import rclpy
import std_msgs.msg
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

HELP = """
Keyboard velocity teleop (focus this terminal)
----------------------------------------------
  Arrow Up/Down/Left/Right : forward / back / left / right (body)
  W / S                    : up / down
  A / D                    : yaw left / right
  T / G                    : faster / slower linear
  Y / H                    : faster / slower yaw
  SPACE                    : arm toggle (velocity_control arms + takeoff)
  CTRL-C                   : quit (zero cmd + disarm request via active end)
"""

MOVE = {
    "w": (0.0, 0.0, 1.0, 0.0),
    "s": (0.0, 0.0, -1.0, 0.0),
    "a": (0.0, 0.0, 0.0, 1.0),
    "d": (0.0, 0.0, 0.0, -1.0),
    "\x1b[A": (1.0, 0.0, 0.0, 0.0),   # up arrow → forward
    "\x1b[B": (-1.0, 0.0, 0.0, 0.0),  # down arrow → back
    "\x1b[D": (0.0, 1.0, 0.0, 0.0),   # left arrow → left
    "\x1b[C": (0.0, -1.0, 0.0, 0.0),  # right arrow → right
}


def _get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    if key == "\x1b":
        key += sys.stdin.read(2)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)
    node = rclpy.create_node("velocity_teleop_keyboard")

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )
    cmd_pub = node.create_publisher(
        geometry_msgs.msg.Twist, "/offboard_velocity_cmd", qos)
    arm_pub = node.create_publisher(std_msgs.msg.Bool, "/arm_message", qos)

    speed = 0.6
    turn = 0.4
    armed = False

    print(HELP)
    print(f"speed={speed:.2f} turn={turn:.2f}")

    try:
        while True:
            key = _get_key(settings)
            if key == "\x03":
                break

            if key == " ":
                armed = not armed
                msg = std_msgs.msg.Bool()
                msg.data = armed
                arm_pub.publish(msg)
                print(f"arm_message = {armed}")
                continue

            if key in ("t", "T"):
                speed *= 1.1
                print(f"speed={speed:.2f} turn={turn:.2f}")
                continue
            if key in ("g", "G"):
                speed = max(0.05, speed * 0.9)
                print(f"speed={speed:.2f} turn={turn:.2f}")
                continue
            if key in ("y", "Y"):
                turn *= 1.1
                print(f"speed={speed:.2f} turn={turn:.2f}")
                continue
            if key in ("h", "H"):
                turn = max(0.05, turn * 0.9)
                print(f"speed={speed:.2f} turn={turn:.2f}")
                continue

            fx = fy = fz = yaw = 0.0
            if key in MOVE:
                fx, fy, fz, yaw = MOVE[key]

            twist = geometry_msgs.msg.Twist()
            # Body ENU-ish: x forward, y left, z up; angular.z yaw CCW
            twist.linear.x = fx * speed
            twist.linear.y = fy * speed
            twist.linear.z = fz * speed
            twist.angular.z = yaw * turn
            cmd_pub.publish(twist)
    finally:
        zero = geometry_msgs.msg.Twist()
        cmd_pub.publish(zero)
        off = std_msgs.msg.Bool()
        off.data = False
        arm_pub.publish(off)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
