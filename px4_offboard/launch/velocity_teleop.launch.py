#!/usr/bin/env python3
"""Launch the Offboard velocity bridge only.

Run the keyboard in a dedicated terminal (needs stdin):
  ros2 run px4_offboard velocity_teleop_keyboard
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="px4_offboard",
            executable="velocity_teleop",
            name="velocity_teleop",
            output="screen",
        ),
    ])
