"""Standalone launch for the EVAL-01 ground-truth bridge.

For bag replay, run with use_sim_time and play the bag with --clock:
  ros2 launch px4_offboard ground_truth_bridge.launch.py
  ros2 bag play bags/slam_loop_02 --clock \\
    --qos-profile-overrides-path bags/px4_qos_overrides.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use sim time (true for bag replay with --clock).'),
        Node(
            package='px4_offboard',
            executable='ground_truth_bridge',
            name='ground_truth_bridge',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
