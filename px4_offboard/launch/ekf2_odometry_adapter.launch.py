"""Standalone launch for the PX4-01 EKF2 odometry adapter.

For bag replay, run with use_sim_time and play the bag so its recorded /clock
drives sim time (no --clock flag needed; the adapter stamps with the node clock):
  ros2 launch px4_offboard ekf2_odometry_adapter.launch.py
  ros2 bag play bags/slam_loop_02 \\
    --qos-profile-overrides-path bags/px4_qos_overrides.yaml

Runs independently of the EVAL-01 ground_truth_bridge; both can run at once.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_px4_timestamp = LaunchConfiguration('use_px4_timestamp')
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use sim time (true for bag replay via recorded /clock).'),
        DeclareLaunchArgument(
            'use_px4_timestamp', default_value='false',
            description='Stamp from PX4 msg.timestamp instead of the node clock. '
                        'Off by default: PX4 wall-clock us breaks sim-time sync.'),
        Node(
            package='px4_offboard',
            executable='ekf2_odometry_adapter',
            name='ekf2_odometry_adapter',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'use_px4_timestamp': use_px4_timestamp,
            }],
        ),
    ])
