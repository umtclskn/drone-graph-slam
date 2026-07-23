"""Launch the EVAL-06 independent ground-truth bridge (Gazebo model state).

Must run LIVE against PX4+Gazebo SITL -- gz-transport topics are not replayable
from a rosbag, so the bag records this node's /ground_truth/pose output:

  ros2 launch px4_offboard gazebo_truth_bridge.launch.py

Do not run this alongside px4_offboard's ground_truth_bridge (EVAL-01, EKF2
source): both publish /ground_truth/pose.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use sim time so stamps match the LiDAR scans.'),
        DeclareLaunchArgument(
            'gz_pose_topic', default_value='/world/my_slam_world/pose/info',
            description='gz.msgs.Pose_V topic from the scene broadcaster.'),
        DeclareLaunchArgument(
            'model_name', default_value='x500_lidar_down_0',
            description='Gazebo model name; PX4 appends the "_0" instance suffix.'),
        DeclareLaunchArgument(
            'link_name', default_value='base_link',
            description='Link whose world pose becomes ground truth.'),
    ]
    return LaunchDescription(args + [
        Node(
            package='px4_offboard',
            executable='gazebo_truth_bridge',
            name='gazebo_truth_bridge',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'gz_pose_topic': LaunchConfiguration('gz_pose_topic'),
                'model_name': LaunchConfiguration('model_name'),
                'link_name': LaunchConfiguration('link_name'),
            }],
        ),
    ])
