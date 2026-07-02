"""EVAL-03 overlay: launch the registration-overlay node, optionally with RViz.

Usage:
  ros2 launch graph_slam eval03_overlay.launch.py                 # live, with RViz
  ros2 launch graph_slam eval03_overlay.launch.py self_test:=true # synthetic demo
  ros2 launch graph_slam eval03_overlay.launch.py rviz:=false     # headless

Then play a bag (e.g. `ros2 bag play bags/slam_loop_02`) and call:
  ros2 service call /eval03_overlay/capture std_srvs/srv/Trigger {}
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("graph_slam")
    rviz_config = os.path.join(share, "rviz", "eval03_overlay.rviz")
    default_params = os.path.join(share, "config", "slam_params.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            # INFRA-02: NDT-pipeline knobs from YAML; edit + re-launch, no rebuild.
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("self_test", default_value="false"),
            DeclareLaunchArgument("lidar_topic", default_value="/x500/lidar_3d/points"),
            DeclareLaunchArgument("plot_on_capture", default_value="true"),
            # Set frame_id to the frame your scans are stamped in. publish_tf adds a
            # static TF for it (needed for bag-only with no /tf); set false when a
            # live sim already publishes that frame.
            DeclareLaunchArgument("frame_id", default_value="lidar"),
            DeclareLaunchArgument("publish_tf", default_value="true"),
            Node(
                package="graph_slam",
                executable="eval03_overlay_node",
                name="eval03_overlay",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "self_test": LaunchConfiguration("self_test"),
                        "lidar_topic": LaunchConfiguration("lidar_topic"),
                        "plot_on_capture": LaunchConfiguration("plot_on_capture"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "publish_tf": LaunchConfiguration("publish_tf"),
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
