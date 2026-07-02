"""EVAL-02 per-pair relative-error node, with INFRA-02 YAML params.

The NDT-pipeline knobs come from config/slam_params.yaml; edit that file and
re-launch to change behaviour WITHOUT a rebuild. Run-control args (csv_path,
stride, use_sim_time, ...) stay on the command line.

Usage:
  ros2 launch graph_slam eval02_relative_error.launch.py \\
      csv_path:=analysis/eval02_pairs.csv stride:=6
  # then, in another shell:
  ros2 bag play bags/slam_loop_02 \\
      --qos-profile-overrides-path bags/px4_qos_overrides.yaml
  # override the param file with your own:
  ros2 launch graph_slam eval02_relative_error.launch.py params_file:=/path/to.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("graph_slam"), "config", "slam_params.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("csv_path", default_value="analysis/eval02_pairs.csv"),
            DeclareLaunchArgument("stride", default_value="1"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="graph_slam",
                executable="eval02_relative_error_node",
                name="eval02_relative_error",
                output="screen",
                # YAML first, then run-control overrides on top.
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "csv_path": LaunchConfiguration("csv_path"),
                        "stride": LaunchConfiguration("stride"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
            ),
        ]
    )
