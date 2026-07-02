"""NDT-13 + INFRA-01: run the full NDT front-end pipeline and REP-105 TF tree.

Usage:
  # Replay the canonical bag WITHOUT --clock and let the node use sim time
  # (EVAL-01 clock-domain rule: PX4 wall-clock vs LiDAR sim-time stay in sync):
  ros2 launch graph_slam ndt_frontend.launch.py
  ros2 bag play bags/slam_loop_02

  ros2 launch graph_slam ndt_frontend.launch.py rviz:=true            # with RViz
  ros2 launch graph_slam ndt_frontend.launch.py publish_debug_clouds:=false
  ros2 launch graph_slam ndt_frontend.launch.py use_sim_time:=false   # live sim

Outputs: ~/ndt_odom (nav_msgs/Odometry, T_odom_base + Sigma_meas), odom->base_link TF,
plus optional ~/scan_raw and ~/scan_target debug clouds.

Static TF (launch file):
  map -> odom          identity stub until Sprint 3 GTSAM back-end owns map->odom
  base_link -> lidar_link   mount offset from slam_params.yaml tf.*
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_tf_mount(params_file: str) -> dict:
    with open(params_file, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    ros_params = cfg.get("/**", {}).get("ros__parameters", {})
    tf_cfg = ros_params.get("tf", {})
    return {
        "x": float(tf_cfg.get("lidar_to_base_x", 0.0)),
        "y": float(tf_cfg.get("lidar_to_base_y", 0.0)),
        "z": float(tf_cfg.get("lidar_to_base_z", 0.0)),
        "roll": float(tf_cfg.get("lidar_to_base_roll", 0.0)),
        "pitch": float(tf_cfg.get("lidar_to_base_pitch", 0.0)),
        "yaw": float(tf_cfg.get("lidar_to_base_yaw", 0.0)),
    }


def _launch_setup(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    mount = _load_tf_mount(params_file)
    use_sim_time = LaunchConfiguration("use_sim_time")

    nodes = [
        Node(
            package="graph_slam",
            executable="ndt_frontend_node",
            name="ndt_frontend",
            output="screen",
            parameters=[
                params_file,
                {
                    "use_sim_time": use_sim_time,
                    "lidar_topic": LaunchConfiguration("lidar_topic"),
                    "ekf2_topic": LaunchConfiguration("ekf2_topic"),
                    "publish_debug_clouds": LaunchConfiguration("publish_debug_clouds"),
                },
            ],
        ),
        # INFRA-01: map->odom identity stub (Sprint 3 GTSAM will publish corrections).
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        # INFRA-01: base_link->lidar_link static mount from YAML tf.* keys.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_lidar",
            arguments=[
                str(mount["x"]),
                str(mount["y"]),
                str(mount["z"]),
                str(mount["yaw"]),
                str(mount["pitch"]),
                str(mount["roll"]),
                "base_link",
                "lidar_link",
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]
    return nodes


def generate_launch_description():
    share = get_package_share_directory("graph_slam")
    default_params = os.path.join(share, "config", "slam_params.yaml")
    rviz_config = os.path.join(share, "rviz", "eval03_overlay.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("lidar_topic", default_value="/x500/lidar_3d/points"),
            DeclareLaunchArgument("ekf2_topic", default_value="/odometry/ekf2"),
            DeclareLaunchArgument("publish_debug_clouds", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("rviz_config", default_value=rviz_config),
            OpaqueFunction(function=_launch_setup),
        ]
    )
