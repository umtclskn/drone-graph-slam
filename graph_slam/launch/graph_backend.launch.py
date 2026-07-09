"""GRAPH-VIS-01: full SLAM stack — NDT front-end + GTSAM graph back-end.

This launch file replaces the static map→odom stub from ndt_frontend.launch.py:
the graph_backend_node publishes a dynamic map→odom TF correction (updated on
each keyframe). Only the base_link→lidar_link static TF is kept here.

Usage:
  # Terminal 1 — EKF2 adapter (converts PX4 odom to /odometry/ekf2 in ENU):
  ros2 launch px4_offboard ekf2_odometry_adapter.launch.py

  # Terminal 2 — Full SLAM stack:
  ros2 launch graph_slam graph_backend.launch.py

  # Terminal 3 — Bag replay (no --clock; node clock mode):
  ros2 bag play bags/slam_loop_02 \\
      --qos-profile-overrides-path bags/px4_qos_overrides.yaml

  # Optionally open RViz2:
  ros2 launch graph_slam graph_backend.launch.py rviz:=true

Verification:
  ros2 topic hz /slam/graph_path         # must receive messages
  ros2 topic hz /slam/keyframes          # must receive messages
  ros2 run tf2_ros tf2_echo map odom     # must print a transform
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
        # NDT front-end: LiDAR → scan-to-scan NDT → /ndt_frontend/ndt_odom
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
        # GTSAM graph back-end: /ndt_frontend/ndt_odom → iSAM2 → /slam/* + map→odom TF
        Node(
            package="graph_slam",
            executable="graph_backend_node",
            name="graph_backend",
            output="screen",
            parameters=[
                params_file,
                {
                    "use_sim_time": use_sim_time,
                    "ndt_odom_topic": "/ndt_frontend/ndt_odom",
                    # Keyframe thresholds come from slam_params.yaml (keyframe_*),
                    # the single knob file — no launch-arg override layer.
                },
            ],
        ),
        # INFRA-01: base_link→lidar_link static mount from YAML tf.* keys.
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
