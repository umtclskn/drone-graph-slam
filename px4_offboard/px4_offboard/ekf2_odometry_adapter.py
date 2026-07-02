"""PX4 EKF2 VehicleOdometry (NED/FRD) -> /odometry/ekf2 (nav_msgs, ENU).

PX4-01. Translates /fmu/out/vehicle_odometry (px4_msgs, NED world / FRD body)
into the standard nav_msgs/Odometry on /odometry/ekf2 that graph_slam expects
as the NDT-08 registration prior (NDT-13 consumes pose.pose). graph_slam never
sees px4_msgs -- this node is the boundary (ARCHITECTURE.md sec 3/11). Runs
fully standalone, alongside the EVAL-01 ground_truth_bridge, sharing nothing.

Every NED->ENU conversion is delegated to px4_offboard.ned_to_enu (untouched).

Stamp domain (the EVAL-01 lesson, verified again here): PX4 message timestamps
are wall-clock microseconds (~1.78e9 s on slam_loop_02), a different clock from
the sim-time LiDAR scans (~22 s). Stamping the output with msg.timestamp would
put /odometry/ekf2 ~1.78e9 s away from /x500/lidar_3d/points and blow the 50 ms
sync budget (criterion 4). So the default stamp is the node clock (sim time
under use_sim_time, fed by the bag's recorded /clock). The microsecond->ROS
conversion is still provided and unit-tested, and selectable via the
``use_px4_timestamp`` parameter for setups whose PX4 and ROS clocks share a
domain (e.g. a live system); it is off by default precisely because of the
mismatch above.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from nav_msgs.msg import Odometry

from px4_msgs.msg import VehicleOdometry

from px4_offboard.ned_to_enu import (
    ned_to_enu_position,
    ned_to_enu_quaternion,
)

ODOM_FRAME = 'odom'
BASE_FRAME = 'base_link'


def px4_timestamp_to_ros_ns(timestamp_us):
    """Convert a PX4 timestamp (microseconds, uint64) to ROS nanoseconds."""
    return int(timestamp_us) * 1000


def enu_twist_from_ned(velocity, angular_velocity):
    """Map NED linear/angular rates to an ENU twist via the position swap.

    Returns ((lx, ly, lz), (ax, ay, az)). NDT-13 consumes only the pose, so
    twist is best-effort world-ENU (no body-frame rotation) -- enough until a
    consumer needs body-frame rates.
    """
    linear = ned_to_enu_position(
        float(velocity[0]), float(velocity[1]), float(velocity[2]))
    angular = ned_to_enu_position(
        float(angular_velocity[0]), float(angular_velocity[1]),
        float(angular_velocity[2]))
    return linear, angular


def enu_pose_covariance_diagonal(position_variance, orientation_variance):
    """Build the 6x6 nav covariance (row-major) diagonal from PX4 variances.

    Order is x, y, z, rx, ry, rz; the NED->ENU position swap relabels the x/y
    variances (signs do not affect a variance). Missing variances stay zero.
    """
    cov = [0.0] * 36
    cov[0] = float(position_variance[1])   # x <- ned y
    cov[7] = float(position_variance[0])   # y <- ned x
    cov[14] = float(position_variance[2])  # z
    cov[21] = float(orientation_variance[0])
    cov[28] = float(orientation_variance[1])
    cov[35] = float(orientation_variance[2])
    return cov


class Ekf2OdometryAdapter(Node):
    """Republish PX4 EKF2 odometry as nav_msgs/Odometry in the ENU frame."""

    def __init__(self):
        super().__init__('ekf2_odometry_adapter')

        self.use_px4_timestamp = self.declare_parameter(
            'use_px4_timestamp', False).value

        self.odom_pub = self.create_publisher(
            Odometry, '/odometry/ekf2', 10)

        # PX4 streams are best-effort sensor data; bag replay must re-offer
        # them volatile (bags/px4_qos_overrides.yaml) or nothing arrives.
        self.sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.callback, qos_profile_sensor_data)

        self.count = 0
        self.warned_frame = False
        self.get_logger().info(
            'ekf2_odometry_adapter up: /fmu/out/vehicle_odometry (NED) -> '
            '/odometry/ekf2 (ENU). stamp source: %s'
            % ('px4_timestamp' if self.use_px4_timestamp else 'node_clock'))

    def callback(self, msg: VehicleOdometry):
        """Convert one VehicleOdometry message and publish it as Odometry."""
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            if not self.warned_frame:
                self.get_logger().warn(
                    f'pose_frame={msg.pose_frame} is not NED (1); '
                    'NED->ENU conversion assumes an NED world frame.')
                self.warned_frame = True
            return

        if self.use_px4_timestamp:
            stamp = Time(
                nanoseconds=px4_timestamp_to_ros_ns(msg.timestamp)).to_msg()
        else:
            stamp = self.get_clock().now().to_msg()

        px, py, pz = ned_to_enu_position(
            float(msg.position[0]), float(msg.position[1]),
            float(msg.position[2]))
        # PX4 q is (w, x, y, z), body-FRD -> NED.
        qw, qx, qy, qz = ned_to_enu_quaternion(
            float(msg.q[0]), float(msg.q[1]),
            float(msg.q[2]), float(msg.q[3]))

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = ODOM_FRAME
        odom.child_frame_id = BASE_FRAME
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.covariance = enu_pose_covariance_diagonal(
            msg.position_variance, msg.orientation_variance)

        (lx, ly, lz), (ax, ay, az) = enu_twist_from_ned(
            msg.velocity, msg.angular_velocity)
        odom.twist.twist.linear.x = lx
        odom.twist.twist.linear.y = ly
        odom.twist.twist.linear.z = lz
        odom.twist.twist.angular.x = ax
        odom.twist.twist.angular.y = ay
        odom.twist.twist.angular.z = az
        self.odom_pub.publish(odom)

        self.count += 1
        if self.count == 1 or self.count % 500 == 0:
            self.get_logger().info(
                f'published {self.count} ekf2 odometry msgs '
                f'(latest ENU xyz = {px:.2f}, {py:.2f}, {pz:.2f})')


def main(args=None):
    """Spin the EKF2 odometry adapter node."""
    rclpy.init(args=args)
    node = Ekf2OdometryAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
