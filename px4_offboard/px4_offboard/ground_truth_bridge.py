"""PX4 VehicleOdometry -> ground-truth PoseStamped / Odometry + TF bridge.

EVAL-01. Exposes the simulator's true pose on standard ROS topics so the SLAM
evaluation can compare estimate vs ground truth, without graph_slam ever
touching px4_msgs (the boundary rule from ARCHITECTURE.md sec 3/11). Every
NED->ENU conversion happens here, via px4_offboard.ned_to_enu.

Ground-truth source: /fmu/out/vehicle_odometry (px4_msgs/VehicleOdometry).
This is EKF2 *output*, not absolute simulator truth -- and the NDT-08 prior
derives from the same source, so metrics computed against it measure
consistency with EKF2, not accuracy. Kept for replaying pre-EVAL-06 bags
(slam_loop_01/02), which carry no Gazebo model-state topic. For independent
truth use gazebo_truth_bridge.py (EVAL-06); both publish /ground_truth/pose,
so run exactly one of them.

Stamp domain: PX4 message timestamps are wall-clock microseconds, a different
clock from the sim-time LiDAR scans (verified on slam_loop_02: PX4 ts ~1.78e9 s
vs scan stamp ~22 s). Matching them by value is impossible, so outputs are
stamped with the node's current clock; run with use_sim_time and
`ros2 bag play --clock` so the stamp lands in the same sim-time domain as
/x500/lidar_3d/points and stays within the 50 ms sync budget (criterion 3).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from px4_msgs.msg import VehicleOdometry

from px4_offboard.ned_to_enu import (
    ned_to_enu_position,
    ned_to_enu_quaternion,
    quat_conjugate,
    rotate_vector,
)

MAP_FRAME = 'map'
# Not 'base_link': the NDT front-end broadcasts odom->base_link, and a second
# parent for the same frame breaks any TF consumer (EVAL-06 AC4).
GT_BASE_FRAME = 'base_link_gt'


class GroundTruthBridge(Node):

    def __init__(self):
        super().__init__('ground_truth_bridge')

        self.pose_pub = self.create_publisher(
            PoseStamped, '/ground_truth/pose', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/ground_truth/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # PX4 streams are best-effort sensor data; bag replay must re-offer
        # them volatile (bags/px4_qos_overrides.yaml) or nothing arrives.
        self.sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.callback, qos_profile_sensor_data)

        self.count = 0
        self.warned_frame = False
        self.get_logger().info(
            'ground_truth_bridge up: /fmu/out/vehicle_odometry (NED) -> '
            f'/ground_truth/pose + /ground_truth/odom + TF map->{GT_BASE_FRAME}')

    def callback(self, msg: VehicleOdometry):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            if not self.warned_frame:
                self.get_logger().warn(
                    f'pose_frame={msg.pose_frame} is not NED (1); '
                    'NED->ENU conversion assumes an NED world frame.')
                self.warned_frame = True
            return

        # Stamp in the node's clock domain (sim time under use_sim_time), not
        # the PX4 wall-clock timestamp -- see module docstring.
        stamp = self.get_clock().now().to_msg()

        px, py, pz = ned_to_enu_position(
            float(msg.position[0]), float(msg.position[1]),
            float(msg.position[2]))
        # PX4 q is (w, x, y, z), body-FRD -> NED.
        qw, qx, qy, qz = ned_to_enu_quaternion(
            float(msg.q[0]), float(msg.q[1]),
            float(msg.q[2]), float(msg.q[3]))

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = MAP_FRAME
        pose.pose.position.x = px
        pose.pose.position.y = py
        pose.pose.position.z = pz
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        self.pose_pub.publish(pose)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = MAP_FRAME
        odom.child_frame_id = GT_BASE_FRAME
        odom.pose.pose = pose.pose
        # Variances are per-axis; the axis swap just relabels them (signs do
        # not affect variance). Order in ROS covariance is x, y, z, rx, ry, rz.
        odom.pose.covariance[0] = float(msg.position_variance[1])   # x <- ned y
        odom.pose.covariance[7] = float(msg.position_variance[0])   # y <- ned x
        odom.pose.covariance[14] = float(msg.position_variance[2])  # z
        odom.pose.covariance[21] = float(msg.orientation_variance[0])
        odom.pose.covariance[28] = float(msg.orientation_variance[1])
        odom.pose.covariance[35] = float(msg.orientation_variance[2])

        # Twist is expressed in child_frame_id (base_link, FLU). Body angular
        # velocity is reported FRD -> FLU flips y, z. Linear velocity is NED
        # world; convert to ENU world then rotate into the body frame.
        q_enu = (qw, qx, qy, qz)
        vx, vy, vz = rotate_vector(
            quat_conjugate(q_enu),
            ned_to_enu_position(
                float(msg.velocity[0]), float(msg.velocity[1]),
                float(msg.velocity[2])))
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.x = float(msg.angular_velocity[0])
        odom.twist.twist.angular.y = -float(msg.angular_velocity[1])
        odom.twist.twist.angular.z = -float(msg.angular_velocity[2])
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = MAP_FRAME
        tf.child_frame_id = GT_BASE_FRAME
        tf.transform.translation.x = px
        tf.transform.translation.y = py
        tf.transform.translation.z = pz
        tf.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        self.count += 1
        if self.count == 1 or self.count % 500 == 0:
            self.get_logger().info(
                f'published {self.count} ground-truth poses '
                f'(latest ENU xyz = {px:.2f}, {py:.2f}, {pz:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthBridge()
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
