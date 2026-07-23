"""Gazebo model-state -> independent ground-truth PoseStamped / Odometry + TF.

EVAL-06. The EVAL-01 bridge (ground_truth_bridge.py) derives "ground truth"
from /fmu/out/vehicle_odometry, i.e. PX4 EKF2 output -- the same source the
NDT-08 prior comes from, so every ATE/RPE/NEES number it feeds measures
*consistency with EKF2*, not accuracy. This node publishes the simulator's
absolute truth instead, so SLAM accuracy can be measured against something
neither the front-end nor the back-end is fed from.

Source: the Gazebo scene-broadcaster pose topic (gz.msgs.Pose_V), subscribed
over gz-transport directly rather than through ros_gz_bridge. Reason: the
ros_gz_bridge Pose_V -> tf2_msgs/TFMessage mapping drops the per-entity `name`
field (verified on this world -- every frame_id/child_frame_id arrives empty),
leaving only array position to identify the drone. Reading gz-transport here
lets us select by name, the same way PX4's own GZBridge::poseInfoCallback does.

Message layout (verified on my_slam_world): the entry named after the model
carries the model's *world* pose; entries named after links carry poses
*relative to the model*. World truth for a link is therefore
    T_world_link = T_world_model * T_model_link.

Gazebo is already ENU/FLU, so no NED->ENU conversion happens here -- only SE(3)
composition. That is why this node does not use px4_offboard.ned_to_enu's
ned_to_* helpers, only its quaternion math.

Stamp domain: outputs are stamped with the node clock, matching the EVAL-01
bridge. Run with use_sim_time so stamps land in the same sim-time domain as
/x500/lidar_3d/points (EVAL-01 criterion 3, < 50 ms).

gz-transport topics cannot be replayed from a rosbag, so this node must run
*live* while recording; the bag stores the resulting /ground_truth/pose.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode

from px4_offboard.ned_to_enu import quat_mul, rotate_vector

MAP_FRAME = 'map'
# Not 'base_link': the NDT front-end broadcasts odom->base_link, and a second
# parent for the same frame breaks any TF consumer (EVAL-06 AC4).
GT_BASE_FRAME = 'base_link_gt'


def _compose(model_pose, link_pose):
    """T_world_model * T_model_link -> ((x, y, z), (w, x, y, z))."""
    mp, mq = model_pose
    lp, lq = link_pose
    offset = rotate_vector(mq, lp)
    position = (mp[0] + offset[0], mp[1] + offset[1], mp[2] + offset[2])
    return position, quat_mul(mq, lq)


def _unique_entry(msg, name):
    """The sole gz pose entry called `name`, or None if absent/ambiguous.

    Link names are not unique across a world (my_slam_world has several links
    called 'link'), so refusing to guess is the honest failure mode -- picking
    the first match would silently track a wall.
    """
    found = [p for p in msg.pose if p.name == name]
    return found[0] if len(found) == 1 else None


def _pose_parts(gz_pose):
    p = gz_pose.position
    q = gz_pose.orientation
    return (p.x, p.y, p.z), (q.w, q.x, q.y, q.z)


class GazeboTruthBridge(Node):

    def __init__(self):
        super().__init__('gazebo_truth_bridge')

        self.gz_topic = self.declare_parameter(
            'gz_pose_topic', '/world/my_slam_world/pose/info').value
        # PX4 appends an instance suffix to the spawned model ('_0').
        self.model_name = self.declare_parameter(
            'model_name', 'x500_lidar_down_0').value
        self.link_name = self.declare_parameter('link_name', 'base_link').value

        self.pose_pub = self.create_publisher(
            PoseStamped, '/ground_truth/pose', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/ground_truth/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.count = 0
        self.missing = 0

        self.gz_node = GzNode()
        if not self.gz_node.subscribe(Pose_V, self.gz_topic, self.callback):
            raise RuntimeError(f'failed to subscribe to gz topic {self.gz_topic}')

        self.get_logger().info(
            f'gazebo_truth_bridge up: gz {self.gz_topic} '
            f'[{self.model_name}/{self.link_name}] -> /ground_truth/pose + '
            f'/ground_truth/odom + TF {MAP_FRAME}->{GT_BASE_FRAME}')

    def callback(self, msg: Pose_V):
        model = _unique_entry(msg, self.model_name)
        link = _unique_entry(msg, self.link_name)
        if model is None or link is None:
            self.missing += 1
            if self.missing == 1 or self.missing % 500 == 0:
                names = sorted({p.name for p in msg.pose})
                self.get_logger().warn(
                    f'no unique entry for model={self.model_name!r} / '
                    f'link={self.link_name!r} ({self.missing} skipped); '
                    f'names present: {names}')
            return

        (px, py, pz), (qw, qx, qy, qz) = _compose(
            _pose_parts(model), _pose_parts(link))

        # Node clock, not the gz header stamp -- see module docstring.
        stamp = self.get_clock().now().to_msg()

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

        # Pose only: gz pose/info carries no velocity, and ground truth has no
        # covariance (ARCHITECTURE.md sec 6). Both stay at their zero defaults.
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = MAP_FRAME
        odom.child_frame_id = GT_BASE_FRAME
        odom.pose.pose = pose.pose
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
                f'published {self.count} Gazebo-truth poses '
                f'(latest ENU xyz = {px:.2f}, {py:.2f}, {pz:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = GazeboTruthBridge()
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
