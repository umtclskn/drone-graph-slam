"""PX4 SensorCombined + VehicleOdometry -> sensor_msgs/Imu bridge for LIO-SAM.

LIO-SAM-02. A LIO-SAM-specific COPY of imu_bridge.py -- the original node and
its /imu/data topic are production for the NDT/EKF2 pipeline and stay
untouched. This copy publishes /liosam/imu/data and differs in exactly two
ways, both required by LIO-SAM's imageProjection deskewInfo():

1. Clock domain: outgoing stamps come from the NODE CLOCK (sim time under
   use_sim_time, fed by the bag's recorded /clock), NOT the PX4 wall-clock
   microsecond timestamp inside SensorCombined (~1.78e9 s vs ~22 s sim time
   on slam_loop_02). Same restamp pattern as ekf2_odometry_adapter; without
   it the IMU queue never spans a scan window and deskewInfo() waits forever.

2. Orientation merge: SensorCombined has no attitude, but LIO-SAM (9-axis
   IMU assumed) reads Imu.orientation for roll/pitch initialization. We hold
   the latest /fmu/out/vehicle_odometry attitude (~100 Hz) and stamp it onto
   every outgoing sample (~250 Hz) -- zero-order hold / latest-value merge,
   at worst ~10 ms of attitude lag, well below LIO-SAM's needs.

Frame / quaternion conventions (explicit, because a wrong conversion
silently poisons LIO-SAM's roll/pitch fusion):
- gyro/accel: PX4 body FRD -> ROS body FLU (x keeps sign, y and z flip).
- VehicleOdometry.q is (w, x, y, z), rotating body-FRD vectors into the NED
  world frame. ned_to_enu_quaternion left-multiplies by the static NED->ENU
  world rotation and right-multiplies by the FRD->FLU body rotation, so the
  published Imu.orientation is (w, x, y, z) body-FLU -> world-ENU (REP 103)
  -- consistent with the FLU angular_velocity / linear_acceleration.
- Samples arriving before the first valid attitude are DROPPED rather than
  published with the identity quaternion: w=1 passes LIO-SAM's norm check
  while encoding garbage attitude.
- Outgoing stamps are STRICTLY monotonic: the sim clock only advances on
  /clock ticks (~250 Hz on slam_loop_02), so two IMU callbacks between ticks
  would get identical stamps and GTSAM's integrateMeasurement throws on
  dt <= 0. Non-advancing samples (~0.4 % on slam_loop_02) are dropped.

QoS: sensor-data (BEST_EFFORT + VOLATILE), as in the original bridge. Bag
replay must re-offer the PX4 topics volatile (bags/px4_qos_overrides.yaml)
and run WITHOUT --clock so the recorded sim-time /clock drives this node.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import SensorCombined, VehicleOdometry
from sensor_msgs.msg import Imu

from px4_offboard.ned_to_enu import ned_to_enu_quaternion

# Fixed attitude variance for the merged orientation (rad^2). EKF2's attitude
# on this platform is a few degrees at worst; 0.01 rad^2 (~5.7 deg std) is a
# conservative, valid (non -1) value -- LIO-SAM only checks it is not -1.
ORIENTATION_VARIANCE = 0.01


class ImuBridgeLiosam(Node):

    def __init__(self):
        super().__init__('imu_bridge_liosam')
        self.pub = self.create_publisher(
            Imu, '/liosam/imu/data', qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(
            SensorCombined, '/fmu/out/sensor_combined',
            self.imu_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self.odom_callback, qos_profile_sensor_data)
        self.latest_q_enu = None  # (w, x, y, z), body-FLU -> world-ENU
        self.last_stamp_ns = 0
        self.count = 0
        self.dropped_no_attitude = 0
        self.dropped_same_stamp = 0
        self.warned_frame = False
        self.get_logger().info(
            'imu_bridge_liosam up: /fmu/out/sensor_combined (FRD) + '
            '/fmu/out/vehicle_odometry (NED att) -> /liosam/imu/data '
            '(FLU, ENU orientation, node-clock stamps)')

    def odom_callback(self, msg: VehicleOdometry):
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            if not self.warned_frame:
                self.get_logger().warn(
                    f'pose_frame={msg.pose_frame} is not NED (1); '
                    'skipping attitude, NED->ENU conversion assumes NED.')
                self.warned_frame = True
            return
        # PX4 q is (w, x, y, z), body-FRD -> NED; convert once, hold latest.
        q = ned_to_enu_quaternion(
            float(msg.q[0]), float(msg.q[1]),
            float(msg.q[2]), float(msg.q[3]))
        if not math.isfinite(q[0]) or abs(
                math.sqrt(sum(c * c for c in q)) - 1.0) > 1e-3:
            return  # NaN / non-unit attitude (e.g. pre-arm): keep previous
        self.latest_q_enu = q

    def imu_callback(self, msg: SensorCombined):
        if self.latest_q_enu is None:
            # No attitude yet: drop instead of publishing identity (w=1
            # passes LIO-SAM's norm check with garbage attitude).
            self.dropped_no_attitude += 1
            if self.dropped_no_attitude == 1:
                self.get_logger().info(
                    'dropping IMU samples until first vehicle_odometry '
                    'attitude arrives')
            return

        # Node clock = sim time under use_sim_time (recorded /clock), the
        # same domain as /x500/lidar_3d/points -- NOT msg.timestamp, which
        # is PX4 wall-clock microseconds (see module docstring, blocker 1).
        now = self.get_clock().now()
        if now.nanoseconds <= self.last_stamp_ns:
            # Sim clock has not advanced since the last sample; a duplicate
            # stamp makes GTSAM preintegration throw on dt <= 0.
            self.dropped_same_stamp += 1
            return
        self.last_stamp_ns = now.nanoseconds

        out = Imu()
        out.header.frame_id = 'base_link'
        out.header.stamp = now.to_msg()

        # FRD -> FLU: x stays, y and z flip sign.
        out.angular_velocity.x = float(msg.gyro_rad[0])
        out.angular_velocity.y = -float(msg.gyro_rad[1])
        out.angular_velocity.z = -float(msg.gyro_rad[2])
        out.linear_acceleration.x = float(msg.accelerometer_m_s2[0])
        out.linear_acceleration.y = -float(msg.accelerometer_m_s2[1])
        out.linear_acceleration.z = -float(msg.accelerometer_m_s2[2])

        # Latest EKF2 attitude, body-FLU -> world-ENU (blocker 2). ZOH at
        # 100 Hz onto the 250 Hz IMU stream.
        qw, qx, qy, qz = self.latest_q_enu
        out.orientation.w = qw
        out.orientation.x = qx
        out.orientation.y = qy
        out.orientation.z = qz
        out.orientation_covariance[0] = ORIENTATION_VARIANCE
        out.orientation_covariance[4] = ORIENTATION_VARIANCE
        out.orientation_covariance[8] = ORIENTATION_VARIANCE

        self.pub.publish(out)
        self.count += 1
        if self.count == 1 or self.count % 500 == 0:
            self.get_logger().info(
                f'republished {self.count} IMU samples (with attitude)')


def main(args=None):
    rclpy.init(args=args)
    node = ImuBridgeLiosam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
