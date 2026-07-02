"""PX4 SensorCombined -> sensor_msgs/Imu bridge.

Keeps the graph_slam package PX4-free: raw gyro/accel from
/fmu/out/sensor_combined (body FRD) is republished as a standard
sensor_msgs/Imu in body FLU on /imu/data. The SLAM front-end integrates
angular_velocity between scans as an NDT rotation initial guess.

Stamp note: PX4 timestamps (microseconds) are passed through unchanged.
They live in PX4's clock domain, not sim time; consumers must only use
them for intra-stream deltas, not for cross-topic matching.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import SensorCombined
from sensor_msgs.msg import Imu

# Sensor-data QoS (BEST_EFFORT + VOLATILE) is what the official PX4 ROS 2
# examples use against the uXRCE-DDS agent. Note for bag replay: rosbag2
# re-offers the recorded TRANSIENT_LOCAL durability and that combination does
# not deliver reliably; play PX4 topics with a QoS override that sets
# durability to volatile (see bags/px4_qos_overrides.yaml).


class ImuBridge(Node):

    def __init__(self):
        super().__init__('imu_bridge')
        self.pub = self.create_publisher(Imu, '/imu/data', qos_profile_sensor_data)
        self.sub = self.create_subscription(
            SensorCombined, '/fmu/out/sensor_combined',
            self.callback, qos_profile_sensor_data)
        self.count = 0
        self.get_logger().info(
            'imu_bridge up: /fmu/out/sensor_combined (FRD) -> /imu/data (FLU)')

    def callback(self, msg: SensorCombined):
        out = Imu()
        out.header.frame_id = 'base_link'
        out.header.stamp.sec = int(msg.timestamp // 1_000_000)
        out.header.stamp.nanosec = int((msg.timestamp % 1_000_000) * 1_000)

        # FRD -> FLU: x stays, y and z flip sign.
        out.angular_velocity.x = float(msg.gyro_rad[0])
        out.angular_velocity.y = -float(msg.gyro_rad[1])
        out.angular_velocity.z = -float(msg.gyro_rad[2])
        out.linear_acceleration.x = float(msg.accelerometer_m_s2[0])
        out.linear_acceleration.y = -float(msg.accelerometer_m_s2[1])
        out.linear_acceleration.z = -float(msg.accelerometer_m_s2[2])

        # No orientation estimate in this message (REP 145 convention).
        out.orientation_covariance[0] = -1.0

        self.pub.publish(out)
        self.count += 1
        if self.count == 1 or self.count % 500 == 0:
            self.get_logger().info(f'republished {self.count} IMU samples')


def main(args=None):
    rclpy.init(args=args)
    node = ImuBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
