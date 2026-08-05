"""PX4 SensorCombined -> sensor_msgs/Imu bridge.

Keeps the graph_slam package PX4-free: raw gyro/accel from
/fmu/out/sensor_combined (body FRD) is republished as a standard
sensor_msgs/Imu in body FLU on /imu/data. The SLAM front-end integrates
angular_velocity between scans as an NDT rotation initial guess.

Stamp note (L5-17i): clock-domain reconciliation happens HERE, at the
bridge, never inside graph_slam (SOTA LIO pattern; keeps the estimator
clock-agnostic).
- use_sim_time=true (SITL / bag replay): outgoing stamps come from the
  NODE CLOCK (sim time, fed by the recorded /clock) — the same domain as
  the LiDAR scans and therefore as /ndt_frontend/ndt_odom's header. The
  raw PX4 stamp is NOT forwarded: it lives in PX4's own clock domain
  (~1.78e9 s vs ~18-54 s sim time on slam_loop_03) and is unusable for
  cross-topic windowing; on this project's SITL it additionally carries a
  one-off ~132 ms lockstep step (L5_MICROSTORIES §L5-17 STEP-0). Same
  restamp pattern already proven by imu_bridge_liosam.py. Non-advancing
  stamps (sim clock only moves on /clock ticks, ~0.4 % of samples) are
  dropped so downstream preintegration never sees dt <= 0.
- use_sim_time=false (real hardware): PX4 timestamps are passed through
  unchanged, exactly as before. uXRCE-DDS timesync (default-on) already
  keeps PX4 stamps in the companion computer's OS-time domain, so the
  pass-through is correct as-is (STEP-0b §Q2/§Q4) — this branch is a
  deliberate no-op.
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
        # L5-17i: sim-time re-stamp is conditional on use_sim_time (see module
        # docstring). rclpy auto-declares use_sim_time on every node.
        self.sim_time = bool(self.get_parameter('use_sim_time').value)
        self.last_stamp_ns = 0
        self.dropped_same_stamp = 0
        self.get_logger().info(
            'imu_bridge up: /fmu/out/sensor_combined (FRD) -> /imu/data (FLU), '
            f'stamps: {"node clock (sim time)" if self.sim_time else "PX4 pass-through"}')

    def callback(self, msg: SensorCombined):
        out = Imu()
        out.header.frame_id = 'base_link'
        if self.sim_time:
            # Node clock = sim time (recorded /clock on replay) — the domain
            # ndt_odom/GT/scan headers live in, so consumers can window on the
            # header stamp (L5-17c). Strictly monotonic: drop the ~0.4 % of
            # samples where the sim clock has not advanced, else downstream
            # preintegration would see dt <= 0.
            now = self.get_clock().now()
            if now.nanoseconds <= self.last_stamp_ns:
                self.dropped_same_stamp += 1
                return
            self.last_stamp_ns = now.nanoseconds
            out.header.stamp = now.to_msg()
        else:
            # Real hardware: PX4 stamps are already in the companion's OS-time
            # domain via uXRCE-DDS timesync — pass through unchanged.
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
