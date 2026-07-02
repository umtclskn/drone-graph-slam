#!/usr/bin/env python3
"""Records PX4 local position + status to a CSV for offline trajectory analysis.

Run this alongside the mission. It logs the *real* odometry (EKF local position)
so we can compare the actual flight path against the world obstacles.

Usage (standalone, no build needed):
    python3 ~/portfolio_ws/drone_ws/src/px4_offboard/px4_offboard/odom_logger.py

Output: /tmp/px4_odom_log.csv  (overwritten each run)
Stop with Ctrl-C when the mission finishes.
"""

import csv
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

CSV_PATH = '/tmp/px4_odom_log.csv'


class OdomLogger(Node):

    def __init__(self):
        super().__init__('odom_logger')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.status = VehicleStatus()
        self.t0 = time.time()
        self.rows = 0

        self.file = open(CSV_PATH, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'wall_t', 'px4_t_us', 'x', 'y', 'z',
            'vx', 'vy', 'vz', 'heading',
            'xy_valid', 'z_valid', 'arming_state', 'nav_state',
        ])

        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self.status_cb, qos)
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.pos_cb, qos)

        self.get_logger().info(f'Logging odometry to {CSV_PATH} ... Ctrl-C to stop.')

    def status_cb(self, msg):
        self.status = msg

    def pos_cb(self, msg):
        self.writer.writerow([
            f'{time.time() - self.t0:.3f}', msg.timestamp,
            f'{msg.x:.3f}', f'{msg.y:.3f}', f'{msg.z:.3f}',
            f'{msg.vx:.3f}', f'{msg.vy:.3f}', f'{msg.vz:.3f}',
            f'{msg.heading:.3f}',
            int(msg.xy_valid), int(msg.z_valid),
            self.status.arming_state, self.status.nav_state,
        ])
        self.file.flush()
        self.rows += 1
        if self.rows % 50 == 0:
            self.get_logger().info(
                f'[{self.rows}] x={msg.x:6.2f} y={msg.y:6.2f} z={msg.z:6.2f} '
                f'nav={self.status.nav_state} arm={self.status.arming_state}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.file.close()
        node.get_logger().info(f'Saved {node.rows} rows to {CSV_PATH}')
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
