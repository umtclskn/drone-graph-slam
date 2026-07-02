#!/usr/bin/env python3
"""PX4 offboard mission: arm, takeoff, fly a rectangular loop, land.

State machine: WAITING_EKF -> ARMING -> TAKEOFF -> FLYING -> LANDING -> DONE
All coordinates are in the PX4 NED frame (Z is down, so altitude is negative).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# Mission constants
TAKEOFF_ALTITUDE = -1.5  # NED, ~1.5 m above ground
LOOP_HZ = 20.0
ARM_STREAM_CYCLES = 20  # ~1 s of setpoints before switching mode
WAYPOINT_RADIUS = 0.5   # m, XY acceptance radius

# IMPORTANT: waypoints are in PX4 NED, which is swapped vs the Gazebo ENU world.
#   NED x (North) == Gazebo +y  -> room's SHORT 15 m side, walls at +-7.4
#   NED y (East)  == Gazebo +x  -> room's LONG  20 m side, walls at +-9.9
# So the loop must be SHORT in x and LONG in y. Obstacles in NED frame:
#   pillar_1   x[ 1.5, 2.5] y[ 2.5, 3.5]
#   pillar_2   x[-3.5,-2.5] y[-4.5,-3.5]
#   inner_wall x[-1.1,-0.9] y[-5.0, 1.0]
# All obstacles sit within x[-3.5,2.5], y[-5,3.5]. The perimeter at x=+-5.5,
# y=+-8 clears every obstacle AND keeps ~1.9 m to the walls. Enter/leave along
# x=0 (East axis), which passes north of inner_wall and clear of the pillars.
WAYPOINTS = [
    (0.0, 8.0, -1.5),    # head East along x=0 to the east edge (clear)
    (5.5, 8.0, -1.5),    # corner
    (5.5, -8.0, -1.5),   # corner
    (-5.5, -8.0, -1.5),  # corner
    (-5.5, 8.0, -1.5),   # corner
    (0.0, 8.0, -1.5),    # close loop at east-edge midpoint
    (0.0, 0.0, -1.5),    # return to origin along x=0
]

# States
WAITING_EKF = 'WAITING_EKF'
ARMING = 'ARMING'
TAKEOFF = 'TAKEOFF'
FLYING = 'FLYING'
LANDING = 'LANDING'
DONE = 'DONE'


class OffboardControl(Node):

    def __init__(self):
        super().__init__('offboard_control')

        # PX4 publishes/subscribes with BEST_EFFORT; commands are RELIABLE.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', sensor_qos)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', sensor_qos)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', command_qos)

        # Subscribers
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.local_position_cb, sensor_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4',
            self.vehicle_status_cb, sensor_qos)

        # State
        self.local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.state = WAITING_EKF
        self.arm_counter = 0
        self.waypoint_index = 0
        self.distance_log_counter = 0

        self.get_logger().info('STATE -> WAITING_EKF')

        self.timer = self.create_timer(1.0 / LOOP_HZ, self.timer_callback)

    # --- Subscriber callbacks ---------------------------------------------

    def local_position_cb(self, msg):
        self.local_position = msg

    def vehicle_status_cb(self, msg):
        self.vehicle_status = msg

    # --- Helpers ----------------------------------------------------------

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()
        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float('nan')
        msg.timestamp = self.now_us()
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.now_us()
        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def set_offboard_mode(self):
        # base mode custom (1), PX4 custom main mode OFFBOARD (6)
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def transition(self, new_state):
        self.state = new_state
        self.get_logger().info(f'STATE -> {new_state}')

    # --- Main loop --------------------------------------------------------

    def timer_callback(self):
        # Always stream offboard heartbeat + a setpoint to satisfy PX4.
        self.publish_offboard_control_mode()

        if self.state == WAITING_EKF:
            self.publish_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE, 0.0)
            if self.local_position.xy_valid:
                self.transition(ARMING)

        elif self.state == ARMING:
            self.publish_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE, 0.0)
            self.arm_counter += 1
            if self.arm_counter == ARM_STREAM_CYCLES:
                self.set_offboard_mode()
                self.arm()
            armed = (self.vehicle_status.arming_state ==
                     VehicleStatus.ARMING_STATE_ARMED)
            offboard = (self.vehicle_status.nav_state ==
                        VehicleStatus.NAVIGATION_STATE_OFFBOARD)
            if self.arm_counter >= ARM_STREAM_CYCLES and armed and offboard:
                self.transition(TAKEOFF)

        elif self.state == TAKEOFF:
            self.publish_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE, 0.0)
            if abs(self.local_position.z) > 1.2:
                self.transition(FLYING)

        elif self.state == FLYING:
            self.fly()

        elif self.state == LANDING:
            # Keep streaming a setpoint; PX4 handles the descent.
            self.publish_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE, 0.0)
            if self.local_position.z > -0.1:
                self.transition(DONE)

        elif self.state == DONE:
            self.get_logger().info('Mission complete.')
            self.timer.cancel()
            rclpy.shutdown()

    def fly(self):
        wx, wy, wz = WAYPOINTS[self.waypoint_index]
        dx = wx - self.local_position.x
        dy = wy - self.local_position.y
        distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)

        self.publish_setpoint(wx, wy, wz, yaw)

        # Log distance to the next waypoint every ~2 s.
        self.distance_log_counter += 1
        if self.distance_log_counter >= int(2 * LOOP_HZ):
            self.distance_log_counter = 0
            self.get_logger().info(
                f'WP {self.waypoint_index + 1}/{len(WAYPOINTS)} '
                f'({wx:.1f}, {wy:.1f}) dist={distance:.2f} m')

        if distance < WAYPOINT_RADIUS:
            self.get_logger().info(f'Reached waypoint {self.waypoint_index + 1}')
            self.waypoint_index += 1
            self.distance_log_counter = 0
            if self.waypoint_index >= len(WAYPOINTS):
                self.land()
                self.transition(LANDING)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
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
