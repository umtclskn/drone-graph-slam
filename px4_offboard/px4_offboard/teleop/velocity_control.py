#!/usr/bin/env python3
"""Classic PX4 Offboard velocity bridge for keyboard teleop.

Adapted from ARK-Electronics ROS2_PX4_Offboard_Example velocity_control.py:
  /offboard_velocity_cmd (Twist, body: x forward, y left, z up)
  /arm_message (Bool)
→ streams OffboardControlMode(velocity) + TrajectorySetpoint(velocity NED).

Topic names match this workspace's PX4 uXRCE bridge
(`/fmu/out/vehicle_status_v4`, `/fmu/out/vehicle_local_position_v1`).

Indoor takeoff altitude ~1.5 m AGL (room clearance), not ARK's 5 m.
Do NOT run alongside offboard_control (position mission).
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool

TAKEOFF_ALTITUDE_NED = -1.5  # ~1.5 m AGL
TWIST_STALE_S = 0.35


class VelocityTeleop(Node):

    def __init__(self):
        super().__init__("velocity_teleop")

        # PX4 uXRCE /fmu/out/* is typically BEST_EFFORT + TRANSIENT_LOCAL.
        # VOLATILE subscribers can miss status and leave us stuck in ARMING.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # ARK uses TRANSIENT_LOCAL for cmd/arm so late subscribers see last arm.
        cmd_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", sensor_qos)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", sensor_qos)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", command_qos)

        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v4",
            self._status_cb, sensor_qos)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self._local_cb, sensor_qos)
        self.create_subscription(
            Twist, "/offboard_velocity_cmd", self._twist_cb, cmd_qos)
        self.create_subscription(
            Bool, "/arm_message", self._arm_msg_cb, cmd_qos)

        self.status = VehicleStatus()
        self.local = VehicleLocalPosition()
        self.got_status = False
        self.got_local = False
        self.twist = Twist()
        self.last_twist_time = self.get_clock().now()
        self.arm_request = False
        self.offboard_ready = False
        self.state = "IDLE"
        self.stream_count = 0
        self.diag_count = 0

        # One 20 Hz loop like offboard_control: stream setpoints + FSM together.
        self.create_timer(1.0 / 20.0, self._timer)
        self.get_logger().info(
            "velocity_teleop up: SPACE in keyboard node to arm; "
            "then fly with arrows/WASD. Takeoff ~1.5 m AGL.")

    def now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _yaw(self) -> float:
        # px4_msgs VehicleLocalPosition has heading_good_for_control, not heading_valid.
        if getattr(self.local, "heading_good_for_control", False) or self.local.xy_valid:
            return float(self.local.heading)
        return 0.0

    def _status_cb(self, msg: VehicleStatus):
        self.status = msg
        self.got_status = True

    def _local_cb(self, msg: VehicleLocalPosition):
        self.local = msg
        self.got_local = True

    def _twist_cb(self, msg: Twist):
        self.twist = msg
        self.last_twist_time = self.get_clock().now()

    def _arm_msg_cb(self, msg: Bool):
        self.arm_request = bool(msg.data)
        self.get_logger().info(f"arm_request={self.arm_request}")

    def _publish_vehicle_command(self, command: int, p1=0.0, p2=0.0, p7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.param7 = float(p7)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.now_us()
        self.vehicle_command_pub.publish(msg)

    def _arm(self, force: bool = False):
        # param2=21196 = MAVLink force-arm (bypass some preflight refusals).
        # Needed in this SITL session: offboard succeeds but pre_flight_checks_pass=False.
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            p1=1.0,
            p2=21196.0 if force else 0.0,
        )

    def _set_offboard(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)

    def _publish_offboard_mode(self, velocity: bool):
        msg = OffboardControlMode()
        msg.timestamp = self.now_us()
        msg.position = not velocity
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

    def _publish_position_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float("nan")
        self.trajectory_pub.publish(msg)

    def _publish_velocity_setpoint(self, vx, vy, vz, yawspeed):
        msg = TrajectorySetpoint()
        msg.timestamp = self.now_us()
        msg.position = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = float("nan")
        msg.yawspeed = float(yawspeed)
        self.trajectory_pub.publish(msg)

    def _timer(self):
        armed = self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
        offboard = self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        xy_ok = bool(self.local.xy_valid)
        yaw = self._yaw()

        # Periodic diagnose while stuck arming/taking off.
        self.diag_count += 1
        if self.state in ("ARMING", "TAKEOFF") and self.diag_count % 40 == 0:
            self.get_logger().info(
                f"wait {self.state}: got_status={self.got_status} "
                f"got_local={self.got_local} xy_valid={xy_ok} "
                f"arming_state={self.status.arming_state} "
                f"nav_state={self.status.nav_state} "
                f"preflight_ok={self.status.pre_flight_checks_pass} "
                f"failsafe={self.status.failsafe} "
                f"z={self.local.z:.2f} stream={self.stream_count} "
                f"armed={armed} offboard={offboard}"
            )

        if self.state == "IDLE":
            if self.arm_request and xy_ok:
                self.state = "ARMING"
                self.stream_count = 0
                self.get_logger().info("STATE -> ARMING")
            elif self.arm_request and not xy_ok and self.diag_count % 40 == 0:
                self.get_logger().warn(
                    f"arm requested but xy_valid=False "
                    f"(got_local={self.got_local}); waiting for EKF"
                )
            return

        if self.state == "ARMING":
            # Stream position setpoints BEFORE/while requesting OFFBOARD (PX4 req).
            self._publish_offboard_mode(velocity=False)
            self._publish_position_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE_NED, yaw)
            self.stream_count += 1
            # Retry arm + offboard every 1 s after the first 1 s of streaming.
            if self.stream_count >= 20 and (self.stream_count % 20 == 0):
                self._set_offboard()
                # After try #2, force-arm if preflight still failing (SITL).
                force = (self.stream_count >= 40) and (
                    not bool(self.status.pre_flight_checks_pass))
                self._arm(force=force)
                self.get_logger().info(
                    f"sent SET_MODE(OFFBOARD)+ARM "
                    f"(try #{self.stream_count // 20}, force={force})"
                )
                if force:
                    self.get_logger().warn(
                        "pre_flight_checks_pass=False — using FORCE ARM "
                        "(param2=21196). If still disarmed, check PX4 console "
                        "for arm denial / set NAV_DLL_ACT=0 in QGC."
                    )
            if self.stream_count >= 20 and armed and offboard:
                self.state = "TAKEOFF"
                self.get_logger().info("STATE -> TAKEOFF")
            return

        if self.state == "TAKEOFF":
            self._publish_offboard_mode(velocity=False)
            self._publish_position_setpoint(0.0, 0.0, TAKEOFF_ALTITUDE_NED, yaw)
            if abs(self.local.z) > 1.2:
                self.state = "FLYING"
                self.offboard_ready = True
                self.get_logger().info("STATE -> FLYING (velocity teleop)")
            if not self.arm_request:
                self.state = "IDLE"
                self.offboard_ready = False
                self.get_logger().info("STATE -> IDLE (arm cleared)")
            return

        if self.state == "FLYING":
            if not self.arm_request or not armed:
                self.state = "IDLE"
                self.offboard_ready = False
                self.get_logger().info("STATE -> IDLE")
                return

            self._publish_offboard_mode(velocity=True)
            age = (self.get_clock().now() - self.last_twist_time).nanoseconds * 1e-9
            if age > TWIST_STALE_S:
                self._publish_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
                return

            c, s = math.cos(yaw), math.sin(yaw)
            bx = float(self.twist.linear.x)
            by = float(self.twist.linear.y)
            bz_up = float(self.twist.linear.z)
            vx = bx * c - by * s
            vy = bx * s + by * c
            vz = -bz_up
            yawspeed = -float(self.twist.angular.z)
            self._publish_velocity_setpoint(vx, vy, vz, yawspeed)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
