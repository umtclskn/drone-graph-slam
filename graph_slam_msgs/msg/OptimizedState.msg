# L5-06 / L5-19: the back-end's optimized keyframe state, published whenever the
# graph estimate updates (keyframe cadence on the single-graph path; per NDT
# correction on dual-graph). Typed companion of /slam/optimized_odom, which is
# plain nav_msgs/Odometry and cannot carry velocity, IMU bias, or a keyframe id.
# L5-19: the front-end submap matches window entries by header.stamp (primary)
# or keyframe_id — never "latest" / "nearest" — and uses the pose as the sole
# submap-fusion source once refined. The NDT initial guess stays on EKF2 (L5-18)
# and does not consume this message.
#
# PX4-agnostic: std_msgs/geometry_msgs only, no px4_msgs anywhere.

# stamp = the scan time of the keyframe this state belongs to
#         (copied from the triggering ndt_odom header);
# frame_id = the map frame the pose and velocity are expressed in (ENU).
std_msgs/Header header

# Backend graph index of this keyframe (sym::X(keyframe_id)). -1 if unknown.
int32 keyframe_id

# X(i): optimized keyframe pose in header.frame_id.
geometry_msgs/Pose pose

# V(i): optimized keyframe velocity in header.frame_id [m/s].
geometry_msgs/Vector3 velocity

# B(i): optimized IMU bias in the body (FLU base_link) frame.
geometry_msgs/Vector3 accel_bias   # [m/s^2]
geometry_msgs/Vector3 gyro_bias    # [rad/s]
