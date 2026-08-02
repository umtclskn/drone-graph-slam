# L5-06: the back-end's latest optimized keyframe state, published whenever the
# graph estimate updates (keyframe cadence). It is the typed companion of
# /slam/optimized_odom, which is a plain nav_msgs/Odometry and therefore cannot
# carry the velocity and IMU-bias half of the L5-02 node model (Pose3,
# Velocity3, imuBias). The front-end resets its IMU predictor from this message
# (L5-06) and L5-07's submap will read the same feed for optimized poses.
#
# PX4-agnostic: std_msgs/geometry_msgs only, no px4_msgs anywhere.

# stamp = the scan time of the keyframe this state belongs to;
# frame_id = the map frame the pose and velocity are expressed in (ENU).
std_msgs/Header header

# X(i): optimized keyframe pose in header.frame_id.
geometry_msgs/Pose pose

# V(i): optimized keyframe velocity in header.frame_id [m/s].
geometry_msgs/Vector3 velocity

# B(i): optimized IMU bias in the body (FLU base_link) frame.
geometry_msgs/Vector3 accel_bias   # [m/s^2]
geometry_msgs/Vector3 gyro_bias    # [rad/s]
