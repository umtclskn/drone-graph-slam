#pragma once

#include <Eigen/Core>

namespace graph_slam {

/// Relative-motion seed for scan-to-scan NDT (NDT-08). Given two absolute poses
/// in a common frame — e.g. two consecutive EKF2 odometry poses, already in the
/// ENU/base_link frame the px4_offboard adapter produces — returns the relative
/// transform T_a_b = world_from_a^{-1} * world_from_b. This is the initial guess
/// fed to NdtRegistrar::align; NDT is initial-guess sensitive, so a good prior is
/// the biggest cheap accuracy win.
///
/// ROS-free and px4-free by design: the px4_msgs -> Eigen / NED->ENU conversion is
/// the adapter's job (ARCHITECTURE §3, §5), never this package's.
inline Eigen::Matrix4f relativePoseGuess(const Eigen::Matrix4f& world_from_a,
                                         const Eigen::Matrix4f& world_from_b) {
  // Rigid inverse of an SE(3) matrix: R^T and -R^T t. Cheaper and numerically
  // cleaner than a general 4x4 inverse for a pose.
  const Eigen::Matrix3f rot = world_from_a.block<3, 3>(0, 0);
  const Eigen::Vector3f trans = world_from_a.block<3, 1>(0, 3);

  Eigen::Matrix4f a_inv = Eigen::Matrix4f::Identity();
  a_inv.block<3, 3>(0, 0) = rot.transpose();
  a_inv.block<3, 1>(0, 3) = -rot.transpose() * trans;
  return a_inv * world_from_b;
}

}  // namespace graph_slam
