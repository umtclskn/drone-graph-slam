#pragma once

#include <Eigen/Core>

namespace graph_slam {

/// Relative-motion seed for scan-to-scan NDT (NDT-08). Given two absolute poses
/// in a common frame — since L5-06 the IMU-predicted pose at the keyframe and at
/// the current scan — returns the relative transform
/// T_a_b = world_from_a^{-1} * world_from_b. This is the initial guess fed to
/// NdtRegistrar::align; NDT is initial-guess sensitive, so a good prior is the
/// biggest cheap accuracy win.
///
/// ROS-free by design: the pose source is the caller's concern.
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
