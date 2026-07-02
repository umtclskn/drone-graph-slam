#pragma once

#include <Eigen/Geometry>

namespace graph_slam {

/// Per-pair relative transform error (EVAL-02, Sprint 1 slice). Plain data:
/// how far one scan-to-scan NDT result is from the ground-truth motion over the
/// same time interval.
struct RelativeErrorResult {
  /// Translation error magnitude ||t(dT)|| in metres.
  float t_err_m = 0.0F;
  /// Rotation error magnitude (geodesic angle of dT) in degrees.
  float r_err_deg = 0.0F;
};

/// Relative error between a measured NDT transform and the ground-truth motion
/// for the SAME ordered scan pair (t_i -> t_{i+1}). Both must express the same
/// relative motion convention: the transform that maps frame-i coordinates onto
/// frame-{i+1} (i.e. T_gt = GT_i^{-1} * GT_{i+1}; T_ndt the analogous registrar
/// output). Error transform dT = T_gt^{-1} * T_ndt; the result is its
/// translation norm and its rotation angle. Pure, ROS-free.
RelativeErrorResult compute(const Eigen::Isometry3f& T_ndt, const Eigen::Isometry3f& T_gt);

}  // namespace graph_slam
