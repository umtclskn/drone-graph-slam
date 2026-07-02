#pragma once

#include <Eigen/Geometry>
#include <cstddef>
#include <vector>

namespace graph_slam {

/// A pose trajectory: index-aligned rigid poses in one common frame. EVAL-02 full
/// slice compares an estimated (dead-reckoned NDT-odometry) trajectory against the
/// ground-truth trajectory sampled at the same scan stamps.
using Trajectory = std::vector<Eigen::Isometry3f>;

/// Absolute Trajectory Error (EVAL-02). RMSE of the per-pose discrepancy after a
/// single rigid first-pose alignment of `est` onto `gt`. First-pose alignment (as
/// opposed to full Umeyama) is the honest measure of accumulated odometry drift:
/// both trajectories are pinned at pose 0 and the error is how far the open chain
/// has wandered by pose i.
struct AbsoluteTrajectoryError {
  /// RMSE of per-pose translation error in metres.
  float trans_rmse_m = 0.0F;
  /// RMSE of per-pose geodesic rotation error in degrees.
  float rot_rmse_deg = 0.0F;
  /// Number of pose pairs used (0 if inputs were empty or mismatched in length).
  std::size_t count = 0;
};

/// Relative Pose Error (EVAL-02), over a fixed index gap `delta`. For each i,
/// compares the relative motion gt_i^{-1} gt_{i+delta} against est_i^{-1}
/// est_{i+delta}; reports the RMSE of the residual translation and rotation. RPE
/// is frame-independent (no alignment), so it isolates local consistency from the
/// global drift that ATE captures.
struct RelativePoseError {
  /// RMSE of relative translation error in metres.
  float trans_rmse_m = 0.0F;
  /// RMSE of relative geodesic rotation error in degrees.
  float rot_rmse_deg = 0.0F;
  /// Index gap the metric was computed over.
  std::size_t delta = 0;
  /// Number of (i, i+delta) pairs used (0 if delta is too large or inputs empty).
  std::size_t count = 0;
};

/// ATE between an estimated and a ground-truth trajectory of equal length. Returns
/// `count == 0` (and zero errors) if the inputs are empty or differ in length.
AbsoluteTrajectoryError absoluteTrajectoryError(const Trajectory& est, const Trajectory& gt);

/// RPE between two equal-length trajectories over the given index gap. Returns
/// `count == 0` if the inputs are empty, mismatched, or shorter than `delta + 1`.
RelativePoseError relativePoseError(const Trajectory& est, const Trajectory& gt, std::size_t delta);

}  // namespace graph_slam
