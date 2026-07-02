#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

namespace graph_slam::graph {

/// Compounded keyframe-to-keyframe motion (ARCHITECTURE §6 concept #2).
/// `delta` is the SE(3) product of the scan-to-scan steps since the last reset;
/// `covariance` is their first-order propagated uncertainty (Σ_prop), in
/// GTSAM tangent order (rx, ry, rz, x, y, z).
struct AccumulatedOdometry {
  gtsam::Pose3 delta;
  gtsam::Matrix66 covariance;
};

/// Front-end accumulator: compounds scan-to-scan NDT measurements (ΔT, Σ_meas)
/// into a single (ΔT_acc, Σ_prop) between keyframes, then restarts on reset().
///
/// Stateless w.r.t. the graph: it knows nothing about node/keyframe IDs,
/// factors, or the optimizer (ARCHITECTURE §6 #2). The node decides when a
/// keyframe is due (SLAM-01) and consumes current() as the BetweenFactor noise
/// model (SLAM-05); this class only accumulates motion + covariance.
class OdometryAccumulator {
 public:
  OdometryAccumulator();

  /// Restore the initial state: identity transform, zero covariance.
  void reset();

  /// Compound one scan-to-scan step. The accumulated pose becomes
  /// `delta_acc * delta_step`; the covariance propagates first-order as
  /// `Σ_new = J_a Σ_acc J_aᵀ + J_b Σ_step J_bᵀ`, where J_a, J_b are the GTSAM
  /// composition Jacobians w.r.t. the accumulated pose and the new step.
  void add_measurement(const gtsam::Pose3& delta_step, const gtsam::Matrix66& sigma_step);

  [[nodiscard]] AccumulatedOdometry current() const;

 private:
  gtsam::Pose3 delta_acc_;
  gtsam::Matrix66 sigma_acc_;
};

}  // namespace graph_slam::graph
