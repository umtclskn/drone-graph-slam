// L5-21: χ² consistency gate between IMU-predicted pose and NDT pose.
// Pure algorithm: no ROS. Used by graph_backend_node on the single-graph path
// before adding an NDT BetweenFactor<Pose3>.
#pragma once

#include <string>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>

namespace graph_slam::graph {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

/// χ²_0.999(6) — default high quantile for a 6-DoF Pose3 innovation.
inline constexpr double kDefaultConsistencyChi2Threshold = 22.457744484825323;

/// Huber k for ~95% asymptotic efficiency on Gaussian data (GTSAM default).
inline constexpr double kDefaultHuberK = 1.345;

enum class ConsistencyGateMode {
  Strict,  // drop NDT factor when χ² > threshold
  Soft,    // inflate Σ_ndt by χ²/threshold and proceed
};

enum class ConsistencyAction {
  None,     // gate not applied
  Accept,   // χ² within budget
  Drop,     // strict reject
  Inflate,  // soft scale of Σ_ndt
};

[[nodiscard]] inline const char* toString(ConsistencyAction action) {
  switch (action) {
    case ConsistencyAction::None:
      return "none";
    case ConsistencyAction::Accept:
      return "accept";
    case ConsistencyAction::Drop:
      return "drop";
    case ConsistencyAction::Inflate:
      return "inflate";
  }
  return "none";
}

struct ConsistencyGateConfig {
  bool enabled{false};  // off by default — slam_loop_03 strict-ON regresses ATE
  ConsistencyGateMode mode{ConsistencyGateMode::Strict};
  double chi2_threshold{kDefaultConsistencyChi2Threshold};
  double huber_k{kDefaultHuberK};
};

struct ConsistencyGateResult {
  Vector6 innovation{Vector6::Zero()};  // ξ = Log(x_pred⁻¹ · x_ndt)
  double chi2{-1.0};
  ConsistencyAction action{ConsistencyAction::None};
  Matrix6 sigma_ndt_used{Matrix6::Zero()};  // possibly inflated Σ for the factor
  bool add_ndt{true};  // false only on strict Drop
};

/// Extract the 6×6 pose block (rotation + position) from GTSAM's 9×9
/// preintegration covariance, order [δθ, δp, δv] → Pose3 [rx,ry,rz,x,y,z].
[[nodiscard]] Matrix6 poseBlockFromPreintCov(
    const Eigen::Matrix<double, 9, 9>& preint_cov);

/// χ² gate: Σ_innov = Σ_ndt + J Σ_x_pred Jᵀ with J = ∂ξ/∂x_pred,
/// ξ = Log(x_pred.between(x_ndt)). Soft mode scales Σ_ndt by chi2/threshold
/// when over budget; strict mode sets add_ndt=false.
[[nodiscard]] ConsistencyGateResult evaluateNdtConsistency(
    const gtsam::Pose3& x_pred,
    const gtsam::Pose3& x_ndt,
    const Matrix6& sigma_ndt,
    const Matrix6& sigma_x_pred,
    const ConsistencyGateConfig& config);

}  // namespace graph_slam::graph
