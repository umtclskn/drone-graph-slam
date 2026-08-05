#include "graph_slam/graph/ndt_consistency_gate.hpp"

#include <cmath>
#include <limits>

#include <gtsam/base/Matrix.h>

namespace graph_slam::graph {

Matrix6 poseBlockFromPreintCov(const Eigen::Matrix<double, 9, 9>& preint_cov) {
  // GTSAM PreintegratedImuMeasurements::preintMeasCov is 9×9 in order
  // [δθ(3), δp(3), δv(3)]. Pose3 tangent is [ω(3), ρ(3)] — same leading block.
  Matrix6 pose = Matrix6::Zero();
  pose.block<3, 3>(0, 0) = preint_cov.block<3, 3>(0, 0);
  pose.block<3, 3>(0, 3) = preint_cov.block<3, 3>(0, 3);
  pose.block<3, 3>(3, 0) = preint_cov.block<3, 3>(3, 0);
  pose.block<3, 3>(3, 3) = preint_cov.block<3, 3>(3, 3);
  return pose;
}

ConsistencyGateResult evaluateNdtConsistency(
    const gtsam::Pose3& x_pred,
    const gtsam::Pose3& x_ndt,
    const Matrix6& sigma_ndt,
    const Matrix6& sigma_x_pred,
    const ConsistencyGateConfig& config) {
  ConsistencyGateResult out;
  out.sigma_ndt_used = sigma_ndt;

  if (!config.enabled) {
    out.action = ConsistencyAction::None;
    out.add_ndt = true;
    return out;
  }

  // ξ = Log(x_pred⁻¹ · x_ndt) and J = ∂ξ/∂x_pred via chain rule of Logmap ∘ between.
  gtsam::Matrix6 H_between_pred, H_between_ndt;
  const gtsam::Pose3 between =
      x_pred.between(x_ndt, H_between_pred, H_between_ndt);
  gtsam::Matrix6 H_log;
  const gtsam::Vector6 xi = gtsam::Pose3::Logmap(between, H_log);
  const Matrix6 J = H_log * H_between_pred;  // ∂ξ/∂x_pred

  out.innovation = xi;

  const Matrix6 sigma_innov =
      sigma_ndt + J * sigma_x_pred * J.transpose();

  // Chi-square: ξᵀ Σ⁻¹ ξ. Use LDLT; fall back to a huge χ² if Σ is singular so
  // a degenerate innovation covariance never silently "accepts".
  Eigen::LDLT<Matrix6> ldlt(sigma_innov);
  if (ldlt.info() != Eigen::Success) {
    out.chi2 = std::numeric_limits<double>::infinity();
  } else {
    const Vector6 solved = ldlt.solve(xi);
    out.chi2 = xi.dot(solved);
    if (!std::isfinite(out.chi2) || out.chi2 < 0.0) {
      out.chi2 = std::numeric_limits<double>::infinity();
    }
  }

  const double threshold = std::max(config.chi2_threshold, 1e-12);
  if (out.chi2 <= threshold) {
    out.action = ConsistencyAction::Accept;
    out.add_ndt = true;
    return out;
  }

  if (config.mode == ConsistencyGateMode::Strict) {
    out.action = ConsistencyAction::Drop;
    out.add_ndt = false;
    return out;
  }

  // Soft: inflate Σ_ndt so the edge is weakly trusted, then proceed.
  const double scale = out.chi2 / threshold;
  out.sigma_ndt_used = sigma_ndt * scale;
  out.action = ConsistencyAction::Inflate;
  out.add_ndt = true;
  return out;
}

}  // namespace graph_slam::graph
