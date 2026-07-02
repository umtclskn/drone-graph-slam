#pragma once

#include <Eigen/Core>

namespace graph_slam {

/// 6x6 single-precision matrix. Namespace-level so a type change is one line, not
/// a refactor. Used for the measurement covariance Sigma_meas below (NDT-12).
using Matrix6f = Eigen::Matrix<float, 6, 6>;

/// Outcome of one scan-to-scan registration (NDT-09). Plain data: the registrar
/// fills it, the caller (node / odometry estimator) decides what to do with it.
struct RegistrationResult {
  /// Rigid transform that maps the source cloud onto the target cloud.
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

  /// Objective value at the solution (lower = better fit). For NdtRegistrar this
  /// is the mean per-point Magnusson NDT score; for PclNdtBaseline it is pcl's
  /// mean-squared nearest-neighbour fitness. Not comparable across the two.
  double fitness_score = 0.0;

  /// Whether the optimiser reported convergence.
  bool converged = false;

  /// Iterations actually run before stopping.
  int iterations = 0;

  /// Final 6x6 cost Hessian at the solution, parameter order [tx, ty, tz, roll,
  /// pitch, yaw]. Retained because NDT-12 builds the measurement covariance
  /// Sigma_meas ~= s * H^-1 from it (ARCHITECTURE §6 concept #1). NDT-09 does NOT
  /// compute Sigma_meas — it only keeps H here. Filled by NdtRegistrar; left zero
  /// by PclNdtBaseline (pcl does not expose its Hessian).
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();

  /// Measurement covariance (NDT-12), Pose3 parameter order [rx, ry, rz, x, y, z]
  /// to match GTSAM — NOTE this is the rotation-first order, NOT the [t, r] order
  /// of `hessian` above; the registrar reorders the blocks when it fills this.
  /// Downstream (Sprint 3) this becomes the BetweenFactor noise model; NDT-12 only
  /// computes it. Zero on the null/empty early-out (a non-converged result is not a
  /// usable measurement).
  Matrix6f sigma_meas = Matrix6f::Zero();

  /// true  -> sigma_meas came from the cost Hessian (Sigma = lambda * H^-1);
  /// false -> the Hessian was (near-)singular, so the geometry-aware diagonal
  /// fallback was used. The node logs this per registration (NDT-12).
  bool sigma_from_hessian = false;
};

}  // namespace graph_slam
