#pragma once

#include <Eigen/Core>

#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/registration_result.hpp"

namespace graph_slam {

/// Plain configuration for the hand-rolled NDT optimiser. No ROS, no hidden
/// defaults: tests construct one directly.
struct NdtConfig {
  int max_iterations = 35;
  double step_epsilon = 1e-4;    // converged once ||delta p|| drops below this
  double outlier_ratio = 0.55;   // Magnusson Gaussian/uniform mixture weight (d1/d2)
  double regularization = 1e-6;  // floor on the Hessian eigenvalues (PD guard)

  // --- Sigma_meas from the cost Hessian (NDT-12) ----------------------------
  double hessian_lambda = 1.0;      // Sigma_meas = hessian_lambda * H^-1 (unit scale)
  // Empirical NEES calibration (EVAL-05): the NDT Hessian has the right SHAPE but
  // the wrong SCALE, so Sigma_meas is overconfident. This scalar multiplies the
  // Hessian-based Sigma_meas (NOT the degenerate fallback). Default 1.0 = no
  // calibration; the front-end YAML sets the measured alpha (see analysis/nees_report.txt).
  double cov_scale_factor = 1.0;    // Sigma_meas = cov_scale_factor * hessian_lambda * H^-1
  double sigma_rot_fallback = 0.01;  // rad,  fallback rotation std (per-axis)
  double sigma_t_fallback = 0.1;     // m,    fallback translation std (per-axis)
  double sigma_max_condition = 1e6;  // H condition number above this -> fallback
};

/// Single-responsibility scan-to-scan aligner: our OWN NDT score + analytic
/// gradient/Hessian + Newton step (Magnusson 2009), consuming the NdtVoxelGrid
/// target Gaussians (NDT-05/06). Pure algorithm, ROS-free; does alignment only.
/// This is the production registrar; `PclNdtBaseline` is the cross-check only.
class NdtRegistrar {
 public:
  explicit NdtRegistrar(NdtConfig config) : config_(config) {}

  /// Align `source` onto the `target` NDT, seeded with `init_guess` (identity by
  /// default; in the pipeline this is the EKF2 relative-motion prior, NDT-08).
  /// Returns the recovered transform, convergence diagnostics, and the final cost
  /// Hessian (for NDT-12). An empty source or empty target grid yields a
  /// non-converged result with an identity transform.
  RegistrationResult align(const NdtVoxelGrid& target, const CloudConstPtr& source,
                           const Eigen::Matrix4f& init_guess = Eigen::Matrix4f::Identity()) const;

 private:
  NdtConfig config_;
};

/// Build the measurement covariance Sigma_meas (NDT-12) from the final NDT cost
/// Hessian `hessian` (parameter order [tx, ty, tz, roll, pitch, yaw], as stored in
/// RegistrationResult::hessian).
///
/// Primary path: Sigma = cov_scale_factor * lambda * H^-1 (ARCHITECTURE §6 #1),
/// then the two 3x3 blocks are swapped so the result is in GTSAM's Pose3 order
/// [rx, ry, rz, x, y, z]. (cov_scale_factor is the EVAL-05 NEES calibration.)
/// Fallback (`from_hessian == false`): when H is (near-)singular — any eigenvalue
/// <= 0 or condition number above `config.sigma_max_condition`, e.g. a single-plane
/// / corridor scene — a geometry-aware diagonal diag(sigma_rot^2 x3, sigma_t^2 x3)
/// is returned instead. Pure function, ROS-free; `align()` calls it to fill the
/// result, and the unit tests exercise it directly.
struct MeasurementCovariance {
  Matrix6f sigma = Matrix6f::Zero();
  bool from_hessian = false;
};
MeasurementCovariance measurementCovariance(const Eigen::Matrix<double, 6, 6>& hessian,
                                            const NdtConfig& config);

}  // namespace graph_slam
