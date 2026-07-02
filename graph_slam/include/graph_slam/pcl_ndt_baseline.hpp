#pragma once

#include <Eigen/Core>

#include "graph_slam/point_types.hpp"
#include "graph_slam/registration_result.hpp"

namespace graph_slam {

/// Config for the PCL NDT baseline (the `pcl::NormalDistributionsTransform`
/// defaults from the PCL tutorial).
struct PclNdtConfig {
  double resolution = 1.0;               // NDT cell edge length [m]
  double step_size = 0.1;                // More-Thuente line-search max step [m]
  double transformation_epsilon = 0.01;  // convergence threshold on the transform
  int max_iterations = 35;
};

/// Reference scan-to-scan aligner wrapping `pcl::NormalDistributionsTransform`.
/// Kept as an OPTIONAL BASELINE only: the production registrar (NDT-09) is the
/// hand-rolled `NdtRegistrar`; this one exists so tests can cross-check our NDT
/// against a trusted implementation, and for later performance comparison
/// (ARCHITECTURE §4 treats pcl::NDT / ndt_omp as the swap-in baseline).
class PclNdtBaseline {
 public:
  explicit PclNdtBaseline(PclNdtConfig config) : config_(config) {}

  /// Align `source` onto `target`, seeded with `init_guess`. A null/empty source
  /// or target yields a non-converged result with an identity transform. The
  /// returned `hessian` is left zero — only the hand-rolled NdtRegistrar exposes it.
  RegistrationResult align(const CloudConstPtr& source, const CloudConstPtr& target,
                           const Eigen::Matrix4f& init_guess = Eigen::Matrix4f::Identity()) const;

 private:
  PclNdtConfig config_;
};

}  // namespace graph_slam
