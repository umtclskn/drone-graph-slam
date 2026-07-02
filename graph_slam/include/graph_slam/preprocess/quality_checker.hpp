#pragma once

#include <string>

#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// Plain configuration for the input-side scan quality gate (NDT-04). No ROS, no
/// hidden defaults; thresholds are setup-dependent and meant to be tuned via
/// INFRA-02 -- the defaults are only sane starting points.
struct QualityConfig {
  /// Minimum number of points in the (preprocessed) cloud. Below this the scan is
  /// too sparse for NDT to build meaningful per-voxel Gaussians.
  int min_points = 100;

  /// Smallest allowed eigenvalue of the point-position covariance (m^2). A single
  /// plane or line collapses one (or two) eigenvalue(s) to ~0, leaving the scan
  /// geometrically degenerate -> NDT cannot constrain all directions and may not
  /// converge. Rejecting here keeps that data out of the pipeline.
  double min_spread_eigenvalue = 0.1;
};

/// Verdict of the quality gate: is this scan usable, and if not, why.
struct QualityResult {
  bool accepted = false;
  std::string reason;  ///< human-readable rejection cause; empty when accepted.
};

/// Input-side scan quality gate (NDT-04). Runs AFTER the Preprocessor and BEFORE
/// the VoxelGridBuilder (ARCHITECTURE §4 pipeline): rejects scans that are too
/// sparse (point count) or too symmetric/planar (PCA spread) for NDT to register.
///
/// This complements -- does not duplicate -- the output-side RegistrationGate
/// (NDT-11): this stops bad data BEFORE NDT runs; NDT-11 judges the result AFTER.
///
/// Pure algorithm, ROS-free; mirrors Preprocessor (config-carrying class in the
/// same `preprocess/` group). Uses PCL/Eigen only as a linear-algebra toolkit.
class QualityChecker {
 public:
  explicit QualityChecker(QualityConfig config) : config_(config) {}

  /// Accept or reject `cloud`. A null or under-sized cloud is rejected on the
  /// point-count check before any covariance is computed. The cloud is never
  /// mutated (taken by const pointer).
  QualityResult check(const CloudConstPtr& cloud) const;

 private:
  QualityConfig config_;
};

}  // namespace graph_slam
