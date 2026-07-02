#pragma once

#include <string>

#include <Eigen/Geometry>
#include <gtsam/linear/NoiseModel.h>

#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// Outcome of verifying one loop-closure candidate (SLAM-10). Plain data: the
/// verifier fills it, the back-end node decides what to do with it. Only
/// `relative_pose` and `noise_model` are meaningful when `accepted` is true.
struct VerificationResult {
  bool accepted = false;
  double ndt_score = 0.0;            ///< fitness from NdtRegistrar (<= 0; lower = better)
  double sigma_min_eigenvalue = 0.0;  ///< smallest eigenvalue of the cost Hessian
                                      ///< (= Sigma_meas^-1, the information matrix):
                                      ///< the worst-constrained direction's stiffness.
  Eigen::Isometry3d relative_pose = Eigen::Isometry3d::Identity();  ///< T_match_query
  gtsam::noiseModel::Gaussian::shared_ptr noise_model;  ///< from Sigma_meas (null unless accepted)
  std::string reason;  ///< why it was rejected (empty when accepted)
};

/// Verifies loop-closure candidates (SLAM-09 -> SLAM-10) by running NDT between
/// the two keyframe clouds. Accepts ONLY a strong, well-conditioned registration;
/// a false positive corrupts the whole graph (ARCHITECTURE §8), so the gate is
/// deliberately conservative — when in doubt, reject. Pure algorithm, ROS-free.
class LoopClosureVerifier {
 public:
  struct Params {
    /// Accept only if the NDT fitness is at or below this (more negative = better
    /// overlap). Stricter than the front-end gate's -0.5 — a loop edge must be a
    /// confident match, not merely a passable one.
    double min_ndt_score = -0.8;

    /// Floor on the smallest cost-Hessian eigenvalue. Below it a pose direction is
    /// (near-)unconstrained -> degenerate geometry -> reject. Same signal as the
    /// NDT-11 registration gate's min_hessian_eigenvalue.
    double min_sigma_eigenvalue = 1e-6;

    /// Largest allowed Hessian condition number. Above it one direction is far
    /// softer than the rest (corridor / plane with no distinctive features), so
    /// Sigma_meas is ill-conditioned -> reject (ARCHITECTURE §8 "well-conditioned").
    double max_condition_number = 1e4;

    NdtConfig ndt_config{};       ///< NDT optimiser knobs (defaults match the front-end).
    NdtGridConfig grid_config{};  ///< target voxel grid built from the match cloud.
  };

  LoopClosureVerifier();
  explicit LoopClosureVerifier(Params p);

  /// Verify a candidate by aligning `query_cloud` onto an NDT target built from
  /// `match_cloud`, seeded with `initial_guess` (T_world_match^-1 * T_world_query
  /// from the optimized estimate). The recovered transform maps query -> match,
  /// i.e. T_match_query, ready to become a BetweenFactor(match_key, query_key).
  [[nodiscard]] VerificationResult verify(const CloudPtr& query_cloud,
                                          const CloudPtr& match_cloud,
                                          const Eigen::Isometry3d& initial_guess) const;

 private:
  Params params_;
  NdtRegistrar registrar_;
};

}  // namespace graph_slam
