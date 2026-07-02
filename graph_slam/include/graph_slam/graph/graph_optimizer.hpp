#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace graph_slam::graph {

/// Incremental pose-graph optimizer (SLAM-07). Owns the factor graph, initial
/// estimates, and iSAM2 solver. Factors and initial guesses are supplied by the
/// caller; this class does not create priors, odometry factors, or covariances.
class GraphOptimizer {
 public:
  GraphOptimizer();

  void add_prior(const gtsam::PriorFactor<gtsam::Pose3>& factor,
                 gtsam::Key key,
                 const gtsam::Pose3& initial_estimate);

  void add_odometry(const gtsam::BetweenFactor<gtsam::Pose3>& factor,
                    gtsam::Key new_key,
                    const gtsam::Pose3& initial_estimate);

  /// Add a loop-closure BetweenFactor between two EXISTING keyframes (SLAM-10).
  /// Unlike add_odometry, it inserts NO new Values — both keys already live in
  /// the graph; the factor is a cross-edge that iSAM2 re-optimizes on the next
  /// update(), pulling the looped poses back together (ARCHITECTURE §8).
  void add_loop_closure(const gtsam::BetweenFactor<gtsam::Pose3>& factor);

  /// Push pending factors/values into iSAM2 and optimize. No-op when nothing
  /// is pending (including on an empty graph).
  void update();

  [[nodiscard]] gtsam::Values estimate() const;

  [[nodiscard]] bool contains(gtsam::Key key) const;

  /// chi2 of the full graph: graph.error(current estimate) (SLAM-11).
  /// 0 for a perfectly consistent open chain; grows when conflicting
  /// constraints (e.g. loop closures) cannot be satisfied simultaneously.
  [[nodiscard]] double chi2() const;

  /// Trace of the 3x3 position block of the marginal covariance for `key`
  /// (Sigma_post, ARCHITECTURE §6 concept #3). GTSAM Pose3 tangent order is
  /// [rx,ry,rz,x,y,z], so the position block is indices 3..5. Returns -1.0 if
  /// `key` is not in the graph.
  [[nodiscard]] double marginalCovPositionTrace(gtsam::Key key) const;

  /// Full 6x6 marginal covariance (Sigma_post, ARCHITECTURE §6 concept #3) for
  /// `key`, in GTSAM Pose3 tangent order [rx,ry,rz,x,y,z]. Returns the zero
  /// matrix if `key` is absent or the marginal computation fails (EVAL-05).
  [[nodiscard]] Eigen::Matrix<double, 6, 6> marginalCovariance(gtsam::Key key) const;

 private:
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  gtsam::ISAM2 isam2_;

  gtsam::NonlinearFactorGraph pending_factors_;
  gtsam::Values pending_values_;
};

}  // namespace graph_slam::graph
