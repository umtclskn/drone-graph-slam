#pragma once

#include <cstddef>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace graph_slam::graph {

/// One keyframe's optimized state (L5-02 node model): pose + velocity + IMU
/// bias, plus the 6x6 pose marginal Sigma_post (ARCHITECTURE §6 concept #3,
/// GTSAM tangent order [rx,ry,rz,x,y,z] — the quantity the EVAL-05 CSV logs).
/// Velocity/bias read back as zero if their keys are absent (pose-only graphs,
/// e.g. pre-L5-02 unit tests).
struct NodeState {
  gtsam::Pose3 pose;
  gtsam::Velocity3 velocity{gtsam::Velocity3::Zero()};
  gtsam::imuBias::ConstantBias bias;
  Eigen::Matrix<double, 6, 6> covariance{Eigen::Matrix<double, 6, 6>::Zero()};
};

/// Incremental graph optimizer (SLAM-07; node model expanded in L5-02 to
/// (Pose3, Velocity3, imuBias::ConstantBias) per keyframe — keys X(i)/V(i)/B(i)
/// via gtsam::symbol_shorthand). Owns the factor graph, initial estimates, and
/// iSAM2 solver. Factors and initial guesses are supplied by the caller; this
/// class does not create priors, odometry factors, or covariances.
class GraphOptimizer {
 public:
  GraphOptimizer();

  /// Add a caller-built prior factor + the initial estimate for its NEW key.
  /// ValueT is any node-state type: gtsam::Pose3 (X keys), gtsam::Velocity3
  /// (V keys), gtsam::imuBias::ConstantBias (B keys).
  template <typename ValueT>
  void add_prior(const gtsam::PriorFactor<ValueT>& factor,
                 gtsam::Key key,
                 const ValueT& initial_estimate) {
    graph_.add(factor);
    pending_factors_.add(factor);
    values_.insert(key, initial_estimate);
    pending_values_.insert(key, initial_estimate);
  }

  /// Ingest already-built factors plus the initial estimates of the NEW keys
  /// they introduce (e.g. the whole GraphBootstrap result: three priors +
  /// X(0)/V(0)/B(0) values). Keys already in the graph must not reappear in
  /// `new_values`.
  void add_factors(const gtsam::NonlinearFactorGraph& factors,
                   const gtsam::Values& new_values);

  void add_odometry(const gtsam::BetweenFactor<gtsam::Pose3>& factor,
                    gtsam::Key new_key,
                    const gtsam::Pose3& initial_estimate);

  /// L5-03: insert a NEW keyframe node j (pose/velocity/bias initial guesses) and
  /// attach the IMU backbone that constrains it: the `ImuFactor` coupling
  /// (X_i, V_i, X_j, V_j, B_i) and the bias random-walk
  /// `BetweenFactor<imuBias::ConstantBias>` (B_i, B_j). This is the unconditional
  /// per-keyframe backbone (LIO-SAM style); the NDT relative-pose edge is a
  /// separate, independent measurement added via add_ndt_edge(). The caller builds
  /// both factors (this class does not create noise models — same contract as
  /// add_prior/add_odometry).
  void add_imu_keyframe(
      gtsam::Key pose_key, const gtsam::Pose3& pose_init,
      gtsam::Key velocity_key, const gtsam::Velocity3& velocity_init,
      gtsam::Key bias_key, const gtsam::imuBias::ConstantBias& bias_init,
      const gtsam::ImuFactor& imu_factor,
      const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>& bias_factor);

  /// L5-03: add the NDT relative-pose measurement `BetweenFactor<Pose3>(X_i, X_j)`
  /// for an edge whose node j was already inserted by add_imu_keyframe. Factor
  /// only, no new Values — a second independent constraint alongside the IMU
  /// factor (no double-counting). Same wiring as add_loop_closure, distinct name
  /// so the call site reads as an odometry edge, not a loop.
  void add_ndt_edge(const gtsam::BetweenFactor<gtsam::Pose3>& factor);

  /// Add a loop-closure BetweenFactor between two EXISTING keyframes (SLAM-10).
  /// Unlike add_odometry, it inserts NO new Values — both keys already live in
  /// the graph; the factor is a cross-edge that iSAM2 re-optimizes on the next
  /// update(), pulling the looped poses back together (ARCHITECTURE §8).
  void add_loop_closure(const gtsam::BetweenFactor<gtsam::Pose3>& factor);

  /// Push pending factors/values into iSAM2 and optimize. No-op when nothing
  /// is pending (including on an empty graph).
  void update();

  /// Full current estimate (all X/V/B keys). Whole-graph consumers
  /// (GraphVisualizer, LoopClosureCandidateFinder) walk this; per-node reads
  /// should prefer nodeEstimate().
  [[nodiscard]] gtsam::Values estimate() const;

  /// Keyframe `index`'s optimized state as one struct (L5-02): pose from
  /// X(index), velocity from V(index), bias from B(index), plus the 6x6 pose
  /// marginal. Absent V/B keys read as zero; an absent X key returns a
  /// default-constructed NodeState with zero covariance (same convention as
  /// marginalCovariance).
  [[nodiscard]] NodeState nodeEstimate(std::size_t index) const;

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
