#include "graph_slam/graph/graph_optimizer.hpp"

#include <exception>

#include <gtsam/inference/Symbol.h>

namespace graph_slam::graph {

GraphOptimizer::GraphOptimizer() = default;

void GraphOptimizer::add_factors(const gtsam::NonlinearFactorGraph& factors,
                                 const gtsam::Values& new_values) {
  graph_.add(factors);
  pending_factors_.add(factors);
  values_.insert(new_values);
  pending_values_.insert(new_values);
}

void GraphOptimizer::add_odometry(const gtsam::BetweenFactor<gtsam::Pose3>& factor,
                                    gtsam::Key new_key,
                                    const gtsam::Pose3& initial_estimate) {
  graph_.add(factor);
  pending_factors_.add(factor);
  values_.insert(new_key, initial_estimate);
  pending_values_.insert(new_key, initial_estimate);
}

void GraphOptimizer::add_imu_keyframe(
    gtsam::Key pose_key, const gtsam::Pose3& pose_init,
    gtsam::Key velocity_key, const gtsam::Velocity3& velocity_init,
    gtsam::Key bias_key, const gtsam::imuBias::ConstantBias& bias_init,
    const gtsam::ImuFactor& imu_factor,
    const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>& bias_factor) {
  // New node values (X/V/B) — inserted together so the ImuFactor (references
  // X_j, V_j) and the bias BetweenFactor (references B_j) both find their keys
  // in the same iSAM2 update().
  values_.insert(pose_key, pose_init);
  values_.insert(velocity_key, velocity_init);
  values_.insert(bias_key, bias_init);
  pending_values_.insert(pose_key, pose_init);
  pending_values_.insert(velocity_key, velocity_init);
  pending_values_.insert(bias_key, bias_init);
  graph_.add(imu_factor);
  graph_.add(bias_factor);
  pending_factors_.add(imu_factor);
  pending_factors_.add(bias_factor);
}

void GraphOptimizer::add_ndt_edge(const gtsam::BetweenFactor<gtsam::Pose3>& factor) {
  // Cross-edge between two existing keyframes (node j already inserted by
  // add_imu_keyframe): factor only, no new Values.
  graph_.add(factor);
  pending_factors_.add(factor);
}

void GraphOptimizer::add_loop_closure(const gtsam::BetweenFactor<gtsam::Pose3>& factor) {
  // Cross-edge between two existing keyframes: factor only, no new Values.
  graph_.add(factor);
  pending_factors_.add(factor);
}

void GraphOptimizer::update() {
  if (pending_factors_.empty() && pending_values_.empty()) {
    return;
  }
  isam2_.update(pending_factors_, pending_values_);
  pending_factors_.resize(0);
  pending_values_.clear();
}

gtsam::Values GraphOptimizer::estimate() const {
  return isam2_.calculateEstimate();
}

NodeState GraphOptimizer::nodeEstimate(std::size_t index) const {
  using gtsam::symbol_shorthand::B;
  using gtsam::symbol_shorthand::V;
  using gtsam::symbol_shorthand::X;

  NodeState state;
  if (!values_.exists(X(index))) {
    return state;  // absent node -> defaults + zero covariance (documented)
  }
  const gtsam::Values est = isam2_.calculateEstimate();
  state.pose = est.at<gtsam::Pose3>(X(index));
  if (est.exists(V(index))) {
    state.velocity = est.at<gtsam::Velocity3>(V(index));
  }
  if (est.exists(B(index))) {
    state.bias = est.at<gtsam::imuBias::ConstantBias>(B(index));
  }
  state.covariance = marginalCovariance(X(index));
  return state;
}

bool GraphOptimizer::contains(gtsam::Key key) const {
  return values_.exists(key);
}

double GraphOptimizer::chi2() const {
  return graph_.error(isam2_.calculateEstimate());
}

double GraphOptimizer::marginalCovPositionTrace(gtsam::Key key) const {
  if (!values_.exists(key)) {
    return -1.0;
  }
  const gtsam::Matrix cov = isam2_.marginalCovariance(key);
  return cov(3, 3) + cov(4, 4) + cov(5, 5);
}

Eigen::Matrix<double, 6, 6> GraphOptimizer::marginalCovariance(gtsam::Key key) const {
  if (!values_.exists(key)) {
    return Eigen::Matrix<double, 6, 6>::Zero();
  }
  try {
    return isam2_.marginalCovariance(key);
  } catch (const std::exception&) {
    return Eigen::Matrix<double, 6, 6>::Zero();
  }
}

}  // namespace graph_slam::graph
