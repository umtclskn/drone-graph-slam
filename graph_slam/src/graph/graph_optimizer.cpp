#include "graph_slam/graph/graph_optimizer.hpp"

#include <exception>

namespace graph_slam::graph {

GraphOptimizer::GraphOptimizer() = default;

void GraphOptimizer::add_prior(const gtsam::PriorFactor<gtsam::Pose3>& factor,
                               gtsam::Key key,
                               const gtsam::Pose3& initial_estimate) {
  graph_.add(factor);
  pending_factors_.add(factor);
  values_.insert(key, initial_estimate);
  pending_values_.insert(key, initial_estimate);
}

void GraphOptimizer::add_odometry(const gtsam::BetweenFactor<gtsam::Pose3>& factor,
                                    gtsam::Key new_key,
                                    const gtsam::Pose3& initial_estimate) {
  graph_.add(factor);
  pending_factors_.add(factor);
  values_.insert(new_key, initial_estimate);
  pending_values_.insert(new_key, initial_estimate);
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
