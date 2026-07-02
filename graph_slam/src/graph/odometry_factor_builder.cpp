#include "graph_slam/graph/odometry_factor_builder.hpp"

#include <gtsam/linear/NoiseModel.h>

namespace graph_slam::graph {

gtsam::BetweenFactor<gtsam::Pose3> OdometryFactorBuilder::create_factor(
    const gtsam::Key from_key,
    const gtsam::Key to_key,
    const gtsam::Pose3& delta,
    const gtsam::Matrix66& sigma_prop) const {
  const auto noise = gtsam::noiseModel::Gaussian::Covariance(sigma_prop);
  return gtsam::BetweenFactor<gtsam::Pose3>(from_key, to_key, delta, noise);
}

}  // namespace graph_slam::graph
