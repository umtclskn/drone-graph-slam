#include "graph_slam/graph/odometry_factor_builder.hpp"

#include <gtsam/linear/LossFunctions.h>
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

gtsam::BetweenFactor<gtsam::Pose3> OdometryFactorBuilder::create_robust_factor(
    const gtsam::Key from_key,
    const gtsam::Key to_key,
    const gtsam::Pose3& delta,
    const gtsam::Matrix66& sigma_prop,
    const double huber_k) const {
  const auto gaussian = gtsam::noiseModel::Gaussian::Covariance(sigma_prop);
  const auto robust = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(huber_k), gaussian);
  return gtsam::BetweenFactor<gtsam::Pose3>(from_key, to_key, delta, robust);
}

}  // namespace graph_slam::graph
