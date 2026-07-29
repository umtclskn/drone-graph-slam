#include "graph_slam/graph/graph_bootstrap.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/slam/PriorFactor.h>

namespace graph_slam::graph {

BootstrapResult GraphBootstrap::create(
    const gtsam::Pose3& initial_pose,
    const gtsam::SharedNoiseModel& pose_prior_noise,
    const gtsam::SharedNoiseModel& velocity_prior_noise,
    const gtsam::SharedNoiseModel& bias_prior_noise) const {
  using gtsam::symbol_shorthand::B;
  using gtsam::symbol_shorthand::V;
  using gtsam::symbol_shorthand::X;

  const gtsam::Velocity3 initial_velocity = gtsam::Velocity3::Zero();
  const gtsam::imuBias::ConstantBias initial_bias;

  BootstrapResult result;
  result.values.insert(X(0), initial_pose);
  result.values.insert(V(0), initial_velocity);
  result.values.insert(B(0), initial_bias);
  result.graph.add(
      gtsam::PriorFactor<gtsam::Pose3>(X(0), initial_pose, pose_prior_noise));
  result.graph.add(gtsam::PriorFactor<gtsam::Velocity3>(V(0), initial_velocity,
                                                        velocity_prior_noise));
  result.graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
      B(0), initial_bias, bias_prior_noise));
  return result;
}

}  // namespace graph_slam::graph
