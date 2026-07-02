#include "graph_slam/graph/graph_bootstrap.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>

namespace graph_slam::graph {

BootstrapResult GraphBootstrap::create(
    const gtsam::Pose3& initial_pose,
    const gtsam::SharedNoiseModel& prior_noise) const {
  BootstrapResult result;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  result.values.insert(x0, initial_pose);
  result.graph.add(gtsam::PriorFactor<gtsam::Pose3>(x0, initial_pose, prior_noise));
  return result;
}

}  // namespace graph_slam::graph
