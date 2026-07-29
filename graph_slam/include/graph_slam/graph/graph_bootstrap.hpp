#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

namespace graph_slam::graph {

/// Initialized factor graph + initial estimate from bootstrap (SLAM-04).
/// Factors live in `graph`; estimates live in `values` — kept separate per
/// ARCHITECTURE §7.
struct BootstrapResult {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
};

/// Creates node 0 with the expanded L5-02 node model — X(0) = initial pose,
/// V(0) = zero velocity, B(0) = zero IMU bias — each held by its own
/// PriorFactor (iSAM2 rejects factorless variables). Stateless: no graph
/// ownership, no optimizer, no keyframe logic. The caller supplies the initial
/// pose and all three prior noise models (bootstrap does not decide the
/// source; graph_backend_node supplies the principled L5-04 sigmas via the
/// bootstrap_* parameters — pose tight at the first NDT-odom pose, velocity
/// near zero for the at-rest start, bias zero-mean). B(0)'s prior is the
/// anchor of the bias chain that L5-03's BetweenFactor<imuBias> extends.
class GraphBootstrap {
 public:
  [[nodiscard]] BootstrapResult create(
      const gtsam::Pose3& initial_pose,
      const gtsam::SharedNoiseModel& pose_prior_noise,
      const gtsam::SharedNoiseModel& velocity_prior_noise,
      const gtsam::SharedNoiseModel& bias_prior_noise) const;
};

}  // namespace graph_slam::graph
