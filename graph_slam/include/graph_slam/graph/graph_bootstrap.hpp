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

/// Creates node 0 (`Symbol('x', 0)`) with a single PriorFactor. Stateless: no
/// graph ownership, no optimizer, no keyframe logic. The caller supplies the
/// initial pose and prior noise (origin, GT, or EKF2 — bootstrap does not
/// decide the source).
class GraphBootstrap {
 public:
  [[nodiscard]] BootstrapResult create(
      const gtsam::Pose3& initial_pose,
      const gtsam::SharedNoiseModel& prior_noise) const;
};

}  // namespace graph_slam::graph
