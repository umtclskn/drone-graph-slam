#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>

namespace graph_slam::graph {

/// Converts accumulated keyframe-to-keyframe odometry (SLAM-03) into a GTSAM
/// odometry factor (ARCHITECTURE §7). Stateless: no graph, Values, or optimizer.
class OdometryFactorBuilder {
 public:
  [[nodiscard]] gtsam::BetweenFactor<gtsam::Pose3> create_factor(
      gtsam::Key from_key,
      gtsam::Key to_key,
      const gtsam::Pose3& delta,
      const gtsam::Matrix66& sigma_prop) const;
};

}  // namespace graph_slam::graph
