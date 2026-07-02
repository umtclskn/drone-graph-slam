#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>

#include <cstddef>
#include <vector>

namespace graph_slam::eval {

/// Ordered keyframe poses extracted from a GTSAM estimate (`Symbol('x', i)`).
struct GraphVisualization {
  std::vector<gtsam::Pose3> poses;

  [[nodiscard]] std::size_t nodeCount() const { return poses.size(); }

  /// Consecutive odometry edges: x_i — x_{i+1} for i = 0 .. N-2.
  [[nodiscard]] std::size_t odometryEdgeCount() const {
    return poses.size() > 1 ? poses.size() - 1 : 0;
  }
};

/// ROS-free graph visualization extractor (GRAPH-VIS-01). Reads
/// `GraphOptimizer::estimate()` output; does not own or modify graph state.
class GraphVisualizer {
 public:
  /// Collect poses in key order x0, x1, … until the first missing `Symbol('x', i)`.
  [[nodiscard]] static GraphVisualization fromEstimate(const gtsam::Values& estimate);
};

}  // namespace graph_slam::eval
