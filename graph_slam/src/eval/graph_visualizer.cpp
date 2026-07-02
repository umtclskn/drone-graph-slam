#include "graph_slam/eval/graph_visualizer.hpp"

#include <gtsam/inference/Symbol.h>

namespace graph_slam::eval {

GraphVisualization GraphVisualizer::fromEstimate(const gtsam::Values& estimate) {
  GraphVisualization viz;
  for (std::size_t i = 0;; ++i) {
    const gtsam::Key key = gtsam::Symbol('x', static_cast<uint64_t>(i));
    if (!estimate.exists(key)) {
      break;
    }
    viz.poses.push_back(estimate.at<gtsam::Pose3>(key));
  }
  return viz;
}

}  // namespace graph_slam::eval
