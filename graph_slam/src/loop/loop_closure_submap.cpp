#include "graph_slam/loop/loop_closure_submap.hpp"

#include <memory>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "graph_slam/initial_guess.hpp"
#include "graph_slam/point_types.hpp"

namespace graph_slam {

CloudPtr buildLoopClosureTarget(const std::vector<LoopSubmapEntry>& window,
                                const Eigen::Matrix4f& pose_ref, double voxel_leaf) {
  auto merged = std::make_shared<Cloud>();
  for (const LoopSubmapEntry& entry : window) {
    if (!entry.cloud) {
      continue;
    }
    Cloud transformed;
    pcl::transformPointCloud(*entry.cloud, transformed,
                             relativePoseGuess(pose_ref, entry.pose));
    *merged += transformed;
  }

  if (merged->empty()) {
    return merged;
  }

  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(merged);
  const auto leaf = static_cast<float>(voxel_leaf);
  voxel.setLeafSize(leaf, leaf, leaf);
  auto downsampled = std::make_shared<Cloud>();
  voxel.filter(*downsampled);
  return downsampled;
}

}  // namespace graph_slam
