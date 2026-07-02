#include "graph_slam/preprocessor.hpp"

#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <vector>

namespace graph_slam {

CloudPtr Preprocessor::process(const CloudConstPtr& input) const {
  auto output = std::make_shared<Cloud>();
  if (!input || input->empty()) {
    return output;
  }

  // 1. Drop non-finite points (NaN and +/-inf). removeNaNFromPointCloud only
  //    runs its std::isfinite check when is_dense is false, so copy the input
  //    and clear the flag: this defends against a cloud that claims density yet
  //    still carries non-finite points, without mutating the caller's cloud.
  auto finite = std::make_shared<Cloud>(*input);
  finite->is_dense = false;
  std::vector<int> kept;
  pcl::removeNaNFromPointCloud(*finite, *finite, kept);

  // 2. Crop to the configured box.
  auto cropped = std::make_shared<Cloud>();
  pcl::CropBox<PointT> crop;
  crop.setInputCloud(finite);
  crop.setMin(config_.crop_min);
  crop.setMax(config_.crop_max);
  crop.filter(*cropped);

  // 3. Voxel-grid downsample.
  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(cropped);
  voxel.setLeafSize(config_.voxel_leaf, config_.voxel_leaf, config_.voxel_leaf);
  voxel.filter(*output);

  output->header = input->header;
  return output;
}

}  // namespace graph_slam
