#pragma once

#include <Eigen/Core>

#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// Plain configuration for the preprocessing pipeline. No ROS, no defaults that
/// hide behaviour: tests construct one directly with literal values.
///
/// The default crop box is intentionally huge so that, left untouched, cropping
/// is a no-op and only voxel downsampling + non-finite removal take effect.
struct PreprocessConfig {
  /// VoxelGrid leaf edge length in metres (applied on x, y and z).
  float voxel_leaf = 0.2F;

  /// Inclusive crop bounds in the cloud frame; w component is ignored by CropBox
  /// but PCL expects a 4-vector.
  Eigen::Vector4f crop_min{-1e3F, -1e3F, -1e3F, 1.0F};
  Eigen::Vector4f crop_max{1e3F, 1e3F, 1e3F, 1.0F};
};

/// Single-responsibility cloud cleaner: removes non-finite points, crops to a
/// box, and voxel-downsamples. Pure PCL, ROS-free; the owning node decides what
/// to feed it and what to do with the result.
class Preprocessor {
 public:
  explicit Preprocessor(PreprocessConfig config) : config_(config) {}

  /// Returns a new filtered cloud (non-finite removed -> cropped -> downsampled).
  /// The input is never mutated (taken by const pointer); the input header
  /// (stamp, frame_id) is preserved on the output. A null or empty input yields
  /// an empty cloud.
  CloudPtr process(const CloudConstPtr& input) const;

 private:
  PreprocessConfig config_;
};

}  // namespace graph_slam
