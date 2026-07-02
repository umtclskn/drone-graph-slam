// EVAL-03: verify the overlay's algorithm path (ROS-free). Runs the exact
// pipeline the node uses — Preprocessor -> NdtVoxelGrid -> NdtRegistrar ->
// transformPointCloud — and checks that aligned = T * source lands on target
// (the "blue overlaps green" claim, quantified). The RViz/matplotlib rendering
// is confirmed visually by the user.

#include <gtest/gtest.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <cmath>
#include <random>

#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocessor.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudConstPtr;
using graph_slam::CloudPtr;
using graph_slam::NdtConfig;
using graph_slam::NdtGridConfig;
using graph_slam::NdtRegistrar;
using graph_slam::NdtVoxelGrid;
using graph_slam::PointT;
using graph_slam::PreprocessConfig;
using graph_slam::Preprocessor;
using graph_slam::RegistrationResult;

CloudPtr makeRoom() {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(42);
  std::normal_distribution<float> jitter(0.0F, 0.01F);
  const auto add = [&](float x, float y, float z) {
    PointT p;
    p.x = x + jitter(rng);
    p.y = y + jitter(rng);
    p.z = z + jitter(rng);
    p.intensity = 1.0F;
    cloud->push_back(p);
  };
  constexpr float kStep = 0.1F;
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
      add(x, y, 0.0F);
      add(x, y, 3.0F);
    }
  }
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(x, 0.0F, z);
      add(x, 5.0F, z);
    }
  }
  for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(0.0F, y, z);
      add(5.0F, y, z);
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

Eigen::Vector4f centroid(const CloudConstPtr& c) {
  Eigen::Vector4f mean;
  pcl::compute3DCentroid(*c, mean);
  return mean;
}

// The pipeline the node runs; aligned = T*source must overlap target.
TEST(Eval03OverlayTest, AlignedSourceOverlapsTarget) {
  // Target = preprocessed room (what the node builds its NDT grid from). Source =
  // a clean rigid transform of it by a known small motion: this isolates the
  // registration + overlay math from voxel-binning aliasing of the synthetic
  // lattice (real, noisy scans don't alias like a perfect grid does).
  PreprocessConfig pre_cfg;
  pre_cfg.voxel_leaf = 0.2F;
  const CloudPtr target = Preprocessor{pre_cfg}.process(makeRoom());

  Eigen::Matrix4f motion = Eigen::Matrix4f::Identity();
  motion.block<3, 1>(0, 3) = Eigen::Vector3f(0.08F, -0.05F, 0.04F);  // known small shift
  auto source = std::make_shared<Cloud>();
  pcl::transformPointCloud(*target, *source, motion);

  NdtGridConfig grid_cfg;
  grid_cfg.resolution = 1.0;  // node default
  NdtVoxelGrid grid{grid_cfg};
  grid.build(target);

  NdtConfig ndt_cfg;
  ndt_cfg.max_iterations = 50;
  const RegistrationResult r = NdtRegistrar{ndt_cfg}.align(grid, source);
  ASSERT_TRUE(r.converged);

  auto aligned = std::make_shared<Cloud>();
  pcl::transformPointCloud(*source, *aligned, r.transform);  // aligned = T * source

  // Before alignment, source is offset from target by the motion; after applying
  // T, aligned should sit on target.
  const float before = (centroid(source) - centroid(target)).head<3>().norm();
  const float after = (centroid(aligned) - centroid(target)).head<3>().norm();
  const float motion_norm = motion.block<3, 1>(0, 3).norm();
  EXPECT_NEAR(before, motion_norm, 2e-2F);  // sanity on the setup
  EXPECT_LT(after, 0.03F);                  // aligned overlaps target
  EXPECT_LT(after, before * 0.3F);          // and is a big improvement over source
}

}  // namespace
