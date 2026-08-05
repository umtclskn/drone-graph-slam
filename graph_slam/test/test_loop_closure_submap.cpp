// L5-12: fusing a small window of historical keyframes into one loop-closure
// verification target. Pure geometry — deterministic clouds/poses, checked
// analytically (no NDT involved here; that is test_loop_closure_verifier.cpp).

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "graph_slam/loop/loop_closure_submap.hpp"

namespace {

using graph_slam::buildLoopClosureTarget;
using graph_slam::Cloud;
using graph_slam::CloudPtr;
using graph_slam::LoopSubmapEntry;
using graph_slam::PointT;

/// A single point at the origin of the entry's own local frame.
CloudPtr onePointAtOrigin() {
  auto cloud = std::make_shared<Cloud>();
  PointT p;
  p.x = 0.0F;
  p.y = 0.0F;
  p.z = 0.0F;
  p.intensity = 1.0F;
  cloud->push_back(p);
  cloud->width = 1;
  cloud->height = 1;
  return cloud;
}

Eigen::Matrix4f translation(float x, float y, float z) {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m(0, 3) = x;
  m(1, 3) = y;
  m(2, 3) = z;
  return m;
}

}  // namespace

TEST(LoopClosureSubmapTest, EmptyWindowGivesEmptyCloud) {
  const CloudPtr out = buildLoopClosureTarget({}, Eigen::Matrix4f::Identity(), 0.1);
  EXPECT_TRUE(out->empty());
}

TEST(LoopClosureSubmapTest, SkipsNullClouds) {
  std::vector<LoopSubmapEntry> window;
  window.push_back(LoopSubmapEntry{nullptr, Eigen::Matrix4f::Identity()});
  const CloudPtr out = buildLoopClosureTarget(window, Eigen::Matrix4f::Identity(), 0.1);
  EXPECT_TRUE(out->empty());
}

// Three keyframes, each a single point at its own local origin, placed 5 m apart
// along x in world frame; pose_ref = the middle one. Fusing with a leaf far
// smaller than 5 m must keep all three points distinct, each landing exactly at
// its world offset relative to pose_ref (T_ref^-1 * T_k applied to a point at
// the local origin recovers T_ref^-1 * T_k's translation).
TEST(LoopClosureSubmapTest, FusesWindowIntoMatchFrame) {
  std::vector<LoopSubmapEntry> window;
  window.push_back(LoopSubmapEntry{onePointAtOrigin(), translation(-5.0F, 0.0F, 0.0F)});
  window.push_back(LoopSubmapEntry{onePointAtOrigin(), translation(0.0F, 0.0F, 0.0F)});
  window.push_back(LoopSubmapEntry{onePointAtOrigin(), translation(5.0F, 0.0F, 0.0F)});

  const Eigen::Matrix4f pose_ref = translation(0.0F, 0.0F, 0.0F);  // the middle entry
  const CloudPtr out = buildLoopClosureTarget(window, pose_ref, /*voxel_leaf=*/0.1);

  ASSERT_EQ(out->size(), 3U);
  std::vector<float> xs;
  for (const PointT& p : *out) {
    xs.push_back(p.x);
  }
  std::sort(xs.begin(), xs.end());
  EXPECT_NEAR(xs[0], -5.0F, 1e-4F);
  EXPECT_NEAR(xs[1], 0.0F, 1e-4F);
  EXPECT_NEAR(xs[2], 5.0F, 1e-4F);
}

// Two keyframes whose points fall in the same voxel after transform must merge
// into one downsampled point (proves the voxel filter is actually applied, not
// just a plain concatenation).
TEST(LoopClosureSubmapTest, VoxelDownsampleMergesCoincidentPoints) {
  std::vector<LoopSubmapEntry> window;
  window.push_back(LoopSubmapEntry{onePointAtOrigin(), translation(0.0F, 0.0F, 0.0F)});
  window.push_back(LoopSubmapEntry{onePointAtOrigin(), translation(0.02F, 0.0F, 0.0F)});

  const CloudPtr out =
      buildLoopClosureTarget(window, Eigen::Matrix4f::Identity(), /*voxel_leaf=*/1.0);
  EXPECT_EQ(out->size(), 1U);
}
