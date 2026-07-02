#include "graph_slam/preprocessor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "graph_slam/point_types.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudPtr;
using graph_slam::PointT;
using graph_slam::Preprocessor;
using graph_slam::PreprocessConfig;

/// Dense 1 m x 1 m planar patch at z = 0, spaced every `step` metres.
CloudPtr makePlane(float step) {
  auto cloud = std::make_shared<Cloud>();
  for (float x = 0.0F; x <= 1.0F + 1e-4F; x += step) {
    for (float y = 0.0F; y <= 1.0F + 1e-4F; y += step) {
      PointT p;
      p.x = x;
      p.y = y;
      p.z = 0.0F;
      p.intensity = 1.0F;
      cloud->push_back(p);
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

// --- count drops as expected -----------------------------------------------
TEST(PreprocessorTest, VoxelDownsampleReducesCount) {
  const auto in = makePlane(0.02F);  // ~2601 points
  PreprocessConfig cfg;
  cfg.voxel_leaf = 0.1F;  // coarser than point spacing -> must collapse points
  const auto out = Preprocessor{cfg}.process(in);

  EXPECT_GT(out->size(), 0U);
  EXPECT_LT(out->size(), in->size());
}

// --- geometry preserved on a non-empty scene -------------------------------
TEST(PreprocessorTest, GeometryPreservedOnPlane) {
  const auto in = makePlane(0.02F);
  PreprocessConfig cfg;
  cfg.voxel_leaf = 0.1F;
  const auto out = Preprocessor{cfg}.process(in);

  ASSERT_GT(out->size(), 0U);
  float cx = 0.0F;
  float cy = 0.0F;
  for (const auto& p : out->points) {
    EXPECT_NEAR(p.z, 0.0F, 1e-4F);   // stays on the plane
    EXPECT_GE(p.x, -1e-4F);          // stays within original extent
    EXPECT_LE(p.x, 1.0F + 1e-4F);
    EXPECT_GE(p.y, -1e-4F);
    EXPECT_LE(p.y, 1.0F + 1e-4F);
    cx += p.x;
    cy += p.y;
  }
  cx /= static_cast<float>(out->size());
  cy /= static_cast<float>(out->size());
  EXPECT_NEAR(cx, 0.5F, 0.05F);  // centroid of a symmetric patch is preserved
  EXPECT_NEAR(cy, 0.5F, 0.05F);
}

// --- voxel size effect verified --------------------------------------------
TEST(PreprocessorTest, LargerLeafYieldsFewerPoints) {
  const auto in = makePlane(0.02F);

  PreprocessConfig fine;
  fine.voxel_leaf = 0.05F;
  PreprocessConfig coarse;
  coarse.voxel_leaf = 0.2F;

  const auto fine_out = Preprocessor{fine}.process(in);
  const auto coarse_out = Preprocessor{coarse}.process(in);

  EXPECT_GT(coarse_out->size(), 0U);
  EXPECT_LT(coarse_out->size(), fine_out->size());
}

// --- crop box removes out-of-bounds points ---------------------------------
TEST(PreprocessorTest, CropBoxRemovesOutOfBoundsPoints) {
  auto in = makePlane(0.1F);
  const std::size_t inside = in->size();
  // Add clearly-outside points far beyond the crop box.
  for (int i = 0; i < 10; ++i) {
    PointT p;
    p.x = 100.0F + static_cast<float>(i);
    p.y = 100.0F;
    p.z = 0.0F;
    p.intensity = 1.0F;
    in->push_back(p);
  }

  PreprocessConfig cfg;
  cfg.voxel_leaf = 0.01F;  // fine enough not to merge the unit-square points
  cfg.crop_min = Eigen::Vector4f{-0.5F, -0.5F, -0.5F, 1.0F};
  cfg.crop_max = Eigen::Vector4f{1.5F, 1.5F, 1.5F, 1.0F};
  const auto out = Preprocessor{cfg}.process(in);

  // Every surviving point must be within the crop box; the far points are gone.
  for (const auto& p : out->points) {
    EXPECT_LE(p.x, 1.5F + 1e-4F);
    EXPECT_GE(p.x, -0.5F - 1e-4F);
  }
  EXPECT_EQ(out->size(), inside);  // exactly the in-box points remain
}

// --- header (stamp + frame_id) preserved on the output ---------------------
TEST(PreprocessorTest, HeaderPreserved) {
  auto in = makePlane(0.05F);
  in->header.stamp = 1234567890ULL;  // PCL stamp is microseconds
  in->header.frame_id = "lidar_link";

  PreprocessConfig cfg;
  cfg.voxel_leaf = 0.1F;
  const auto out = Preprocessor{cfg}.process(in);

  ASSERT_GT(out->size(), 0U);
  EXPECT_EQ(out->header.stamp, in->header.stamp);
  EXPECT_EQ(out->header.frame_id, in->header.frame_id);
}

// --- empty / NaN / inf inputs handled gracefully ---------------------------
TEST(PreprocessorTest, EmptyInputYieldsEmptyOutput) {
  PreprocessConfig cfg;
  const Preprocessor pre{cfg};

  EXPECT_EQ(pre.process(std::make_shared<Cloud>())->size(), 0U);
  EXPECT_EQ(pre.process(nullptr)->size(), 0U);
}

TEST(PreprocessorTest, NonFinitePointsRemoved) {
  auto in = makePlane(0.1F);
  const std::size_t finite_count = in->size();

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (float bad : {nan, inf, -inf}) {
    PointT p;
    p.x = bad;
    p.y = bad;
    p.z = bad;
    p.intensity = 1.0F;
    in->push_back(p);
  }

  PreprocessConfig cfg;
  cfg.voxel_leaf = 0.01F;  // fine: keeps the unit-square points distinct
  const auto out = Preprocessor{cfg}.process(in);

  ASSERT_GT(out->size(), 0U);
  for (const auto& p : out->points) {
    EXPECT_TRUE(std::isfinite(p.x));
    EXPECT_TRUE(std::isfinite(p.y));
    EXPECT_TRUE(std::isfinite(p.z));
  }
  EXPECT_EQ(out->size(), finite_count);  // the 3 bad points are gone
}

}  // namespace
