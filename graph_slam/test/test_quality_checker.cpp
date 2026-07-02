#include <gtest/gtest.h>

#include <random>

#include "graph_slam/point_types.hpp"
#include "graph_slam/preprocess/quality_checker.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudPtr;
using graph_slam::PointT;
using graph_slam::QualityChecker;
using graph_slam::QualityConfig;
using graph_slam::QualityResult;

// Small jitter keeps a "degenerate" axis from being exactly singular, so the test
// exercises the threshold rather than a trivial zero.
PointT makePoint(float x, float y, float z) {
  PointT p;
  p.x = x;
  p.y = y;
  p.z = z;
  p.intensity = 1.0F;
  return p;
}

// A solid axis-aligned box of points: good spread in all three directions.
CloudPtr makeBox(int per_axis, float extent) {
  auto cloud = std::make_shared<Cloud>();
  const float step = extent / static_cast<float>(per_axis - 1);
  for (int i = 0; i < per_axis; ++i) {
    for (int j = 0; j < per_axis; ++j) {
      for (int k = 0; k < per_axis; ++k) {
        cloud->push_back(makePoint(i * step, j * step, k * step));
      }
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

// A single flat plane at z=0: x,y spread is large but z spread is ~0.
CloudPtr makePlane(int per_axis, float extent) {
  auto cloud = std::make_shared<Cloud>();
  const float step = extent / static_cast<float>(per_axis - 1);
  for (int i = 0; i < per_axis; ++i) {
    for (int j = 0; j < per_axis; ++j) {
      cloud->push_back(makePoint(i * step, j * step, 0.0F));
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

// A corridor: long in x, negligibly thin in y and z (one-directional spread).
CloudPtr makeCorridor(int n, float length) {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(11);
  std::normal_distribution<float> thin(0.0F, 0.01F);  // ~1 cm wall thickness noise
  const float step = length / static_cast<float>(n - 1);
  for (int i = 0; i < n; ++i) {
    cloud->push_back(makePoint(i * step, thin(rng), thin(rng)));
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

QualityConfig defaultConfig() { return QualityConfig{}; }  // min_points=100, min_spread=0.1

// --- #1: enough points + good 3D spread -> accepted ---------------------------
TEST(QualityCheckerTest, GoodScanIsAccepted) {
  const CloudPtr cloud = makeBox(8, 5.0F);  // 512 points, ~2 m^2 spread per axis
  ASSERT_GT(cloud->size(), 100U);
  const QualityResult r = QualityChecker{defaultConfig()}.check(cloud);
  EXPECT_TRUE(r.accepted);
  EXPECT_TRUE(r.reason.empty());
}

// --- #2: fewer points than min_points -> rejected, with a meaningful reason ----
TEST(QualityCheckerTest, TooFewPointsIsRejected) {
  const CloudPtr cloud = makeBox(3, 5.0F);  // 27 points < 100, but good spread
  ASSERT_LT(cloud->size(), 100U);
  const QualityResult r = QualityChecker{defaultConfig()}.check(cloud);
  EXPECT_FALSE(r.accepted);
  EXPECT_NE(r.reason.find("too few points"), std::string::npos);
  EXPECT_NE(r.reason.find("27"), std::string::npos);  // actual count surfaced
}

// --- #3: single plane (all z=0) -> zero spread in z -> rejected ----------------
TEST(QualityCheckerTest, FlatPlaneIsRejected) {
  const CloudPtr cloud = makePlane(20, 5.0F);  // 400 points, but z-extent == 0
  ASSERT_GT(cloud->size(), 100U);
  const QualityResult r = QualityChecker{defaultConfig()}.check(cloud);
  EXPECT_FALSE(r.accepted);
  EXPECT_NE(r.reason.find("insufficient spread"), std::string::npos);
}

// --- #4: corridor (spread in one direction only) -> rejected -------------------
TEST(QualityCheckerTest, CorridorIsRejected) {
  const CloudPtr cloud = makeCorridor(300, 20.0F);  // long in x, ~0 in y,z
  ASSERT_GT(cloud->size(), 100U);
  const QualityResult r = QualityChecker{defaultConfig()}.check(cloud);
  EXPECT_FALSE(r.accepted);
  EXPECT_NE(r.reason.find("insufficient spread"), std::string::npos);
}

// --- null / empty input is rejected on the point-count check (no PCA crash) ----
TEST(QualityCheckerTest, NullAndEmptyAreRejected) {
  const QualityResult null_r = QualityChecker{defaultConfig()}.check(nullptr);
  EXPECT_FALSE(null_r.accepted);
  EXPECT_NE(null_r.reason.find("too few points"), std::string::npos);

  const QualityResult empty_r = QualityChecker{defaultConfig()}.check(std::make_shared<Cloud>());
  EXPECT_FALSE(empty_r.accepted);
  EXPECT_NE(empty_r.reason.find("too few points"), std::string::npos);
}

// --- thresholds come from config: a lenient spread floor accepts the plane ------
TEST(QualityCheckerTest, ThresholdsAreConfigurable) {
  QualityConfig cfg;
  cfg.min_points = 10;
  cfg.min_spread_eigenvalue = -1.0;  // effectively disables the spread check
  const QualityResult r = QualityChecker{cfg}.check(makePlane(20, 5.0F));
  EXPECT_TRUE(r.accepted);
}

}  // namespace
