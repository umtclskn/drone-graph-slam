#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>

#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudPtr;
using graph_slam::NdtGridConfig;
using graph_slam::NdtVoxelGrid;
using graph_slam::PointT;
using graph_slam::VoxelCell;
using graph_slam::VoxelKey;

/// Build a cloud from a list of (x, y, z) triples.
CloudPtr makeCloud(const std::vector<Eigen::Vector3d>& pts) {
  auto cloud = std::make_shared<Cloud>();
  for (const auto& p : pts) {
    PointT q;
    q.x = static_cast<float>(p.x());
    q.y = static_cast<float>(p.y());
    q.z = static_cast<float>(p.z());
    q.intensity = 1.0F;
    cloud->push_back(q);
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

// ============================ NDT-05: partition ============================

// --- occupancy indices: floor(p / resolution), incl. negative coords -------
TEST(NdtVoxelGridTest, VoxelIndexFloorsIncludingNegatives) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  const NdtVoxelGrid grid{cfg};

  EXPECT_EQ(grid.voxelIndex({0.3, 0.9, 0.0}), (VoxelKey{0, 0, 0}));
  EXPECT_EQ(grid.voxelIndex({2.3, 1.0, 3.99}), (VoxelKey{2, 1, 3}));
  EXPECT_EQ(grid.voxelIndex({-0.5, -1.0, -0.01}), (VoxelKey{-1, -1, -1}));
}

// --- resolution scales the partition ---------------------------------------
TEST(NdtVoxelGridTest, ResolutionScalesIndex) {
  NdtGridConfig cfg;
  cfg.resolution = 2.0;
  const NdtVoxelGrid grid{cfg};
  EXPECT_EQ(grid.voxelIndex({3.5, 0.5, 4.5}), (VoxelKey{1, 0, 2}));
}

// --- occupancy: filled cells land at the right keys ------------------------
TEST(NdtVoxelGridTest, OccupancyCountsFilledCellsAtCorrectKeys) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 2;

  // Two clusters in two distinct voxels: (0,0,0) and (5,0,0).
  const auto cloud = makeCloud(
      {{0.1, 0.1, 0.1}, {0.2, 0.3, 0.2}, {0.4, 0.1, 0.3}, {5.1, 0.1, 0.1}, {5.2, 0.2, 0.2}});
  NdtVoxelGrid grid{cfg};
  grid.build(cloud);

  EXPECT_EQ(grid.numVoxels(), 2U);
  ASSERT_NE(grid.cell({0, 0, 0}), nullptr);
  ASSERT_NE(grid.cell({5, 0, 0}), nullptr);
  EXPECT_EQ(grid.cell({0, 0, 0})->count, 3U);
  EXPECT_EQ(grid.cell({5, 0, 0})->count, 2U);
}

// --- empty region -> no cell -----------------------------------------------
TEST(NdtVoxelGridTest, EmptyRegionHasNoCell) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 1;
  NdtVoxelGrid grid{cfg};
  grid.build(makeCloud({{0.1, 0.1, 0.1}}));

  EXPECT_EQ(grid.cell({9, 9, 9}), nullptr);  // never occupied
  EXPECT_EQ(grid.cell({-3, 0, 0}), nullptr);
}

// --- single-point voxel -> skipped by min_points ---------------------------
TEST(NdtVoxelGridTest, SinglePointVoxelSkippedByMinPoints) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 5;

  std::vector<Eigen::Vector3d> pts;
  // Voxel (0,0,0): 5 points -> kept.
  for (int n = 0; n < 5; ++n) {
    pts.push_back({0.1 + 0.05 * n, 0.2, 0.3});
  }
  // Voxel (7,0,0): a lone point -> dropped.
  pts.push_back({7.5, 0.1, 0.1});

  NdtVoxelGrid grid{cfg};
  grid.build(makeCloud(pts));

  EXPECT_EQ(grid.numVoxels(), 1U);
  EXPECT_NE(grid.cell({0, 0, 0}), nullptr);
  EXPECT_EQ(grid.cell({7, 0, 0}), nullptr);  // single point, below min_points
}

// --- dense voxel -> exactly one cell ---------------------------------------
TEST(NdtVoxelGridTest, DensePointsCollapseToSingleVoxel) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 5;

  std::vector<Eigen::Vector3d> pts;
  for (int n = 0; n < 50; ++n) {
    const double t = 0.01 * n;  // stays inside [0,1) -> voxel (0,0,0)
    pts.push_back({t, 0.5 * t, 0.9 - t});
  }
  NdtVoxelGrid grid{cfg};
  grid.build(makeCloud(pts));

  EXPECT_EQ(grid.numVoxels(), 1U);
  ASSERT_NE(grid.cell({0, 0, 0}), nullptr);
  EXPECT_EQ(grid.cell({0, 0, 0})->count, 50U);
}

// ========================= NDT-06: voxel statistics =========================

// --- mean + covariance match a hand-computed 4-point example ----------------
// Points (all in voxel (0,0,0)):
//   A=(0,0,0) B=(0.4,0,0) C=(0,0.4,0) D=(0,0,0.4),  N=4
//   mean = (0.1, 0.1, 0.1)
//   sample cov (1/(N-1)) = (1/3) * [[0.12,-0.04,-0.04],
//                                   [-0.04,0.12,-0.04],
//                                   [-0.04,-0.04,0.12]]
//             = diag 0.0400000, off-diagonal -0.0133333...
// Eigenvalues are {0.0133.., 0.0533.., 0.0533..}; the smallest is far above
// reg_ratio*lambda_max, so the singular guard is a no-op and Sigma is exact.
TEST(NdtVoxelGridTest, MeanAndCovarianceMatchHandComputed) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 4;
  cfg.reg_ratio = 1e-3;

  NdtVoxelGrid grid{cfg};
  grid.build(makeCloud({{0.0, 0.0, 0.0}, {0.4, 0.0, 0.0}, {0.0, 0.4, 0.0}, {0.0, 0.0, 0.4}}));

  ASSERT_EQ(grid.numVoxels(), 1U);
  const VoxelCell* c = grid.cell({0, 0, 0});
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->count, 4U);

  // Tolerance 1e-6: points are stored as float in the cloud (e.g. 0.4f), so the
  // exact double targets are only reproduced to single-precision.
  EXPECT_NEAR(c->mean.x(), 0.1, 1e-6);
  EXPECT_NEAR(c->mean.y(), 0.1, 1e-6);
  EXPECT_NEAR(c->mean.z(), 0.1, 1e-6);

  const double diag = 0.04;
  const double off = -0.04 / 3.0;  // -0.0133333...
  Eigen::Matrix3d expected;
  expected << diag, off, off, off, diag, off, off, off, diag;
  EXPECT_TRUE(c->cov.isApprox(expected, 1e-6)) << "cov was\n"
                                               << c->cov << "\nexpected\n"
                                               << expected;

  // Sigma * Sigma^-1 == I.
  EXPECT_TRUE((c->cov * c->cov_inv).isApprox(Eigen::Matrix3d::Identity(), 1e-9));
}

// --- singular guard: coplanar points stay invertible, smallest eigenvalue
//     sits on the reg_ratio * lambda_max floor, Sigma^-1 has no NaN/inf -------
TEST(NdtVoxelGridTest, SingularGuardOnCoplanarPoints) {
  NdtGridConfig cfg;
  cfg.resolution = 1.0;
  cfg.min_points = 5;
  cfg.reg_ratio = 1e-3;

  // Five points on the plane z = 0.5 (zero variance along z) within voxel (0,0,0).
  NdtVoxelGrid grid{cfg};
  grid.build(makeCloud(
      {{0.1, 0.1, 0.5}, {0.9, 0.1, 0.5}, {0.1, 0.9, 0.5}, {0.9, 0.9, 0.5}, {0.5, 0.5, 0.5}}));

  ASSERT_EQ(grid.numVoxels(), 1U);
  const VoxelCell* c = grid.cell({0, 0, 0});
  ASSERT_NE(c, nullptr);

  // No NaN / inf anywhere in the inverse.
  EXPECT_TRUE(c->cov_inv.allFinite());

  // Eigenvalues of the guarded covariance: the degenerate (z) direction was
  // lifted to exactly reg_ratio * lambda_max.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(c->cov);
  const Eigen::Vector3d evals = solver.eigenvalues();  // ascending
  const double lambda_max = evals(2);
  EXPECT_NEAR(evals(0), cfg.reg_ratio * lambda_max, 1e-12);

  // And the guard genuinely lifted it from (near) zero.
  EXPECT_GT(evals(0), 0.0);
}

}  // namespace
