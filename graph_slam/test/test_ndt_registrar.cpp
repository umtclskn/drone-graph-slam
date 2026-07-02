#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <cmath>
#include <random>

#include "graph_slam/initial_guess.hpp"
#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/pcl_ndt_baseline.hpp"
#include "graph_slam/point_types.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudConstPtr;
using graph_slam::CloudPtr;
using graph_slam::NdtConfig;
using graph_slam::NdtGridConfig;
using graph_slam::NdtRegistrar;
using graph_slam::NdtVoxelGrid;
using graph_slam::PclNdtBaseline;
using graph_slam::PclNdtConfig;
using graph_slam::PointT;
using graph_slam::RegistrationResult;
using graph_slam::relativePoseGuess;

/// A closed 5 x 5 x 3 m room: floor + ceiling + four walls. A full enclosure
/// constrains all six DOF strongly, so NDT recovers a controlled transform
/// tightly. Points carry ~1 cm deterministic surface jitter so each cell's
/// covariance is full-rank; the source is later an exact rigid transform of this
/// cloud, so the jitter conditions the problem without adding registration error.
CloudPtr makeRoom() {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(42);  // fixed seed -> deterministic, reproducible fixture
  std::normal_distribution<float> jitter(0.0F, 0.01F);
  const auto add = [&](float x, float y, float z) {
    PointT p;
    p.x = x + jitter(rng);
    p.y = y + jitter(rng);
    p.z = z + jitter(rng);
    p.intensity = 1.0F;
    cloud->push_back(p);
  };
  constexpr float kStep = 0.2F;  // sparse enough to keep the NDT suite fast
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
      add(x, y, 0.0F);  // floor
      add(x, y, 3.0F);  // ceiling
    }
  }
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(x, 0.0F, z);  // wall y = 0
      add(x, 5.0F, z);  // wall y = 5
    }
  }
  for (float y = 0.0F; y <= 5.0F + 1e-4F; y += kStep) {
    for (float z = 0.0F; z <= 3.0F + 1e-4F; z += kStep) {
      add(0.0F, y, z);  // wall x = 0
      add(5.0F, y, z);  // wall x = 5
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

/// Build the NDT target representation (NDT-05/06) the registrar consumes.
NdtVoxelGrid buildGrid(const CloudConstPtr& cloud, double resolution) {
  NdtGridConfig cfg;
  cfg.resolution = resolution;
  NdtVoxelGrid grid{cfg};
  grid.build(cloud);
  return grid;
}

NdtConfig ourConfig() {
  NdtConfig cfg;
  cfg.max_iterations = 50;
  cfg.step_epsilon = 1e-4;
  return cfg;
}

/// yaw + translation transform.
Eigen::Matrix4f makeTransform(float yaw_rad, const Eigen::Vector3f& t) {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  m.block<3, 1>(0, 3) = t;
  return m;
}

CloudPtr transformedCopy(const CloudConstPtr& cloud, const Eigen::Matrix4f& m) {
  auto out = std::make_shared<Cloud>();
  pcl::transformPointCloud(*cloud, *out, m);
  return out;
}

float translationError(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
  return (a.block<3, 1>(0, 3) - b.block<3, 1>(0, 3)).norm();
}

/// Geodesic rotation angle between two transforms' rotation blocks [rad].
float rotationError(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
  const Eigen::Matrix3f r = a.block<3, 3>(0, 0) * b.block<3, 3>(0, 0).transpose();
  const float c = (r.trace() - 1.0F) * 0.5F;
  return std::acos(std::max(-1.0F, std::min(1.0F, c)));
}

// =============== NDT-09: hand-rolled NDT, consuming NdtVoxelGrid ==============

// --- a known translation is recovered (assert the actual transform) ---------
TEST(NdtRegistrarTest, RecoversKnownTranslation) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  // Realistic per-scan motion (~0.1 m): identity is inside the NDT basin here.
  const Eigen::Matrix4f motion = makeTransform(0.0F, {0.08F, -0.05F, 0.04F});
  const auto source = transformedCopy(target, motion);

  const RegistrationResult r = NdtRegistrar{ourConfig()}.align(grid, source);

  ASSERT_TRUE(r.converged);
  const Eigen::Matrix4f expected = motion.inverse();  // align recovers motion^{-1}
  EXPECT_LT(translationError(r.transform, expected), 0.05F);
  EXPECT_LT(rotationError(r.transform, expected), 0.02F);  // ~1.1 deg

  // The final Hessian is retained (NDT-12 will build Sigma_meas from it).
  EXPECT_GT(r.hessian.norm(), 0.0);
  EXPECT_TRUE(r.hessian.isApprox(r.hessian.transpose(), 1e-9));
}

// --- a known yaw rotation is recovered --------------------------------------
TEST(NdtRegistrarTest, RecoversKnownRotation) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  const float yaw = 5.0F * static_cast<float>(M_PI) / 180.0F;  // small in-basin rotation
  const Eigen::Matrix4f motion = makeTransform(yaw, {0.05F, 0.0F, 0.0F});
  const auto source = transformedCopy(target, motion);

  const RegistrationResult r = NdtRegistrar{ourConfig()}.align(grid, source);

  ASSERT_TRUE(r.converged);
  const Eigen::Matrix4f expected = motion.inverse();
  EXPECT_LT(translationError(r.transform, expected), 0.05F);
  EXPECT_LT(rotationError(r.transform, expected), 0.02F);
}

// --- registrar genuinely consumes the NdtVoxelGrid target -------------------
TEST(NdtRegistrarTest, ConsumesNdtVoxelGridTarget) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  ASSERT_GT(grid.numVoxels(), 0U);  // the target is the per-voxel Gaussians, not a cloud

  const Eigen::Matrix4f motion = makeTransform(0.0F, {0.06F, 0.03F, -0.02F});
  const auto source = transformedCopy(target, motion);

  const RegistrationResult r = NdtRegistrar{ourConfig()}.align(grid, source);

  ASSERT_TRUE(r.converged);
  EXPECT_LT(translationError(r.transform, motion.inverse()), 0.05F);
}

// --- null / empty inputs -> non-converged, identity -------------------------
TEST(NdtRegistrarTest, NullOrEmptyInputIsNonConverged) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  const NdtRegistrar reg{ourConfig()};

  const RegistrationResult r_null = reg.align(grid, nullptr);
  EXPECT_FALSE(r_null.converged);
  EXPECT_TRUE(r_null.transform.isApprox(Eigen::Matrix4f::Identity()));

  const NdtVoxelGrid empty_grid = buildGrid(std::make_shared<Cloud>(), 0.5);
  const RegistrationResult r_empty = reg.align(empty_grid, makeRoom());
  EXPECT_FALSE(r_empty.converged);
}

// ====================== NDT-08: initial guess / prior ========================

// --- relativePoseGuess matches a hand-computed example ----------------------
TEST(InitialGuessTest, RelativePoseGuessMatchesHandComputed) {
  // a at (1,0,0) yawed +90 deg; b at (1,1,0) yawed +90 deg.
  const Eigen::Matrix4f world_a =
      makeTransform(static_cast<float>(M_PI) / 2.0F, {1.0F, 0.0F, 0.0F});
  const Eigen::Matrix4f world_b =
      makeTransform(static_cast<float>(M_PI) / 2.0F, {1.0F, 1.0F, 0.0F});

  const Eigen::Matrix4f t_a_b = relativePoseGuess(world_a, world_b);

  // Same orientation -> relative rotation identity; a's +x axis points along
  // world +y, so a (0,1) world offset reads as (+1, 0) in a's frame.
  EXPECT_TRUE((t_a_b.block<3, 3>(0, 0).isApprox(Eigen::Matrix3f::Identity(), 1e-5F)));
  EXPECT_NEAR(t_a_b(0, 3), 1.0F, 1e-5F);
  EXPECT_NEAR(t_a_b(1, 3), 0.0F, 1e-5F);
  EXPECT_NEAR(t_a_b(2, 3), 0.0F, 1e-5F);
  EXPECT_TRUE((world_a * t_a_b).isApprox(world_b, 1e-5F));
}

// --- good prior measurably beats identity, and beats a bad/random guess ------
// Large motion (0.8 m + 20 deg yaw) on a coarse 1.0 m grid: identity stalls in a
// wrong basin; the EKF2-style prior (near-truth) recovers it.
TEST(NdtRegistrarTest, GoodPriorImprovesConvergenceOverIdentityAndBadGuess) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 1.0);
  const float yaw = 20.0F * static_cast<float>(M_PI) / 180.0F;
  const Eigen::Matrix4f motion = makeTransform(yaw, {0.8F, -0.5F, 0.2F});
  const auto source = transformedCopy(target, motion);

  const Eigen::Matrix4f truth = motion.inverse();  // what align should recover
  const NdtRegistrar reg{ourConfig()};

  const Eigen::Matrix4f good_guess =
      truth * makeTransform(3.0F * static_cast<float>(M_PI) / 180.0F, {0.1F, 0.05F, 0.0F});
  const Eigen::Matrix4f bad_guess =
      makeTransform(90.0F * static_cast<float>(M_PI) / 180.0F, {-3.0F, 3.0F, 0.0F});

  const float err_identity = translationError(reg.align(grid, source).transform, truth);
  const RegistrationResult r_good = reg.align(grid, source, good_guess);
  const float err_good = translationError(r_good.transform, truth);
  const float err_bad = translationError(reg.align(grid, source, bad_guess).transform, truth);

  ASSERT_TRUE(r_good.converged);
  EXPECT_LT(err_good, 0.1F);          // good prior recovers the motion
  EXPECT_LT(err_good, err_identity);  // ...better than identity...
  EXPECT_LT(err_good, err_bad);       // ...and far better than a bad/random guess
}

// ============== Cross-check: our NDT ~= the trusted PCL baseline ==============

TEST(NdtRegistrarTest, MatchesPclBaselineOnSameInput) {
  const auto target = makeRoom();
  const Eigen::Matrix4f motion = makeTransform(0.0F, {0.08F, -0.05F, 0.04F});
  const auto source = transformedCopy(target, motion);

  // Seed both near the optimum (truth perturbed ~2 cm / 1 deg). This compares the
  // two NDTs' *optima* — validating our score/gradient/Hessian against the trusted
  // reference — rather than basin robustness (our trust-region NDT actually
  // recovers larger motions than the pcl baseline; see the recovery tests).
  const Eigen::Matrix4f truth = motion.inverse();
  const Eigen::Matrix4f seed =
      truth * makeTransform(1.0F * static_cast<float>(M_PI) / 180.0F, {0.02F, -0.01F, 0.0F});

  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  const RegistrationResult ours = NdtRegistrar{ourConfig()}.align(grid, source, seed);

  PclNdtConfig pcfg;
  pcfg.resolution = 0.5;
  pcfg.transformation_epsilon = 1e-4;
  pcfg.max_iterations = 50;
  const RegistrationResult baseline = PclNdtBaseline{pcfg}.align(source, target, seed);

  ASSERT_TRUE(ours.converged);
  ASSERT_TRUE(baseline.converged);
  // Both land on essentially the same optimum (different internal grids -> a small
  // but bounded discrepancy), validating our implementation against the reference.
  EXPECT_LT(translationError(ours.transform, baseline.transform), 0.03F);
  EXPECT_LT(rotationError(ours.transform, baseline.transform), 0.02F);
}

}  // namespace
