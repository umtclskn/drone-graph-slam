// NDT-12: measurement covariance Sigma_meas from the cost Hessian, with a
// geometry-aware diagonal fallback. ROS-free unit tests, mirroring the fixtures
// of test_ndt_registrar.cpp (closed room = strong geometry; single plane = weak).

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>

#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudConstPtr;
using graph_slam::CloudPtr;
using graph_slam::Matrix6f;
using graph_slam::MeasurementCovariance;
using graph_slam::measurementCovariance;
using graph_slam::NdtConfig;
using graph_slam::NdtGridConfig;
using graph_slam::NdtRegistrar;
using graph_slam::NdtVoxelGrid;
using graph_slam::PointT;
using graph_slam::RegistrationResult;

using Matrix6d = Eigen::Matrix<double, 6, 6>;

/// Closed 5 x 5 x 3 m room (floor + ceiling + four walls): a full enclosure
/// constrains all six DOF, so the NDT cost Hessian is well-conditioned and
/// positive-definite. Same construction as test_ndt_registrar's makeRoom().
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
  constexpr float kStep = 0.2F;
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

/// A single horizontal plane (floor only): constrains z, roll, pitch but leaves
/// x, y and yaw free -> a rank-deficient (degenerate) cost Hessian.
CloudPtr makePlane() {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(7);
  std::normal_distribution<float> jitter(0.0F, 0.01F);
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += 0.1F) {
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += 0.1F) {
      PointT p;
      p.x = x + jitter(rng);
      p.y = y + jitter(rng);
      p.z = jitter(rng);
      p.intensity = 1.0F;
      cloud->push_back(p);
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

NdtVoxelGrid buildGrid(const CloudConstPtr& cloud, double resolution) {
  NdtGridConfig cfg;
  cfg.resolution = resolution;
  NdtVoxelGrid grid{cfg};
  grid.build(cloud);
  return grid;
}

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

/// Smallest eigenvalue of a symmetric 6x6 (positive => positive-definite).
double minEigenvalue(const Matrix6f& m) {
  Eigen::SelfAdjointEigenSolver<Matrix6d> es(m.cast<double>());
  return es.eigenvalues()(0);
}

/// Condition number (max/min eigenvalue) of a symmetric 6x6 covariance.
double conditionNumber(const Matrix6f& m) {
  Eigen::SelfAdjointEigenSolver<Matrix6d> es(m.cast<double>());
  return es.eigenvalues()(5) / es.eigenvalues()(0);
}

NdtConfig ourConfig() {
  NdtConfig cfg;
  cfg.max_iterations = 50;
  cfg.step_epsilon = 1e-4;
  return cfg;
}

// ===== Criterion 1 & 4: perfect alignment -> Hessian path, Sigma PD ==========

TEST(NdtCovarianceTest, PerfectAlignmentYieldsPositiveDefiniteHessianCovariance) {
  const auto target = makeRoom();
  const NdtVoxelGrid grid = buildGrid(target, 0.5);
  const Eigen::Matrix4f motion = makeTransform(0.0F, {0.08F, -0.05F, 0.04F});
  const auto source = transformedCopy(target, motion);

  const RegistrationResult r = NdtRegistrar{ourConfig()}.align(grid, source);

  ASSERT_TRUE(r.converged);
  ASSERT_TRUE(r.sigma_from_hessian);  // strong geometry -> well-conditioned H
  // Symmetric and positive-definite (all eigenvalues > 0).
  EXPECT_TRUE(r.sigma_meas.isApprox(r.sigma_meas.transpose(), 1e-6F));
  EXPECT_GT(minEigenvalue(r.sigma_meas), 0.0);
}

// ===== Criterion 2 & 5: fallback path -> exact expected diagonal =============

TEST(NdtCovarianceTest, DegenerateHessianGivesExpectedFallbackDiagonal) {
  NdtConfig cfg = ourConfig();
  cfg.sigma_rot_fallback = 0.02;
  cfg.sigma_t_fallback = 0.3;

  // Zero Hessian == "no usable Hessian": forces the fallback branch.
  const MeasurementCovariance cov = measurementCovariance(Matrix6d::Zero(), cfg);

  EXPECT_FALSE(cov.from_hessian);
  Matrix6f expected = Matrix6f::Zero();
  const float sr2 = 0.02F * 0.02F;
  const float st2 = 0.3F * 0.3F;
  expected.diagonal() << sr2, sr2, sr2, st2, st2, st2;
  EXPECT_TRUE(cov.sigma.isApprox(expected, 1e-6F));
  // Off-diagonals are exactly zero in the fallback.
  EXPECT_FLOAT_EQ(cov.sigma(0, 3), 0.0F);
}

// ===== Criterion 3: ill-conditioned H (condition > cap) -> fallback ==========

TEST(NdtCovarianceTest, ConditionNumberAboveCapTriggersFallback) {
  NdtConfig cfg = ourConfig();
  cfg.sigma_max_condition = 1e6;

  // Condition number 1e7 > 1e6 cap -> fallback.
  Matrix6d H = Matrix6d::Identity();
  H(5, 5) = 1e-7;
  const MeasurementCovariance cov = measurementCovariance(H, cfg);
  EXPECT_FALSE(cov.from_hessian);

  // Just under the cap -> Hessian path is taken.
  H(5, 5) = 1e-5;  // condition 1e5 < 1e6
  EXPECT_TRUE(measurementCovariance(H, cfg).from_hessian);
}

// --- weak geometry yields a much larger / more anisotropic covariance --------
// The story's headline acceptance criterion: strong vs weak geometry -> small vs
// large/anisotropic Sigma_meas. A single plane only weakly constrains x, y and
// yaw, so its cost Hessian is far more ill-conditioned than the closed room's,
// and Sigma = lambda * H^-1 inherits that anisotropy. Note: the NDT grid's
// per-voxel covariance regularization leaks enough in-plane stiffness that the
// plane stays just inside the well-conditioned regime (cond ~6.6e4 < the 1e6
// cap), so this case exercises the *Hessian path's anisotropy*, not the fallback;
// the fallback threshold itself is covered by ConditionNumberAboveCapTriggersFallback.
TEST(NdtCovarianceTest, WeakGeometryGivesMoreAnisotropicCovariance) {
  const auto room = makeRoom();
  const RegistrationResult r_room = NdtRegistrar{ourConfig()}.align(
      buildGrid(room, 0.5), transformedCopy(room, makeTransform(0.0F, {0.05F, -0.03F, 0.02F})));

  const auto plane = makePlane();
  const RegistrationResult r_plane = NdtRegistrar{ourConfig()}.align(
      buildGrid(plane, 1.0), transformedCopy(plane, makeTransform(0.0F, {0.0F, 0.0F, 0.02F})));

  ASSERT_TRUE(r_room.sigma_from_hessian);
  ASSERT_TRUE(r_plane.sigma_from_hessian);
  // Weak geometry -> strongly anisotropic covariance (some direction far less
  // certain). The margin here is ~400x in practice; assert a conservative 50x.
  EXPECT_GT(conditionNumber(r_plane.sigma_meas), 50.0 * conditionNumber(r_room.sigma_meas));
  // ...and its most-uncertain direction carries more variance than the room's.
  Eigen::SelfAdjointEigenSolver<Matrix6d> es_p(r_plane.sigma_meas.cast<double>());
  Eigen::SelfAdjointEigenSolver<Matrix6d> es_r(r_room.sigma_meas.cast<double>());
  EXPECT_GT(es_p.eigenvalues()(5), es_r.eigenvalues()(5));
}

// --- the [t,r] Hessian is reordered into Pose3 [r,t] order -------------------

TEST(NdtCovarianceTest, BlocksAreReorderedToPose3Order) {
  NdtConfig cfg = ourConfig();
  cfg.hessian_lambda = 1.0;
  // Diagonal H in [tx,ty,tz, rx,ry,rz] order: translation stiffness 4, rotation 25.
  Matrix6d H = Matrix6d::Zero();
  H.diagonal() << 4.0, 4.0, 4.0, 25.0, 25.0, 25.0;

  const MeasurementCovariance cov = measurementCovariance(H, cfg);

  ASSERT_TRUE(cov.from_hessian);
  // After reorder, the first 3 (rotation) diagonal entries = 1/25, last 3 = 1/4.
  EXPECT_NEAR(cov.sigma(0, 0), 1.0F / 25.0F, 1e-6F);  // rotation block
  EXPECT_NEAR(cov.sigma(3, 3), 1.0F / 4.0F, 1e-6F);   // translation block
}

// --- lambda scales the covariance linearly ----------------------------------

TEST(NdtCovarianceTest, LambdaScalesCovariance) {
  Matrix6d H = Matrix6d::Identity() * 10.0;
  NdtConfig a = ourConfig();
  NdtConfig b = ourConfig();
  a.hessian_lambda = 1.0;
  b.hessian_lambda = 3.0;

  const Matrix6f sa = measurementCovariance(H, a).sigma;
  const Matrix6f sb = measurementCovariance(H, b).sigma;
  EXPECT_TRUE(sb.isApprox(3.0F * sa, 1e-6F));
}

}  // namespace
