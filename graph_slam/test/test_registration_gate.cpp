#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <random>

#include "graph_slam/ndt_registrar.hpp"
#include "graph_slam/ndt_voxel_grid.hpp"
#include "graph_slam/point_types.hpp"
#include "graph_slam/registration_gate.hpp"
#include "graph_slam/registration_result.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudPtr;
using graph_slam::evaluateRegistration;
using graph_slam::NdtConfig;
using graph_slam::NdtGridConfig;
using graph_slam::NdtRegistrar;
using graph_slam::NdtVoxelGrid;
using graph_slam::PointT;
using graph_slam::RegistrationGateConfig;
using graph_slam::RegistrationResult;
using graph_slam::RegistrationStatus;

/// Hand-build a RegistrationResult with a diagonal Hessian of given eigenvalues.
RegistrationResult makeResult(bool converged, double fitness,
                              const Eigen::Matrix<double, 6, 1>& eig) {
  RegistrationResult r;
  r.converged = converged;
  r.fitness_score = fitness;
  r.hessian = eig.asDiagonal();
  return r;
}

/// A well-conditioned positive-definite Hessian (cond = 1, min eig = 100).
Eigen::Matrix<double, 6, 1> wellConditioned() {
  return Eigen::Matrix<double, 6, 1>::Constant(100.0);
}

// Default thresholds used across cases unless a test overrides them.
RegistrationGateConfig defaultConfig() { return RegistrationGateConfig{}; }

// --- good: converged + good fitness + well-conditioned H -> Reliable ---------
TEST(RegistrationGateTest, GoodMeasurementIsReliable) {
  const RegistrationResult r = makeResult(true, -1.5, wellConditioned());
  EXPECT_EQ(evaluateRegistration(r, defaultConfig()), RegistrationStatus::Reliable);
}

// --- bad: not converged -> NotConverged (regardless of the other fields) -----
TEST(RegistrationGateTest, NotConvergedIsRejected) {
  // Even with otherwise-good fitness / Hessian, non-convergence wins.
  const RegistrationResult r = makeResult(false, -1.5, wellConditioned());
  EXPECT_EQ(evaluateRegistration(r, defaultConfig()), RegistrationStatus::NotConverged);
}

// --- bad: converged but poor (too-high) fitness -> PoorFit -------------------
TEST(RegistrationGateTest, PoorFitnessIsRejected) {
  const RegistrationResult r = makeResult(true, -0.1, wellConditioned());  // > -0.5
  EXPECT_EQ(evaluateRegistration(r, defaultConfig()), RegistrationStatus::PoorFit);
}

// --- the fitness convention is honoured (lower = better) --------------------
TEST(RegistrationGateTest, FitnessConventionLowerIsBetter) {
  RegistrationGateConfig cfg;
  cfg.max_fitness_score = -0.5;
  // Clearly-good (very negative) passes; clearly-bad (~0) is rejected.
  EXPECT_EQ(evaluateRegistration(makeResult(true, -3.0, wellConditioned()), cfg),
            RegistrationStatus::Reliable);
  EXPECT_EQ(evaluateRegistration(makeResult(true, 0.0, wellConditioned()), cfg),
            RegistrationStatus::PoorFit);
}

// --- symmetric: good fit but a near-zero Hessian eigenvalue -> Degenerate ----
TEST(RegistrationGateTest, DegenerateHessianIsRejectedDespiteGoodFit) {
  Eigen::Matrix<double, 6, 1> eig = wellConditioned();
  eig(5) = 1e-9;                                             // one (near-)unconstrained direction
  const RegistrationResult r = makeResult(true, -1.5, eig);  // fitness is good
  EXPECT_EQ(evaluateRegistration(r, defaultConfig()), RegistrationStatus::Degenerate);
}

// --- degeneracy via condition number (min eig above floor, but huge spread) --
TEST(RegistrationGateTest, HighConditionNumberIsDegenerate) {
  RegistrationGateConfig cfg;
  cfg.min_hessian_eigenvalue = 1e-3;  // min eig 1.0 passes this...
  cfg.max_condition_number = 1e4;     // ...but the spread trips this
  Eigen::Matrix<double, 6, 1> eig;
  eig << 1e6, 1e6, 1e6, 1e6, 1e6, 1.0;  // cond = 1e6 > 1e4
  const RegistrationResult r = makeResult(true, -1.5, eig);
  EXPECT_EQ(evaluateRegistration(r, cfg), RegistrationStatus::Degenerate);
}

// --- empty input: default result (converged=false, zero Hessian) -> rejected -
TEST(RegistrationGateTest, EmptyInputResultIsRejected) {
  const RegistrationResult r;  // NdtRegistrar returns this for empty/degenerate input
  EXPECT_EQ(evaluateRegistration(r, defaultConfig()), RegistrationStatus::NotConverged);
}

// ---- end-to-end: gate a real NDT-09 result on a degenerate vs full scene ----

CloudPtr makePlane(bool walls) {
  auto cloud = std::make_shared<Cloud>();
  std::mt19937 rng(7);
  std::normal_distribution<float> jitter(0.0F, 0.01F);
  const auto add = [&](float x, float y, float z) {
    PointT p;
    p.x = x + jitter(rng);
    p.y = y + jitter(rng);
    p.z = z + jitter(rng);
    p.intensity = 1.0F;
    cloud->push_back(p);
  };
  for (float x = 0.0F; x <= 5.0F + 1e-4F; x += 0.2F) {
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += 0.2F) {
      add(x, y, 0.0F);  // floor (constrains z, roll, pitch; x,y,yaw are free)
    }
  }
  if (walls) {  // add two perpendicular walls -> fully constrained
    for (float x = 0.0F; x <= 5.0F + 1e-4F; x += 0.2F) {
      for (float z = 0.0F; z <= 3.0F + 1e-4F; z += 0.2F) add(x, 0.0F, z);
    }
    for (float y = 0.0F; y <= 5.0F + 1e-4F; y += 0.2F) {
      for (float z = 0.0F; z <= 3.0F + 1e-4F; z += 0.2F) add(0.0F, y, z);
    }
  }
  cloud->width = cloud->size();
  cloud->height = 1;
  return cloud;
}

RegistrationResult registerSmallMotion(const CloudPtr& target) {
  NdtGridConfig gc;
  gc.resolution = 0.5;
  NdtVoxelGrid grid{gc};
  grid.build(target);
  Eigen::Matrix4f motion = Eigen::Matrix4f::Identity();
  motion(2, 3) = 0.05F;  // small across-plane (z) shift, observable for both scenes
  auto source = std::make_shared<Cloud>();
  pcl::transformPointCloud(*target, *source, motion);
  NdtConfig cfg;
  cfg.max_iterations = 50;
  return NdtRegistrar{cfg}.align(grid, source);
}

// Both scenes converge with a similarly good fit; only the Hessian condition
// number separates them. Measured (resolution 0.5): floor-only cond ~3.3e4 (one
// soft in-plane direction), full room cond ~1.4e2. The fitness threshold is
// calibrated to res 0.5, where a good fit scores ~-0.33 (resolution-dependent).
TEST(RegistrationGateTest, EndToEndFlagsDegenerateSceneButPassesFullScene) {
  RegistrationGateConfig cfg;
  cfg.max_fitness_score = -0.2;
  cfg.min_hessian_eigenvalue = 1.0;
  cfg.max_condition_number = 1e3;  // between the two scenes' condition numbers

  const RegistrationResult floor_only = registerSmallMotion(makePlane(false));
  const RegistrationResult full = registerSmallMotion(makePlane(true));

  // A flat floor leaves x, y and yaw weakly constrained -> degenerate.
  EXPECT_EQ(evaluateRegistration(floor_only, cfg), RegistrationStatus::Degenerate);
  // Floor + two walls constrain all six DOF -> reliable.
  EXPECT_EQ(evaluateRegistration(full, cfg), RegistrationStatus::Reliable);
}

}  // namespace
