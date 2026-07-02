// SLAM-10: loop-closure verification (NDT). ROS-free unit tests separating
// correct from false candidates (ARCHITECTURE §8): accept a strong, well-
// conditioned registration; reject a poor score or degenerate geometry so a
// false closure never corrupts the graph.
//
// Fixtures mirror test_ndt_covariance.cpp: a closed room (all 6 DOF constrained,
// well-conditioned Hessian) vs a single plane (x, y, yaw free -> ill-conditioned).

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include <random>

#include <Eigen/Geometry>

#include "graph_slam/loop/loop_closure_verifier.hpp"
#include "graph_slam/point_types.hpp"

namespace {

using graph_slam::Cloud;
using graph_slam::CloudConstPtr;
using graph_slam::CloudPtr;
using graph_slam::LoopClosureVerifier;
using graph_slam::PointT;
using graph_slam::VerificationResult;

/// Closed 5 x 5 x 3 m room: floor + ceiling + four walls constrain all six DOF.
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
/// x, y and yaw free -> ill-conditioned cost Hessian.
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

CloudPtr transformedCopy(const CloudConstPtr& cloud, const Eigen::Isometry3d& m) {
  auto out = std::make_shared<Cloud>();
  pcl::transformPointCloud(*cloud, *out, m.matrix().cast<float>());
  return out;
}

LoopClosureVerifier::Params roomParams() {
  LoopClosureVerifier::Params p;  // defaults; grid resolution 1.0 like the pipeline
  return p;
}

LoopClosureVerifier::Params planeParams() {
  LoopClosureVerifier::Params p;
  p.grid_config.resolution = 1.0;
  return p;
}

// ===== Case A — reject on poor NDT score (clouds do not overlap) =============
TEST(LoopClosureVerifierTest, RejectsPoorScore) {
  const CloudPtr room = makeRoom();
  // Query is the room shifted 10 m away with an identity guess: no overlap, so
  // NDT cannot find a strong match.
  Eigen::Isometry3d far = Eigen::Isometry3d::Identity();
  far.translation() = Eigen::Vector3d(10.0, 10.0, 0.0);
  const CloudPtr query = transformedCopy(room, far);

  const LoopClosureVerifier verifier(roomParams());
  const VerificationResult v =
      verifier.verify(query, room, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(v.accepted);
  EXPECT_EQ(v.noise_model, nullptr);
}

// ===== Case B — reject on degenerate / ill-conditioned geometry ==============
TEST(LoopClosureVerifierTest, RejectsDegenerateGeometry) {
  const CloudPtr plane = makePlane();
  // Perfectly overlapping planes: NDT converges with a strong score, but x, y and
  // yaw are unconstrained -> Sigma_meas ill-conditioned -> must be rejected.
  const LoopClosureVerifier verifier(planeParams());
  const VerificationResult v =
      verifier.verify(plane, plane, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(v.accepted);
  EXPECT_EQ(v.noise_model, nullptr);
}

// ===== Case C — accept on strong, well-conditioned registration ==============
TEST(LoopClosureVerifierTest, AcceptsStrongMatch) {
  const CloudPtr room = makeRoom();
  const LoopClosureVerifier verifier(roomParams());
  // Same cloud for query and match, identity guess: perfect overlap.
  const VerificationResult v =
      verifier.verify(room, room, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(v.accepted) << "score=" << v.ndt_score
                          << " min_eig=" << v.sigma_min_eigenvalue;
  EXPECT_LE(v.ndt_score, roomParams().min_ndt_score);
  EXPECT_GT(v.sigma_min_eigenvalue, roomParams().min_sigma_eigenvalue);
  EXPECT_NE(v.noise_model, nullptr);
  // Perfect overlap -> recovered relative pose ~ identity.
  EXPECT_LT(v.relative_pose.translation().norm(), 0.05);
}

}  // namespace
