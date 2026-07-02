// EVAL-02 (Sprint 1): per-pair relative transform error (ROS-free).
// Covers the acceptance criteria: perfect alignment -> ~0 error; a known pure
// translation -> that translation magnitude; a known pure rotation -> that
// angle. Pure Eigen, no ROS, no PCL.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>

#include "graph_slam/eval/relative_error.hpp"

namespace {

using graph_slam::compute;
using graph_slam::RelativeErrorResult;

Eigen::Isometry3f translation(float x, float y, float z) {
  Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
  t.translation() = Eigen::Vector3f(x, y, z);
  return t;
}

// Perfect alignment: T_gt == T_ndt -> both errors vanish.
TEST(RelativeErrorTest, PerfectAlignmentIsZero) {
  Eigen::Isometry3f T = translation(1.2F, -0.4F, 0.3F);
  T.linear() = Eigen::AngleAxisf(0.35F, Eigen::Vector3f::UnitZ()).toRotationMatrix();

  const RelativeErrorResult r = compute(T, T);
  EXPECT_LT(r.t_err_m, 1e-5F);
  EXPECT_LT(r.r_err_deg, 1e-4F);
}

// T_gt = identity, T_ndt = 0.1 m translation -> t_err == 0.1 m, no rotation.
TEST(RelativeErrorTest, PureTranslationRecoversMagnitude) {
  const RelativeErrorResult r =
      compute(translation(0.1F, 0.0F, 0.0F), Eigen::Isometry3f::Identity());
  EXPECT_NEAR(r.t_err_m, 0.1F, 1e-6F);
  EXPECT_LT(r.r_err_deg, 1e-4F);
}

// A known 10 deg yaw error, no translation -> r_err == 10 deg.
TEST(RelativeErrorTest, PureRotationRecoversAngle) {
  Eigen::Isometry3f T_ndt = Eigen::Isometry3f::Identity();
  const float deg10 = 10.0F * static_cast<float>(M_PI) / 180.0F;
  T_ndt.linear() = Eigen::AngleAxisf(deg10, Eigen::Vector3f::UnitZ()).toRotationMatrix();

  const RelativeErrorResult r = compute(T_ndt, Eigen::Isometry3f::Identity());
  EXPECT_LT(r.t_err_m, 1e-6F);
  EXPECT_NEAR(r.r_err_deg, 10.0F, 1e-3F);
}

}  // namespace
