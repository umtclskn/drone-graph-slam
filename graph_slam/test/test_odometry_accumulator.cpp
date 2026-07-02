#include <gtest/gtest.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Eigenvalues>

#include "graph_slam/graph/odometry_accumulator.hpp"

namespace {

using graph_slam::graph::OdometryAccumulator;

constexpr double kTol = 1e-9;

gtsam::Matrix66 diagSigma(double r, double t) {
  gtsam::Matrix66 s = gtsam::Matrix66::Zero();
  s.diagonal() << r, r, r, t, t, t;
  return s;
}

TEST(OdometryAccumulatorTest, InitialStateIsIdentityAndZero) {
  const OdometryAccumulator acc;
  const auto cur = acc.current();
  EXPECT_TRUE(cur.delta.equals(gtsam::Pose3(), kTol));
  EXPECT_TRUE(cur.covariance.isZero(kTol));
}

TEST(OdometryAccumulatorTest, SingleMeasurementReproducesInput) {
  OdometryAccumulator acc;
  const gtsam::Pose3 dt(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.05), gtsam::Point3(1.0, -0.5, 0.3));
  const gtsam::Matrix66 sigma = diagSigma(0.01, 0.04);

  acc.add_measurement(dt, sigma);
  const auto cur = acc.current();

  EXPECT_TRUE(cur.delta.equals(dt, kTol));
  // From identity (J_a Σ_acc J_aᵀ term vanishes, J_b = I): Σ_acc == Σ_step.
  EXPECT_TRUE(cur.covariance.isApprox(sigma, 1e-9));
}

TEST(OdometryAccumulatorTest, TwoStepComposition) {
  OdometryAccumulator acc;
  const gtsam::Pose3 dt1(gtsam::Rot3::RzRyRx(0.1, 0.0, 0.0), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Pose3 dt2(gtsam::Rot3::RzRyRx(0.0, 0.2, 0.0), gtsam::Point3(0.0, 2.0, 0.0));
  const gtsam::Matrix66 sigma = diagSigma(0.01, 0.01);

  acc.add_measurement(dt1, sigma);
  acc.add_measurement(dt2, sigma);

  EXPECT_TRUE(acc.current().delta.equals(dt1.compose(dt2), kTol));
}

TEST(OdometryAccumulatorTest, CovarianceGrowsWithMoreMeasurements) {
  OdometryAccumulator acc;
  const gtsam::Pose3 dt(gtsam::Rot3::RzRyRx(0.05, 0.05, 0.05), gtsam::Point3(0.5, 0.5, 0.5));
  const gtsam::Matrix66 sigma = diagSigma(0.02, 0.03);

  acc.add_measurement(dt, sigma);
  const double trace_after_1 = acc.current().covariance.trace();
  acc.add_measurement(dt, sigma);
  const double trace_after_2 = acc.current().covariance.trace();

  EXPECT_GT(trace_after_2, trace_after_1);
}

TEST(OdometryAccumulatorTest, ResetRestoresIdentityAndZero) {
  OdometryAccumulator acc;
  acc.add_measurement(gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.1, 0.2), gtsam::Point3(2.0, 1.0, 3.0)),
                      diagSigma(0.05, 0.05));
  acc.reset();

  const auto cur = acc.current();
  EXPECT_TRUE(cur.delta.equals(gtsam::Pose3(), kTol));
  EXPECT_TRUE(cur.covariance.isZero(kTol));
}

TEST(OdometryAccumulatorTest, PureTranslationComposes) {
  OdometryAccumulator acc;
  const gtsam::Matrix66 sigma = diagSigma(0.0, 0.01);
  acc.add_measurement(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), sigma);
  acc.add_measurement(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.0, 2.0, 0.0)), sigma);

  const gtsam::Pose3 expected(gtsam::Rot3(), gtsam::Point3(1.0, 2.0, 0.0));
  EXPECT_TRUE(acc.current().delta.equals(expected, kTol));
}

TEST(OdometryAccumulatorTest, PureRotationComposes) {
  OdometryAccumulator acc;
  const gtsam::Matrix66 sigma = diagSigma(0.01, 0.0);
  const gtsam::Pose3 r1(gtsam::Rot3::Rz(0.3), gtsam::Point3(0.0, 0.0, 0.0));
  const gtsam::Pose3 r2(gtsam::Rot3::Rz(0.4), gtsam::Point3(0.0, 0.0, 0.0));
  acc.add_measurement(r1, sigma);
  acc.add_measurement(r2, sigma);

  const gtsam::Pose3 expected(gtsam::Rot3::Rz(0.7), gtsam::Point3(0.0, 0.0, 0.0));
  EXPECT_TRUE(acc.current().delta.equals(expected, kTol));
}

TEST(OdometryAccumulatorTest, MixedSE3Composes) {
  OdometryAccumulator acc;
  const gtsam::Pose3 dt1(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3), gtsam::Point3(1.0, 2.0, 3.0));
  const gtsam::Pose3 dt2(gtsam::Rot3::RzRyRx(-0.2, 0.05, 0.1), gtsam::Point3(-0.5, 1.0, 0.2));
  const gtsam::Matrix66 sigma = diagSigma(0.01, 0.02);

  acc.add_measurement(dt1, sigma);
  acc.add_measurement(dt2, sigma);

  EXPECT_TRUE(acc.current().delta.equals(dt1.compose(dt2), kTol));
}

TEST(OdometryAccumulatorTest, PropagatedCovarianceStaysSymmetricPsd) {
  OdometryAccumulator acc;
  const gtsam::Pose3 dt(gtsam::Rot3::RzRyRx(0.07, -0.04, 0.11), gtsam::Point3(0.6, -0.3, 0.8));
  const gtsam::Matrix66 sigma = diagSigma(0.015, 0.025);

  for (int i = 0; i < 10; ++i) {
    acc.add_measurement(dt, sigma);
  }
  const gtsam::Matrix66 cov = acc.current().covariance;

  EXPECT_TRUE(cov.isApprox(cov.transpose(), 1e-9));

  Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> solver(cov);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-9);
}

}  // namespace
