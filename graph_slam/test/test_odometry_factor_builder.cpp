#include <gtest/gtest.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>

#include "graph_slam/graph/odometry_factor_builder.hpp"

namespace {

using graph_slam::graph::OdometryFactorBuilder;

constexpr double kTol = 1e-9;

const gtsam::noiseModel::Gaussian* asGaussian(const gtsam::SharedNoiseModel& noise) {
  return dynamic_cast<const gtsam::noiseModel::Gaussian*>(noise.get());
}

TEST(OdometryFactorBuilderTest, FactorCreation) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta(gtsam::Rot3::RzRyRx(0.1, 0.0, 0.0), gtsam::Point3(1.0, 0.0, 0.0));
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_EQ(factor.keys().size(), 2U);
  EXPECT_NE(dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(&factor), nullptr);
}

TEST(OdometryFactorBuilderTest, CorrectKeys) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta;
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_EQ(factor.key1(), x0);
  EXPECT_EQ(factor.key2(), x1);
}

TEST(OdometryFactorBuilderTest, CorrectRelativePose) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta(gtsam::Rot3::RzRyRx(0.2, -0.1, 0.05), gtsam::Point3(1.5, -0.5, 2.0));
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_TRUE(factor.measured().equals(delta, kTol));
}

TEST(OdometryFactorBuilderTest, CorrectCovariance) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta;
  gtsam::Matrix66 sigma = gtsam::Matrix66::Zero();
  sigma.diagonal() << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);
  const auto* gaussian = asGaussian(factor.noiseModel());
  ASSERT_NE(gaussian, nullptr);
  EXPECT_TRUE(gaussian->covariance().isApprox(sigma, kTol));
}

TEST(OdometryFactorBuilderTest, TranslationOnlyMotion) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta(gtsam::Rot3(), gtsam::Point3(3.0, -2.0, 1.0));
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.02;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_TRUE(factor.measured().equals(delta, kTol));
}

TEST(OdometryFactorBuilderTest, RotationOnlyMotion) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta(gtsam::Rot3::RzRyRx(0.3, -0.2, 0.1), gtsam::Point3(0.0, 0.0, 0.0));
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.02;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_TRUE(factor.measured().equals(delta, kTol));
}

TEST(OdometryFactorBuilderTest, FullSe3Motion) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta(gtsam::Rot3::RzRyRx(0.5, -0.3, 0.2), gtsam::Point3(4.0, 1.0, -2.0));
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.03;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);

  EXPECT_TRUE(factor.measured().equals(delta, kTol));
}

TEST(OdometryFactorBuilderTest, CorrelationPreservation) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Pose3 delta;
  gtsam::Matrix66 sigma = gtsam::Matrix66::Zero();
  sigma(0, 1) = 0.05;
  sigma(1, 0) = 0.05;
  sigma(2, 5) = -0.03;
  sigma(5, 2) = -0.03;
  sigma(3, 4) = 0.02;
  sigma(4, 3) = 0.02;
  sigma.diagonal() << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

  const auto factor = builder.create_factor(x0, x1, delta, sigma);
  const auto* gaussian = asGaussian(factor.noiseModel());
  ASSERT_NE(gaussian, nullptr);
  EXPECT_TRUE(gaussian->covariance().isApprox(sigma, kTol));
  EXPECT_NE(gaussian->covariance()(0, 1), 0.0);
  EXPECT_NE(gaussian->covariance()(2, 5), 0.0);
}

TEST(OdometryFactorBuilderTest, StatelessAcrossInvocations) {
  const OdometryFactorBuilder builder;
  const gtsam::Key x0 = gtsam::Symbol('x', 0);
  const gtsam::Key x1 = gtsam::Symbol('x', 1);
  const gtsam::Key x2 = gtsam::Symbol('x', 2);
  const gtsam::Matrix66 sigma = gtsam::Matrix66::Identity() * 0.01;

  const auto first = builder.create_factor(
      x0, x1, gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(1.0, 0.0, 0.0)), sigma);
  const auto second = builder.create_factor(
      x1, x2, gtsam::Pose3(gtsam::Rot3::RzRyRx(0.1, 0.0, 0.0), gtsam::Point3(0.0, 2.0, 0.0)), sigma);

  EXPECT_EQ(first.key1(), x0);
  EXPECT_EQ(first.key2(), x1);
  EXPECT_EQ(second.key1(), x1);
  EXPECT_EQ(second.key2(), x2);
  EXPECT_TRUE(first.measured().translation().isApprox(gtsam::Point3(1.0, 0.0, 0.0), kTol));
  EXPECT_TRUE(second.measured().translation().isApprox(gtsam::Point3(0.0, 2.0, 0.0), kTol));
}

}  // namespace
