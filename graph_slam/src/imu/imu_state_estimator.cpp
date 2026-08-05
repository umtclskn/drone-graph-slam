// DUAL-GRAPH: ImuStateEstimator implementation (see imu_state_estimator.hpp).
//
// Line-by-line port of LIO-SAM's `IMUPreintegration::odometryHandler`
// (`/home/umut/lio_sam_ws/src/LIO-SAM/src/imuPreintegration.cpp:286-490`) onto
// our ImuPreintegrator, with our L5-04 bootstrap sigmas instead of LIO-SAM's.
// Deliberate deviations from the reference are marked "DEVIATION".
#include "graph_slam/imu/imu_state_estimator.hpp"

#include <cmath>
#include <exception>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace graph_slam::imu {
namespace {

namespace sym = gtsam::symbol_shorthand;  // X pose, V velocity, B bias

// LIO-SAM's side-graph solver settings (imuPreintegration.cpp:267-269). Tighter
// than the mapping graph's defaults because this graph is tiny and must track
// the newest state exactly.
gtsam::ISAM2 makeIsam2() {
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.1;
  params.relinearizeSkip = 1;
  return gtsam::ISAM2(params);
}

}  // namespace

ImuStateEstimator::ImuStateEstimator(const ImuStateEstimatorConfig& config)
    : config_(config), preint_(config.preint), isam2_(makeIsam2()) {
  prior_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
      gtsam::Vector6::Constant(config_.prior_pose_sigma));
  prior_velocity_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, config_.prior_velocity_sigma);
  gtsam::Vector6 bias_sigmas;  // ConstantBias tangent order: accel, then gyro
  bias_sigmas << config_.prior_accel_bias_sigma, config_.prior_accel_bias_sigma,
      config_.prior_accel_bias_sigma, config_.prior_gyro_bias_sigma,
      config_.prior_gyro_bias_sigma, config_.prior_gyro_bias_sigma;
  prior_bias_noise_ = gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);

  gtsam::Vector6 correction_sigmas;  // Pose3 tangent order: rotation, then position
  correction_sigmas << config_.correction_rot_sigma, config_.correction_rot_sigma,
      config_.correction_rot_sigma, config_.correction_pos_sigma,
      config_.correction_pos_sigma, config_.correction_pos_sigma;
  correction_noise_ = gtsam::noiseModel::Diagonal::Sigmas(correction_sigmas);
}

void ImuStateEstimator::resetOptimization() {
  isam2_ = makeIsam2();
  pending_factors_.resize(0);
  pending_values_.clear();
}

void ImuStateEstimator::seedFromPrevState(
    const gtsam::SharedNoiseModel& pose_noise,
    const gtsam::SharedNoiseModel& velocity_noise,
    const gtsam::SharedNoiseModel& bias_noise) {
  const gtsam::Pose3 pose = prev_state_.pose();
  const gtsam::Velocity3 velocity = prev_state_.velocity();

  pending_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(sym::X(0), pose, pose_noise));
  pending_factors_.add(
      gtsam::PriorFactor<gtsam::Velocity3>(sym::V(0), velocity, velocity_noise));
  pending_factors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(
      sym::B(0), prev_bias_, bias_noise));
  pending_values_.insert(sym::X(0), pose);
  pending_values_.insert(sym::V(0), velocity);
  pending_values_.insert(sym::B(0), prev_bias_);

  isam2_.update(pending_factors_, pending_values_);
  pending_factors_.resize(0);
  pending_values_.clear();

  key_ = 1;
  // NB: the pending preintegration is deliberately NOT cleared here. On the
  // periodic reset path the samples accumulated since the last correction still
  // belong to the edge about to be built (LIO-SAM keeps them queued, unintegrated,
  // at the same point). initialize() clears it explicitly instead.
}

void ImuStateEstimator::initialize(const gtsam::Pose3& pose) {
  resetOptimization();
  prev_state_ = gtsam::NavState(pose, gtsam::Velocity3::Zero());
  prev_bias_ = gtsam::imuBias::ConstantBias{};
  preint_.reset(prev_bias_);  // anything before X(0) precedes the chain
  seedFromPrevState(prior_pose_noise_, prior_velocity_noise_, prior_bias_noise_);

  latest_ = ImuStateEstimate{};
  latest_.state = prev_state_;
  latest_.bias = prev_bias_;
  initialized_ = true;
}

void ImuStateEstimator::resetWithMarginals() {
  // Read the last node's marginals BEFORE wiping the graph, then re-seed node 0
  // with them so the fresh graph inherits the accumulated uncertainty
  // (imuPreintegration.cpp:353-381). If a marginal is unavailable, fall back to
  // the bootstrap priors rather than dropping the correction.
  gtsam::SharedNoiseModel pose_noise = prior_pose_noise_;
  gtsam::SharedNoiseModel velocity_noise = prior_velocity_noise_;
  gtsam::SharedNoiseModel bias_noise = prior_bias_noise_;
  try {
    const std::size_t last = key_ - 1;
    pose_noise =
        gtsam::noiseModel::Gaussian::Covariance(isam2_.marginalCovariance(sym::X(last)));
    velocity_noise =
        gtsam::noiseModel::Gaussian::Covariance(isam2_.marginalCovariance(sym::V(last)));
    bias_noise =
        gtsam::noiseModel::Gaussian::Covariance(isam2_.marginalCovariance(sym::B(last)));
  } catch (const std::exception&) {
    // keep the configured priors
  }

  resetOptimization();
  seedFromPrevState(pose_noise, velocity_noise, bias_noise);
  ++reset_count_;
}

bool ImuStateEstimator::addImuSample(const Eigen::Vector3d& accel,
                                     const Eigen::Vector3d& gyro, double dt) {
  if (!initialized_) {
    return false;
  }
  return preint_.integrate(accel, gyro, dt);
}

std::optional<ImuStateEstimate> ImuStateEstimator::correct(const gtsam::Pose3& lidar_pose) {
  if (!initialized_) {
    // First correction (or the one after a failure) bootstraps the chain at the
    // LiDAR pose, exactly as LIO-SAM's `systemInitialized == false` branch does.
    initialize(lidar_pose);
    return latest_;
  }
  if (preint_.count() == 0) {
    return std::nullopt;  // no IMU in this interval — ImuFactor would be empty
  }
  if (config_.reset_key_interval > 0 &&
      key_ >= static_cast<std::size_t>(config_.reset_key_interval)) {
    resetWithMarginals();
  }

  const gtsam::PreintegratedImuMeasurements& pim = preint_.finish();
  const std::size_t j = key_;
  const std::size_t i = key_ - 1;

  const gtsam::ImuFactor imu_factor(sym::X(i), sym::V(i), sym::X(j), sym::V(j),
                                    sym::B(i), pim);
  const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias> bias_factor(
      sym::B(i), sym::B(j), gtsam::imuBias::ConstantBias{},
      biasRandomWalkNoise(pim.deltaTij()));
  // The LiDAR pose enters ABSOLUTELY here — this is the whole point of the dual
  // graph. DEVIATION: LIO-SAM widens this to `correctionNoise2` when the mapper
  // flags a degenerate scan match; our front-end's NDT-11 gate drops every
  // non-Reliable registration before it is ever published (L5-06c), so no
  // degraded pose can reach this call and the branch would be dead code.
  const gtsam::PriorFactor<gtsam::Pose3> pose_factor(sym::X(j), lidar_pose,
                                                     correction_noise_);
  pending_factors_.add(imu_factor);
  pending_factors_.add(bias_factor);
  pending_factors_.add(pose_factor);

  // Initial values come from the IMU prediction, NOT from the LiDAR pose
  // (imuPreintegration.cpp:414-418).
  const gtsam::NavState predicted = preint_.predict(prev_state_, prev_bias_);
  pending_values_.insert(sym::X(j), predicted.pose());
  pending_values_.insert(sym::V(j), predicted.velocity());
  pending_values_.insert(sym::B(j), prev_bias_);

  gtsam::Values result;
  try {
    isam2_.update(pending_factors_, pending_values_);
    isam2_.update();
    pending_factors_.resize(0);
    pending_values_.clear();
    result = isam2_.calculateEstimate();
  } catch (const std::exception&) {
    // Indeterminate system (or any other solver failure): drop this correction
    // and re-initialize on the next one, same recovery as failureDetection.
    pending_factors_.resize(0);
    pending_values_.clear();
    initialized_ = false;
    ++failure_count_;
    return std::nullopt;
  }

  const gtsam::Pose3 pose_j = result.at<gtsam::Pose3>(sym::X(j));
  const gtsam::Velocity3 velocity_j = result.at<gtsam::Velocity3>(sym::V(j));
  const gtsam::imuBias::ConstantBias bias_j =
      result.at<gtsam::imuBias::ConstantBias>(sym::B(j));

  ImuStateEstimate estimate;
  estimate.state = gtsam::NavState(pose_j, velocity_j);
  estimate.bias = bias_j;
  estimate.imu_chi2 = 2.0 * imu_factor.error(result);
  estimate.bias_chi2 = 2.0 * bias_factor.error(result);
  estimate.prior_chi2 = 2.0 * pose_factor.error(result);
  estimate.delta_t_s = pim.deltaTij();
  estimate.imu_samples = preint_.count();

  prev_state_ = estimate.state;
  prev_bias_ = bias_j;
  preint_.reset(prev_bias_);  // next interval linearizes at the fresh bias

  if (failureDetected(velocity_j, bias_j)) {
    initialized_ = false;  // next correct() re-initializes at its LiDAR pose
    ++failure_count_;
    return std::nullopt;
  }

  ++key_;
  ++correction_count_;
  latest_ = estimate;
  return estimate;
}

bool ImuStateEstimator::failureDetected(
    const gtsam::Velocity3& velocity, const gtsam::imuBias::ConstantBias& bias) const {
  if (!velocity.allFinite() || velocity.norm() > config_.max_velocity_mps) {
    return true;
  }
  return bias.accelerometer().norm() > config_.max_bias_norm ||
         bias.gyroscope().norm() > config_.max_bias_norm;
}

gtsam::SharedNoiseModel ImuStateEstimator::biasRandomWalkNoise(double dt) const {
  const double s = std::sqrt(std::max(dt, 1e-9));
  gtsam::Vector6 sigmas;
  sigmas << config_.preint.accel_bias_rw_sigma * s,
      config_.preint.accel_bias_rw_sigma * s, config_.preint.accel_bias_rw_sigma * s,
      config_.preint.gyro_bias_rw_sigma * s, config_.preint.gyro_bias_rw_sigma * s,
      config_.preint.gyro_bias_rw_sigma * s;
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

}  // namespace graph_slam::imu
