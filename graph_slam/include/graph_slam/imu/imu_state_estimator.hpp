// DUAL-GRAPH (src/DUAL_GRAPH_ANALYSIS.md, 2026-08-03): the LIO-SAM-style IMU
// state estimator — a SECOND, small factor graph that owns the IMU chain, kept
// disjoint from the mapping pose graph.
//
// Why a second graph. Today (single-graph) the `ImuFactor` and the NDT
// `BetweenFactor<Pose3>` sit on the SAME X(i) nodes, so any error in the
// preintegrated IMU measurement is injected straight into the pose solution and
// its marginals. LIO-SAM instead keeps the `ImuFactor` in its own tiny iSAM2
// where the LiDAR pose enters as an ABSOLUTE `PriorFactor<Pose3>`, and its
// mapping pose graph never sees an IMU factor at all. This class is that side
// graph: a direct port of `LIO-SAM/src/imuPreintegration.cpp:265-490`
// (`odometryHandler` + `failureDetection`) onto our `ImuPreintegrator`.
//
// What it is for. Its only products are (a) an optimized (pose, velocity, bias)
// NavState that seeds the front-end's IMU predictor and (b) diagnostics. Its
// covariance is deliberately NOT part of the ARCHITECTURE §6 covariance chain —
// Sigma_meas -> Sigma_prop -> BetweenFactor -> Sigma_post lives entirely in the
// mapping pose graph, which this class does not touch.
//
// Pure algorithm class: no ROS, no PX4 (unit-tested in
// test/test_imu_state_estimator.cpp). Failure is reported by return value; the
// node decides what to do with it.
#pragma once

#include <cstddef>
#include <optional>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "graph_slam/imu/imu_preintegrator.hpp"

namespace graph_slam::imu {

// Knobs of the side graph. The preintegration noise is shared with the rest of
// the stack (same PreintegrationConfig the back-end builds from slam_params.yaml)
// so the two graph architectures are compared at identical IMU noise.
struct ImuStateEstimatorConfig {
  PreintegrationConfig preint{};

  // Bootstrap priors on X(0)/V(0)/B(0). Defaults are the L5-04 sigmas, NOT
  // LIO-SAM's (its priorBiasNoise sigma 1e-3 fights the ~0.036 m/s^2 accel bias
  // this rig settles at, at ~36 sigma — L5-03 journal).
  double prior_pose_sigma{0.001};        // rad & m, all six dof
  double prior_velocity_sigma{0.1};      // m/s (at-rest bootstrap)
  double prior_accel_bias_sigma{0.1};    // m/s^2
  double prior_gyro_bias_sigma{0.01};    // rad/s

  // Noise of the ABSOLUTE PriorFactor<Pose3> that carries each NDT correction
  // into this graph. Hand-fixed constants, exactly as LIO-SAM does
  // (`correctionNoise`, imuPreintegration.cpp:257): our Sigma_meas/Sigma_prop
  // are RELATIVE-edge covariances (ARCHITECTURE §6 #1/#2) and are not a valid
  // absolute uncertainty for a drifting odometry pose.
  double correction_rot_sigma{0.05};     // rad, per axis
  double correction_pos_sigma{0.1};      // m, per axis

  // Health / cost guards (ports of imuPreintegration.cpp:353-381, :472-490).
  // reset_key_interval: rebuild the graph from the last node's marginals every
  // N corrections so the side graph stays O(1) forever (<=0 disables).
  int reset_key_interval{100};
  double max_velocity_mps{30.0};         // above this -> failure -> re-init
  double max_bias_norm{1.0};             // accel or gyro bias norm
};

// One optimized side-graph state, returned by correct().
struct ImuStateEstimate {
  gtsam::NavState state;                 // optimized pose + velocity
  gtsam::imuBias::ConstantBias bias;     // optimized IMU bias
  double imu_chi2{-1.0};                 // 2*ImuFactor.error on the fresh estimate
  double bias_chi2{-1.0};                // 2*BetweenFactor<imuBias>.error
  double prior_chi2{-1.0};               // 2*PriorFactor<Pose3>.error (LiDAR pull)
  double delta_t_s{0.0};                 // preintegration span of this correction
  std::size_t imu_samples{0};            // samples folded into that span
};

class ImuStateEstimator {
 public:
  explicit ImuStateEstimator(const ImuStateEstimatorConfig& config);

  // Seed X(0)/V(0)/B(0) at `pose`, zero velocity, zero bias, each with its
  // configured prior (port of imuPreintegration.cpp:307-350). Clears any
  // previous graph, so it doubles as the re-initialization path.
  void initialize(const gtsam::Pose3& pose);

  [[nodiscard]] bool initialized() const { return initialized_; }

  // Fold one IMU sample into the pending preintegration. dt<=0 is dropped by
  // ImuPreintegrator (bag replay yields ~0.4 % duplicate sim-time stamps).
  // Ignored until initialize() has run — samples before X(0) precede the chain.
  bool addImuSample(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt);

  // One correction: ImuFactor + bias BetweenFactor + the LiDAR pose as an
  // ABSOLUTE PriorFactor<Pose3> on X(key), initial values from the IMU
  // prediction, two iSAM2 updates, read-back, preintegrator reset to the fresh
  // bias (port of imuPreintegration.cpp:384-470).
  //
  // Returns nullopt when the correction could not be applied:
  //   * no IMU accumulated since the last correction (nothing to constrain);
  //   * the post-solve state failed the velocity/bias sanity check;
  //   * iSAM2 threw (indeterminate system).
  // The last two also drop `initialized_`, so the NEXT call re-initializes at
  // the pose it is given — the same self-healing LIO-SAM gets from
  // resetParams() + its `systemInitialized` branch.
  std::optional<ImuStateEstimate> correct(const gtsam::Pose3& lidar_pose);

  // Freshest optimized state (the bootstrap state until the first correction).
  [[nodiscard]] const ImuStateEstimate& latest() const { return latest_; }

  // Health counters for diagnostics.
  [[nodiscard]] std::size_t key() const { return key_; }
  [[nodiscard]] std::size_t correctionCount() const { return correction_count_; }
  [[nodiscard]] std::size_t resetCount() const { return reset_count_; }
  [[nodiscard]] std::size_t failureCount() const { return failure_count_; }

 private:
  // Wipe the iSAM2 + staging buffers (port of resetOptimization(), :265-277).
  void resetOptimization();

  // Insert X(0)/V(0)/B(0) = prev_* with the given priors and optimize once.
  // Shared by initialize() and the periodic marginals-as-priors reset.
  void seedFromPrevState(const gtsam::SharedNoiseModel& pose_noise,
                         const gtsam::SharedNoiseModel& velocity_noise,
                         const gtsam::SharedNoiseModel& bias_noise);

  // Periodic reset for speed: re-seed X/V/B(0) from the last node's marginal
  // covariances so the graph never grows (port of :353-381).
  void resetWithMarginals();

  [[nodiscard]] bool failureDetected(const gtsam::Velocity3& velocity,
                                     const gtsam::imuBias::ConstantBias& bias) const;

  // sqrt(dt) * [accBiasRW x3, gyrBiasRW x3] — LIO-SAM's noiseModelBetweenBias
  // scaling (ConstantBias tangent order is accelerometer-then-gyroscope).
  [[nodiscard]] gtsam::SharedNoiseModel biasRandomWalkNoise(double dt) const;

  ImuStateEstimatorConfig config_;
  ImuPreintegrator preint_;

  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph pending_factors_;
  gtsam::Values pending_values_;

  gtsam::SharedNoiseModel prior_pose_noise_;
  gtsam::SharedNoiseModel prior_velocity_noise_;
  gtsam::SharedNoiseModel prior_bias_noise_;
  gtsam::SharedNoiseModel correction_noise_;

  gtsam::NavState prev_state_;
  gtsam::imuBias::ConstantBias prev_bias_;
  ImuStateEstimate latest_;

  bool initialized_{false};
  std::size_t key_{0};
  std::size_t correction_count_{0};
  std::size_t reset_count_{0};
  std::size_t failure_count_{0};
};

}  // namespace graph_slam::imu
