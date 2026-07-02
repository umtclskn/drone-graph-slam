#include "graph_slam/graph/odometry_accumulator.hpp"

namespace graph_slam::graph {

OdometryAccumulator::OdometryAccumulator() { reset(); }

void OdometryAccumulator::reset() {
  delta_acc_ = gtsam::Pose3();          // identity
  sigma_acc_ = gtsam::Matrix66::Zero();
}

void OdometryAccumulator::add_measurement(const gtsam::Pose3& delta_step,
                                          const gtsam::Matrix66& sigma_step) {
  gtsam::Matrix66 Ja;  // d(acc * step) / d(acc)
  gtsam::Matrix66 Jb;  // d(acc * step) / d(step)
  delta_acc_ = delta_acc_.compose(delta_step, Ja, Jb);
  sigma_acc_ = Ja * sigma_acc_ * Ja.transpose() + Jb * sigma_step * Jb.transpose();
}

AccumulatedOdometry OdometryAccumulator::current() const {
  return AccumulatedOdometry{delta_acc_, sigma_acc_};
}

}  // namespace graph_slam::graph
