#include "graph_slam/imu/imu_window_buffer.hpp"

#include <algorithm>

namespace graph_slam::imu {

void ImuWindowBuffer::reset(double anchor_stamp_s) {
  while (!queue_.empty() && queue_.front().stamp_s <= anchor_stamp_s) {
    queue_.pop_front();
  }
  last_stamp_s_ = anchor_stamp_s;
}

void ImuWindowBuffer::pruneOlderThan(double stamp_s) {
  while (!queue_.empty() && queue_.front().stamp_s < stamp_s) {
    queue_.pop_front();
  }
}

WindowIntegrationResult ImuWindowBuffer::integrateUpTo(double cut_time_s,
                                                       double max_dt_gap_s,
                                                       ImuPreintegrator& preint) {
  WindowIntegrationResult result;
  while (!queue_.empty() && queue_.front().stamp_s < cut_time_s) {
    const ImuSample sample = queue_.front();
    queue_.pop_front();
    const double dt = sample.stamp_s - last_stamp_s_;
    // dt<=0 / non-finite (duplicate or out-of-order stamp, or a never-anchored
    // buffer) is skipped inside integrate(); the sample is still consumed.
    if (last_stamp_s_ >= 0.0 && preint.integrate(sample.accel, sample.gyro, dt)) {
      ++result.integrated;
      if (dt > result.max_dt_s) {
        result.max_dt_s = dt;
        result.max_dt_stamp_s = sample.stamp_s;
      }
      if (dt > max_dt_gap_s) {
        result.gap_detected = true;
      }
    }
    // Never move the anchor backwards: an out-of-order stamp must not shrink
    // the next sample's dt below its true value.
    last_stamp_s_ = std::max(last_stamp_s_, sample.stamp_s);
  }
  return result;
}

}  // namespace graph_slam::imu
