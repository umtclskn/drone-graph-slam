#include "graph_slam/eval/trajectory_metrics.hpp"

#include <cmath>

namespace graph_slam {
namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

double rotationAngle(const Eigen::Matrix3f& r) {
  // AngleAxis is numerically robust near 0 and pi, unlike acos((tr-1)/2).
  return static_cast<double>(Eigen::AngleAxisf(r).angle());
}

}  // namespace

AbsoluteTrajectoryError absoluteTrajectoryError(const Trajectory& est, const Trajectory& gt) {
  AbsoluteTrajectoryError out;
  if (est.empty() || est.size() != gt.size()) {
    return out;
  }

  // First-pose alignment: rigid transform that maps est pose 0 onto gt pose 0.
  // Pins the open chain at its start so the error is pure accumulated drift.
  const Eigen::Isometry3f align = gt.front() * est.front().inverse();

  double trans_sq = 0.0;
  double rot_sq = 0.0;
  for (std::size_t i = 0; i < est.size(); ++i) {
    const Eigen::Isometry3f error = gt[i].inverse() * (align * est[i]);
    trans_sq += static_cast<double>(error.translation().squaredNorm());
    const double angle = rotationAngle(error.linear());
    rot_sq += angle * angle;
  }

  const auto n = static_cast<double>(est.size());
  out.count = est.size();
  out.trans_rmse_m = static_cast<float>(std::sqrt(trans_sq / n));
  out.rot_rmse_deg = static_cast<float>(std::sqrt(rot_sq / n) * kRadToDeg);
  return out;
}

RelativePoseError relativePoseError(const Trajectory& est, const Trajectory& gt,
                                    std::size_t delta) {
  RelativePoseError out;
  out.delta = delta;
  if (delta == 0 || est.empty() || est.size() != gt.size() || est.size() <= delta) {
    return out;
  }

  double trans_sq = 0.0;
  double rot_sq = 0.0;
  std::size_t count = 0;
  for (std::size_t i = 0; i + delta < est.size(); ++i) {
    const Eigen::Isometry3f gt_rel = gt[i].inverse() * gt[i + delta];
    const Eigen::Isometry3f est_rel = est[i].inverse() * est[i + delta];
    const Eigen::Isometry3f residual = gt_rel.inverse() * est_rel;
    trans_sq += static_cast<double>(residual.translation().squaredNorm());
    const double angle = rotationAngle(residual.linear());
    rot_sq += angle * angle;
    ++count;
  }

  const auto m = static_cast<double>(count);
  out.count = count;
  out.trans_rmse_m = static_cast<float>(std::sqrt(trans_sq / m));
  out.rot_rmse_deg = static_cast<float>(std::sqrt(rot_sq / m) * kRadToDeg);
  return out;
}

}  // namespace graph_slam
