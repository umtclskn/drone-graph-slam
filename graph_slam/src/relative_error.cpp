#include "graph_slam/eval/relative_error.hpp"

#include <cmath>

namespace graph_slam {

RelativeErrorResult compute(const Eigen::Isometry3f& T_ndt, const Eigen::Isometry3f& T_gt) {
  const Eigen::Isometry3f dT = T_gt.inverse() * T_ndt;

  RelativeErrorResult out;
  out.t_err_m = dT.translation().norm();
  // Geodesic rotation angle: AngleAxis extracts the single-rotation angle of the
  // residual rotation, numerically robust near 0 and pi (no acos((tr-1)/2) edge
  // cases). Always reported in [0, 180] degrees.
  const Eigen::AngleAxisf aa(dT.linear());
  out.r_err_deg = aa.angle() * 180.0F / static_cast<float>(M_PI);
  return out;
}

}  // namespace graph_slam
