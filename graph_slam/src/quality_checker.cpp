#include "graph_slam/preprocess/quality_checker.hpp"

#include <pcl/common/centroid.h>

#include <Eigen/Eigenvalues>
#include <sstream>

namespace graph_slam {

namespace {

/// Format a rejection reason carrying one numeric value at sensible precision.
std::string formatReason(const std::string& prefix, double value, double threshold,
                         const std::string& suffix) {
  std::ostringstream os;
  os.precision(4);
  os << prefix << value << " < " << threshold << suffix;
  return os.str();
}

}  // namespace

QualityResult QualityChecker::check(const CloudConstPtr& cloud) const {
  // 1. Point count: too sparse for NDT to build per-voxel Gaussians. Guards null
  //    / empty clouds too (size() == 0 < min_points), so the PCA below is safe.
  const int num_points = cloud ? static_cast<int>(cloud->size()) : 0;
  if (num_points < config_.min_points) {
    std::ostringstream os;
    os << "too few points: " << num_points << " < " << config_.min_points;
    return {false, os.str()};
  }

  // 2. Spread: the smallest eigenvalue of the point-position covariance measures
  //    extent along the least-spread direction. A plane or line collapses it to
  //    ~0 -> geometrically degenerate, NDT cannot constrain that direction.
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
  pcl::computeMeanAndCovarianceMatrix(*cloud, covariance, centroid);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance, Eigen::EigenvaluesOnly);
  const double min_eigenvalue = solver.eigenvalues()(0);  // ascending order
  if (min_eigenvalue < config_.min_spread_eigenvalue) {
    return {false, formatReason("insufficient spread: smallest eigenvalue ", min_eigenvalue,
                                config_.min_spread_eigenvalue, " (planar/degenerate scene)")};
  }

  return {true, {}};
}

}  // namespace graph_slam
