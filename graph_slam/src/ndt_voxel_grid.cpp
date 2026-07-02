#include "graph_slam/ndt_voxel_grid.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>

namespace graph_slam {

namespace {

/// Running per-voxel accumulator used during the partition pass. Mean and sample
/// covariance both come from these sums: the second moment Sigma_pp lets us form
/// the covariance as (Sigma_pp - sum sum^T / N) / (N - 1) in one pass.
struct VoxelAccumulator {
  std::size_t count = 0;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();        // sum of points
  Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();  // sum of p * p^T
};

}  // namespace

VoxelKey NdtVoxelGrid::voxelIndex(const Eigen::Vector3d& point) const {
  const double res = config_.resolution;
  return VoxelKey{static_cast<int>(std::floor(point.x() / res)),
                  static_cast<int>(std::floor(point.y() / res)),
                  static_cast<int>(std::floor(point.z() / res))};
}

const VoxelCell* NdtVoxelGrid::cell(const VoxelKey& key) const {
  const auto it = cells_.find(key);
  return it == cells_.end() ? nullptr : &it->second;
}

void NdtVoxelGrid::build(const CloudConstPtr& target) {
  cells_.clear();
  if (!target || target->empty()) {
    return;
  }

  // --- NDT-05: partition points into voxels, accumulating moments per cell ----
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> acc;
  acc.reserve(target->size());
  for (const auto& pt : target->points) {
    const Eigen::Vector3d p(static_cast<double>(pt.x), static_cast<double>(pt.y),
                            static_cast<double>(pt.z));
    VoxelAccumulator& a = acc[voxelIndex(p)];
    ++a.count;
    a.sum += p;
    a.sum_outer += p * p.transpose();
  }

  // --- NDT-06: fit one Gaussian per sufficiently populated voxel --------------
  for (const auto& [key, a] : acc) {
    if (a.count < config_.min_points) {
      continue;  // single-/sparse-point voxels are dropped
    }

    const double n = static_cast<double>(a.count);
    const Eigen::Vector3d mean = a.sum / n;
    // Sample covariance with the (N-1) denominator.
    Eigen::Matrix3d cov = (a.sum_outer - a.sum * a.sum.transpose() / n) / (n - 1.0);

    // Singular guard: floor every eigenvalue at reg_ratio * lambda_max so the
    // cell stays full-rank (planar voxels otherwise give a non-invertible Sigma).
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d evals = solver.eigenvalues();  // ascending
    const Eigen::Matrix3d& evecs = solver.eigenvectors();
    const double floor = config_.reg_ratio * evals(2);  // reg_ratio * lambda_max
    for (int idx = 0; idx < 3; ++idx) {
      evals(idx) = std::max(evals(idx), floor);
    }
    cov = evecs * evals.asDiagonal() * evecs.transpose();
    const Eigen::Matrix3d cov_inv = evecs * evals.cwiseInverse().asDiagonal() * evecs.transpose();

    cells_.emplace(key, VoxelCell{mean, cov, cov_inv, a.count});
  }
}

}  // namespace graph_slam
