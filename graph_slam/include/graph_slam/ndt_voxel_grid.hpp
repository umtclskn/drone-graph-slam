#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <functional>
#include <unordered_map>

#include "graph_slam/point_types.hpp"

namespace graph_slam {

/// Plain configuration for the NDT voxel grid. No ROS, no hidden behaviour:
/// tests construct one directly with literal values.
struct NdtGridConfig {
  /// Voxel edge length in metres; a point's integer cell index is floor(p / res).
  double resolution = 1.0;

  /// Cells with fewer points are skipped (no Gaussian): a covariance from one or
  /// two points is meaningless.
  std::size_t min_points = 5;

  /// Singular-covariance guard: each eigenvalue is floored at reg_ratio * lambda_max
  /// so the cell stays full-rank and Sigma^-1 is finite even for (near-)planar cells.
  double reg_ratio = 1e-3;
};

/// Integer index of a voxel (heart of the NDT partition: floor(p / resolution)).
struct VoxelKey {
  int i = 0;
  int j = 0;
  int k = 0;

  bool operator==(const VoxelKey& other) const {
    return i == other.i && j == other.j && k == other.k;
  }
};

/// Hash for VoxelKey so cells live in an unordered_map keyed by the 3 integers.
struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    // Mix the three components; constants are large odd primes (boost-style).
    std::size_t seed = std::hash<int>{}(key.i);
    seed ^= std::hash<int>{}(key.j) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(key.k) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    return seed;
  }
};

/// The per-voxel Gaussian: mean, (guarded) covariance, its inverse, and the point
/// count it was built from. All in double — covariance/eigen math wants the range.
struct VoxelCell {
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d cov_inv = Eigen::Matrix3d::Zero();
  std::size_t count = 0;
};

using VoxelCellMap = std::unordered_map<VoxelKey, VoxelCell, VoxelKeyHash>;

/// Single-responsibility NDT target builder: partitions a preprocessed cloud into
/// voxels (NDT-05) and fits one Gaussian per sufficiently populated voxel (NDT-06).
/// Pure algorithm, ROS-free and PCL-light (only iterates the cloud); does no map
/// management or alignment.
class NdtVoxelGrid {
 public:
  explicit NdtVoxelGrid(NdtGridConfig config) : config_(config) {}

  /// Partition `target` and (re)build the per-voxel Gaussians. Replaces any
  /// previous contents. A null or empty cloud leaves the grid empty.
  void build(const CloudConstPtr& target);

  /// Occupancy: number of voxels that produced a valid Gaussian (>= min_points).
  std::size_t numVoxels() const { return cells_.size(); }

  /// Cell edge length [m]. NdtRegistrar needs it for the Magnusson d1/d2 constants.
  double resolution() const { return config_.resolution; }

  /// Gaussian for a voxel, or nullptr if the voxel is empty / below min_points.
  const VoxelCell* cell(const VoxelKey& key) const;

  /// Integer voxel index of a point: floor(p / resolution), component-wise.
  VoxelKey voxelIndex(const Eigen::Vector3d& point) const;

  /// All valid cells, keyed by voxel index.
  const VoxelCellMap& cells() const { return cells_; }

 private:
  NdtGridConfig config_;
  VoxelCellMap cells_;
};

}  // namespace graph_slam
