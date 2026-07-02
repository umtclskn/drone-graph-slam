#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace graph_slam {

/// Single point type used throughout the SLAM pipeline (XYZ + intensity).
using PointT = pcl::PointXYZI;

/// PCL cloud aliases. These are the only internal cloud types graph_slam uses;
/// the ROS <-> PCL conversion (sensor_msgs/PointCloud2 <-> CloudPtr) is the
/// subscriber's job (NDT-02), keeping the algorithm classes ROS-free.
using Cloud = pcl::PointCloud<PointT>;
using CloudPtr = Cloud::Ptr;
using CloudConstPtr = Cloud::ConstPtr;

}  // namespace graph_slam
