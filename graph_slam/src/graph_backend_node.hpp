#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>

// Private header (lives in src/, NOT installed): factory so the executable's
// main() and the gtest smoke test can both construct the node without the
// class leaking out of its translation unit.
namespace graph_slam {

std::shared_ptr<rclcpp::Node> createGraphBackendNode(
    rclcpp::NodeOptions options = rclcpp::NodeOptions());

}  // namespace graph_slam
