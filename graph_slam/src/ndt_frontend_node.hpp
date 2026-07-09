#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>

// Private header (lives in src/, NOT installed): exposes only a factory so the
// executable's main and the gtest smoke test can both build the node without the
// node class leaking out of its translation unit. The class itself stays in an
// anonymous namespace in ndt_frontend_node.cpp.
namespace graph_slam {

/// `options` lets tests inject parameter overrides (e.g. gate thresholds);
/// production callers use the default.
std::shared_ptr<rclcpp::Node> createNdtFrontendNode(rclcpp::NodeOptions options = {});

}  // namespace graph_slam
