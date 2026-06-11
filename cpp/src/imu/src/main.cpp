//
// Created by ckelton on 6/10/26.
//

#include <rclcpp/rclcpp.hpp>

#include "ImuPreintegrationNode.h"
#include "TransformFusionNode.h"


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node1 = std::make_shared<ImuPreintegrationNode>();
    auto node2 = std::make_shared<TransformFusionNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node1);
    executor.add_node(node2);
    executor.spin();

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
