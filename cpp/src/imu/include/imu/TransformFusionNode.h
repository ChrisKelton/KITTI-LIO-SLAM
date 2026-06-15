//
// Created by ckelton on 6/10/26.
//
#pragma once

#ifndef IMU_TRANSFORMFUSIONNODE_H
#define IMU_TRANSFORMFUSIONNODE_H

#include <deque>
#include <mutex>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include "slam_utilities/utility.h"
#include "utils/TransformFusionNodeConfig.h"


class TransformFusionNode : public rclcpp::Node {
public:
    TransformFusionNode();
    ~TransformFusionNode() override;

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& odomMsg);
    void imuOdometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& odomMsg);

private:
    void setup_config();
    void initialize();
    void setup_subpub();
    // Publishes a latched box marker in the base_link frame so RViz renders a vehicle body that
    // rides the odom->base_link transform (i.e. follows the trajectory).
    void publishRobotMarker();

    TransformFusionNodeConfig* config;
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pubRobotMarker;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    std::unique_ptr<tf2_ros::Buffer>            tfBuffer;
    std::unique_ptr<tf2_ros::TransformListener> tfListener;
    geometry_msgs::msg::TransformStamped        lidar2Baselink;

    double lidarOdomTime = -1;
    std::deque<nav_msgs::msg::Odometry> imuOdomQueue;

    tf2_ros::StaticTransformBroadcaster tfMap2Odom;
    tf2_ros::StaticTransformBroadcaster tfOdom2BaseLink;
};


#endif //IMU_TRANSFORMFUSIONNODE_H
