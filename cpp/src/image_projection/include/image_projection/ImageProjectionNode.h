//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_IMAGEPROJECTIONNODE_H
#define KITTI_LIO_SLAM_IMAGEPROJECTIONNODE_H

#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/impl/instantiate.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "slam_msgs/msg/cloud_info.hpp"
#include "slam_utilities/utility.h"
#include "utils/Config.h"


// A single sensor agnostic representation.
// `alignas(16)`: forces 16-byte alignment for SIMD safety (PCL uses movaps internally).
struct alignas(16) LidarPoint {
    PCL_ADD_POINT4D;  // x, y, z + SSE padding
    float intensity;
    uint16_t ring;
    double time;  // seconds relative to scan start
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Generates template specializations PCL uses to introspect LidarPoint at compile time.
POINT_CLOUD_REGISTER_POINT_STRUCT(LidarPoint,
    (float, x, x) (float, y, y) (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, time, time)
)


constexpr int queueLength = 2000;


class ImageProjectionNode final : public rclcpp::Node {
private:
    std::mutex imuLock;
    std::mutex odomLock;

    // Point clouds
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointCloud;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<slam_msgs::msg::CloudInfo>::SharedPtr pubPointCloudInfo;

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    sensor_msgs::msg::PointCloud2 currentCloudMsg;

    // ::SharedPtr is ROS2 specific and not supported by PCL >= v1.12. Use ::Ptr for
    // point clouds that are not being published.
    pcl::PointCloud<LidarPoint>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr extractedCloud;

    slam_msgs::msg::CloudInfo cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::msg::Header cloudHeader;

    // IMU
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;

    std::unique_ptr<double[]> imuTime = std::make_unique<double[]>(queueLength);
    std::unique_ptr<double[]> imuRotX = std::make_unique<double[]>(queueLength);
    std::unique_ptr<double[]> imuRotY = std::make_unique<double[]>(queueLength);
    std::unique_ptr<double[]> imuRotZ = std::make_unique<double[]>(queueLength);

    int imuPointerCur;

    // Odometry
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // General
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    int ringFlag;
    int deskewFlag;
    cv::Mat rangeMat;

    // unique_ptr so the node owns the config and it is freed automatically (previously a raw
    // pointer that the destructor never deleted).
    std::unique_ptr<Config> config;

public:
    ImageProjectionNode();
    ~ImageProjectionNode() override;

    void setup_config();
    void setup_subpub();
    void initialize();
    void allocateMemory();
    void resetParameters();

    void imuHandler(const sensor_msgs::msg::Imu::ConstSharedPtr& msg);
    void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
    void cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);
    bool deskewInfo();
    void imuDeskewInfo();
    void odomDeskewInfo();

    void projectPointCloud();
    PointType deskewPoint(PointType* point, double relTime);
    void findRotation(double pointTime, float* rotXCur, float* rotYCur, float* rotZCur) const;
    void findPosition(double relTime, float* posXCur, float* posYCur, float* posZCur);

    void cloudExtraction();
    void publishClouds();
};

#endif //KITTI_LIO_SLAM_IMAGEPROJECTIONNODE_H
