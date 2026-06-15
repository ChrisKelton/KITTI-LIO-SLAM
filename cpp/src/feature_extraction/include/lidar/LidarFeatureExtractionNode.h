//
// Created by ckelton on 6/2/26.
//
#pragma once

#ifndef FEATURE_EXTRACTION_LIDAR_FEATURE_EXTRACTION_NODE_H
#define FEATURE_EXTRACTION_LIDAR_FEATURE_EXTRACTION_NODE_H

#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include "slam_msgs/msg/cloud_info.hpp"
#include "utils/Config.h"


typedef pcl::PointXYZI PointType;


struct smoothness_t {
    float value;
    size_t ind;
};


class LidarFeatureExtractionNode : public rclcpp::Node {
public:
    LidarFeatureExtractionNode() : Node("lidar_feature_extraction") {
        config = new Config();
        setup_config();
        initialize();
        setup_subpub();

        RCLCPP_INFO(this->get_logger(), "Setup LidarFeatureExtractionNode!");
    }

    ~LidarFeatureExtractionNode() {
        delete config;
    }

private:
    Config* config;

    rclcpp::Subscription<slam_msgs::msg::CloudInfo>::SharedPtr subLaserCloudInfo;

    rclcpp::Publisher<slam_msgs::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downsizeFilter;

    slam_msgs::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    std::vector<float> cloudCurvature;
    std::vector<int> cloudNeighborPicked;
    std::vector<int> cloudLabel;

    void setup_config();
    void initialize();
    void setup_subpub();

    void laserCloudInfoHandler(const slam_msgs::msg::CloudInfo::ConstSharedPtr& msg);
    void calculateSmoothness();
    void markOccludedPoints();
    void extractFeatures();
    void freeCloudInfoMemory();
    void publishFeatureCloud();
};

#endif //FEATURE_EXTRACTION_LIDAR_FEATURE_EXTRACTION_NODE_H
