//
// Created by ckelton on 6/2/26.
//
#pragma once

#ifndef INC_3D_KITTI_OBJECTDETECTION_OBJECT_DETECTION_NODE_H
#define INC_3D_KITTI_OBJECTDETECTION_OBJECT_DETECTION_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <queue>

#include "utils/Config.h"


typedef pcl::PointXYZI PointType;


struct smoothness_t {
    float value;
    size_t ind;
};


struct by_value {
    bool operator()(smoothness_t const& left, smoothness_t const& right) {
        return left.value < right.value;
    }
};


class LidarFeatureExtractionNode : public rclcpp::Node {
public:
    LidarFeatureExtractionNode() : Node("lidar_feature_extraction") {
        config = new Config();
        setup_config();
        initialize();
        setup_subpub();
    }

    ~LidarFeatureExtractionNode() {
        delete config;
        //delete[] cloudCurvature;
        //delete[] cloudNeighborPicked;
        //delete[] cloudLabel;
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointCloud;

    rclcpp::Publisher<lio_sam::cloud_info>::SharedPtr pubPointCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    std::unique_ptr<float> cloudCurvature;
    std::unique_ptr<int> cloudNeighborPicked;
    std::unique_ptr<int> cloudLabel;
    //float *cloudCurvature;
    //int *cloudNeighborPicked;
    //int *cloudLabel;

    // Config* config;

    // int max_num_point_clouds = 1;
    // // Have separate point clouds for roads & non-road objects
    // std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> roadPointClouds;
    // std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> objectPointClouds;
    // std::queue<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> objectDetectionPointClouds;
    // int total_point_clouds = 0;

    std::uint32_t roadHexColor = 0xFFFFFF;
    inline std::uint8_t get_road_red() {
        return (roadHexColor >> 16) & 0xFF;
    }
    inline std::uint8_t get_road_green() {
        return (roadHexColor >> 8) & 0xFF;
    }
    inline std::uint8_t get_road_blue() {
        return roadHexColor & 0xFF;
    }

    void setup_config() {
        //
        config->N_SCAN = this->declare_parameter<int>("N_SCAN", 64);
        config->Horizon_SCAN = this->declare_parameter<int>("Horizon_SCAN", 4096);
        config->downsampleRate = this->declare_parameter<int>("downsampleRate", 1);
        // Values specific for KITTI dataset
        //     Most ground-truth in benchmark are maxed to 80m
        config->lidarMinRange = this->declare_parameter<float>("lidarMinRange", 1.0);
        config->lidarMaxRange = this->declare_parameter<float>("lidarMaxRange", 400.0);

        config->surfLeafSize = this->declare_parameter<float>("surfLeafSize", 0.4);
    }

    void initialize() {
        cloudSmoothness.resize(config->N_SCAN*config->Horizon_SCAN);
        downsizeFilter.setLeafSize(config->surfLeafSize, config->surfLeafSize, config->surfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[config->N_SCAN*config->Horizon_SCAN];
        cloudNeighborPicked = new int[config->N_SCAN*config->Horizon_SCAN];
        cloudLabel = new int[config->N_SCAN*config->Horizon_SCAN];
    }

    void setup_subpub() {
        subPointCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature_extraction/data_pcl", rclcpp::SensorDataQoS(),
            std::bind(&VelodyneNode::handlePointCloud, this, std::placeholders::_1));
        RCLCPP_INFO(this->get_logger(), "Subscribing: feature_extraction/data_pcl");

        pubPointCloudInfo = this->create_publish<lio_sam::cloud_info>("feature/cloud_info", 1);
        RCLCPP_INFO(this->get_logger(), "Publishing: feature/cloud_info");
        pubCornerPoints = this->create_publish<sensor_msgs::msg::PointCloud2>("feature/cloud_corner", 1);
        RCLCPP_INFO(this->get_logger(), "Publishing: feature/cloud_corner");
        pubSurfacePoints = this->create_publish<sensor_msgs::msg::PointCloud2>("feature/cloud_surface", 1);
        RCLCPP_INFO(this->get_logger(), "Publishing: feature/cloud_surface");

        //pubPointCloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        //    "feature_extraction/detections_pcl", max_num_point_clouds + 1);
        //RCLCPP_INFO(this->get_logger(), "Publishing: feature_extraction/detections_pcl");
    }

    void handlePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

    void publishPointClouds(const std_msgs::msg::Header& header);
};

#endif //INC_3D_KITTI_OBJECTDETECTION_OBJECT_DETECTION_NODE_H