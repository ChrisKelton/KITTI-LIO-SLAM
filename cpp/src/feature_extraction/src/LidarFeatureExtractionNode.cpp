//
// Created by ckelton on 6/2/26.
//

/*
Using brackets looks for system installed packages at /usr/include/ ...

Using quotes for package means to search locally for the include file and on any paths included by -I, if it is not
    found, then default back to looking like brackets looks. I.e., in /usr/include/, ...
*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <dbscan/dbScan.h>

#include "LidarFeatureExtractionNode.h"
#include "utils/Config.h"

void LidarFeatureExtractionNode::handlePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    /*
        1. Ingest point cloud from msg into PCL point cloud (does PCL have the same capabilities as open3d?).
        2. Downsample point cloud.
        3. Segment road from point cloud.
        4. Run DBSCAN on point cloud.
        5. Store point cloud into this->pointClouds removing any older pointClouds that need to be removed.
    */
    total_point_clouds++;
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Ingesting new point cloud. Number of point clouds ingested: %d", total_point_clouds);

    // 1. Create a native PCL point cloud object
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    // Convert the ROS message to the PCL object
    pcl::fromROSMsg(*msg, *cloud);
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Number of points in cloud '%zu'", cloud->points.size());

    // 2. Downsample point cloud
    DownsampleSettings ds_settings = config->downsample_settings;
    if (ds_settings.method == DownsampleMethod::Voxel) {
        // Instantiate the VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZI> vox;
        vox.setInputCloud(cloud);
        // Set the voxel leaf size
        vox.setLeafSize(ds_settings.voxelSize, ds_settings.voxelSize, ds_settings.voxelSize);
        vox.filter(*cloud_filtered);
    } else if (ds_settings.method == DownsampleMethod::Uniform) {
        throw std::logic_error("Functionality for 'Uniform' point cloud downsampling method not implemented yet.");
    } else if (ds_settings.method == DownsampleMethod::Farthest) {
        throw std::logic_error("Functionality for 'Farthest' point cloud downsampling method not implemented yet.");
    } else {
        throw std::logic_error("Unrecognized downsampling method.");
    }
    cloud.reset();
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Number of points after downsampling '%zu'", cloud_filtered->points.size());

    // 3. Segment road from point cloud
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setInputCloud(cloud_filtered);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(config->segment_road_ransac_inliers_dist_th_m);
    seg.setMaxIterations(config->segment_road_ransac_num_iterations);

    seg.segment(*inliers, *coefficients);
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Number of road points '%zu'", inliers->indices.size());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_objects_xyzi(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_road_xyzi(new pcl::PointCloud<pcl::PointXYZI>);
    extract.setNegative(false);
    extract.filter(*cloud_road_xyzi);

    if (roadPointClouds.size() >= max_num_point_clouds) {
        roadPointClouds.pop();
    }
    roadPointClouds.push(cloud_road_xyzi);

    extract.setNegative(true);
    extract.filter(*cloud_objects_xyzi);
    if (objectPointClouds.size() >= max_num_point_clouds) {
        objectPointClouds.pop();
    }
    objectPointClouds.push(cloud_objects_xyzi);

    // TODO: Replace this with PointNet++ or some other deep learning architecture
    // 4. Run DBSCAN
    // DBSCAN requires pcl::PointXYZRGB, so make that.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_objects_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_objects_xyzi, *cloud_objects_rgb);

    RCLCPP_INFO(this->get_logger(), "[Velodyne] Number of points in objects cloud '%zu'", cloud_objects_rgb->points.size());
    std::vector<htr::Point3D> dummy;  // unused by the library, but required by the template
    dbScanSpace::dbscan dbscan;
    // parameters after cloud_objects_rgb
    //     octreeResolution = 120 (voxel size for the octree)
    //     eps = 40.0f (search radius in same units as your point cloud; e.g., KITTI point cloud would be in meters)
    //     minPtsAux = 5 (minimum points for initial micro-clusters)
    //     minPts = 5 (minimum points for final merged clusters [must be >= 3])
    dbscan.init(dummy, cloud_objects_rgb, config->dbscan_octree_resolution, config->dbscan_eps, 5, config->dbscan_cluster_min_points);
    dbscan.generateClusters_fast();

    std::vector<dbScanSpace::cluster>& clusters = dbscan.getClusters();
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Number of clusters found '%zu'", clusters.size());

    auto hsv_to_rgb = [](float hue) -> std::array<uint8_t, 3> {
        float s = 1.0f, v = 1.0f;
        float c = v * s;
        float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
        float m = v - c;

        float r, g, b;
        if      (hue < 60)  { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else                { r = c; g = 0; b = x; }

        return {
            static_cast<uint8_t>((r + m) * 255),
            static_cast<uint8_t>((g + m) * 255),
            static_cast<uint8_t>((b + m) * 255)
        };
    };

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters[i].clusterPoints.size() > config->dbscan_cluster_max_points)
            continue;
        float hue = i * (360.0f / clusters.size());
        auto [r, g, b] = hsv_to_rgb(hue);
        for (const auto& pt : clusters[i].clusterPoints) {
            pcl::PointXYZRGB p;
            p.x = pt.x; p.y = pt.y; p.z = pt.z;
            p.r = r; p.g = g; p.b = b;
            cloud_clustered->push_back(p);
        }
    }

    RCLCPP_INFO(this->get_logger(), "[Velodyne] Total number of detected object points in cloud '%zu'", cloud_clustered->points.size());

    if (objectDetectionPointClouds.size() >= max_num_point_clouds) {
        objectDetectionPointClouds.pop();
    }
    objectDetectionPointClouds.push(cloud_clustered);

    publishPointClouds(msg->header);
}


void LidarFeatureExtractionNode::publishPointClouds(const std_msgs::msg::Header& header) {
    // TODO: Make own msg that is a vector of msg::PointCloud look at:
    //     https://robotics.stackexchange.com/questions/94297/publishing-a-vector-of-point-clouds
    //     https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv
    sensor_msgs::msg::PointCloud2 output_msg;
    auto object_detections_pt_cloud = objectDetectionPointClouds.front();
    auto road_pt_cloud = roadPointClouds.front();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr road_pt_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*road_pt_cloud, *road_pt_cloud_rgb);
    auto road_red = get_road_red();
    auto road_green = get_road_green();
    auto road_blue = get_road_blue();
    for (auto& point : road_pt_cloud_rgb->points) {
        point.r = road_red;
        point.g = road_green;
        point.b = road_blue;
    }
    *object_detections_pt_cloud += *road_pt_cloud_rgb;
    RCLCPP_INFO(this->get_logger(), "[Velodyne] Publishing point cloud with '%zu' points", object_detections_pt_cloud->points.size());
    pcl::toROSMsg(*object_detections_pt_cloud, output_msg);
    output_msg.header = header;
    output_msg.header.frame_id = "feature_extraction";

    pubPointCloud->publish(output_msg);
}
