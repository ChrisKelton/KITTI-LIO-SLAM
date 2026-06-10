//
// Created by ckelton on 6/2/26.
//
#pragma once

#ifndef INC_3D_KITTI_OBJECTDETECTION_CONFIG_H
#define INC_3D_KITTI_OBJECTDETECTION_CONFIG_H

#include <cstdint>

enum struct DownsampleMethod : std::uint8_t {
    Voxel = 0,
    Uniform = 1,
    Farthest = 2
};


struct DownsampleSettings {
    DownsampleMethod method = DownsampleMethod::Voxel;
    // Voxel
    float voxelSize = 0.10f;  // 10 cm grid
    // Uniform
    std::size_t every_k_points = 4;  // drops every k-th laser point
    // Farthest
    std::size_t target_points = 16384;  // standard size for architectures like PointNet++
};


class Config {
public:
    Config() {};
    ~Config() {};

    DownsampleSettings downsample_settings;

    // Segment road from point cloud
    float segment_road_ransac_inliers_dist_th_m = 0.3f;  // used to determine inlier points to candidate planes when segmenting road from point cloud
    std::size_t segment_road_ransac_n = 3;  // number of points to use as inliers for plane calculation (minimum is 3)
    std::size_t segment_road_ransac_num_iterations = 150;  // number of iterations to run RANSAC

    // Cluster objects on road using DBSCAN
    // TODO: Switch this out with a deep learning model
    int dbscan_octree_resolution = 1;  // octree voxel size in meters; must be same scale as point cloud units
    float dbscan_eps = 0.45;
    std::size_t dbscan_cluster_min_points = 25;
    std::size_t dbscan_cluster_max_points = 300;
    bool use_oriented_bboxes = false;

    bool verbose = false;
};

#endif //INC_3D_KITTI_OBJECTDETECTION_CONFIG_H