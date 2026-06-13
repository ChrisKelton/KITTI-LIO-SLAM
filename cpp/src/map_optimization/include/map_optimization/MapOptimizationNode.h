//
// Created by ckelton on 6/11/26.
//
#pragma once

#ifndef MAP_OPTIMIZATION_MAPOPTIMIZATIONNODE_H
#define MAP_OPTIMIZATION_MAPOPTIMIZATIONNODE_H

#include <deque>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <opencv2/opencv.hpp>

#include <pcl/impl/point_types.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "slam_msgs/msg/cloud_info.hpp"
#include "slam_srvs/srv/save_map.hpp"
#include "slam_utilities/utility.h"
#include "utils/Config.h"


/*
 * A point cloud type that has 6D pose info ([x,y,z,roll.pitch,yaw] intensity is time stamp)
 */
// enforce SSE padding for correct memory alignment
struct alignas(16) PointXYZIRPYT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
};
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
    (float, x, x) (float, y, y) (float, z, z)
    (float, intensity, intensity)
    (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
    (double, time, time)
)

typedef PointXYZIRPYT PointTypePose;


class MapOptimizationNode final : public rclcpp::Node {
public:
    MapOptimizationNode();
    ~MapOptimizationNode() override;

    // Run as dedicated std::threads alongside rclcpp::spin (see main.cpp)
    void visualizeGlobalMapThread();
    void loopClosureThread();

private:
    Config* config;

    // gtsam
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2* isam2;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;

    rclcpp::Publisher<slam_msgs::msg::CloudInfo>::SharedPtr pubSLAMInfo;

    rclcpp::Subscription<slam_msgs::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    rclcpp::Service<slam_srvs::srv::SaveMap>::SharedPtr srvSaveMap;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    slam_msgs::msg::CloudInfo cloudInfo;

    std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    // Snapshots taken under mtx for the loop-closure thread, which reads keyframes concurrently
    // with the subscriber thread's push_back into the vectors above (a reallocation mid-read is UB).
    // Copying is cheap: these are vectors of shared_ptrs, so only the handles are duplicated.
    std::vector<pcl::PointCloud<PointType>::Ptr> copy_cornerCloudKeyFrames;
    std::vector<pcl::PointCloud<PointType>::Ptr> copy_surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // From feature_extraction?
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;  // corner feature set from odomOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;  // surf feature set from odomOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDs;  // downsampled corner feature set from odomOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDs;  // downsampled surf feature set from odomOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOriginal;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriginalCornerVec;  // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriginalCornerFlag;
    std::vector<PointType> laserCloudOriginalSurfVec;  // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriginalSurfFlag;

    std::map<int, std::pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDs;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDs;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downsizeFilterCorner;
    pcl::VoxelGrid<PointType> downsizeFilterSurf;
    pcl::VoxelGrid<PointType> downsizeFilterICP;
    pcl::VoxelGrid<PointType> downsizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerFromMapDsNum = 0;
    int laserCloudSurfFromMapDsNum = 0;
    int laserCloudCornerLastDsNum = 0;
    int laserCloudSurfLastDsNum = 0;

    bool aLoopIsClosed = false;
    std::map<int, int> loopIndexContainer;  // from new to old
    std::vector<std::pair<int, int>> loopIndexQueue;
    std::vector<gtsam::Pose3> loopPoseQueue;
    std::vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    std::deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    void setup_config();
    void initialize();
    void allocate_memory();
    void setup_subpub();

    void laserCloudInfoHandler(const slam_msgs::msg::CloudInfo::ConstSharedPtr& msg);
    void gpsHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
    void pointAssociateToMap(PointType const* const pt_in, PointType* const pt_out);
    bool saveMapService(slam_srvs::srv::SaveMap::Request& req, slam_srvs::srv::SaveMap::Response& res);
    void publishGlobalMap();
    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::ConstSharedPtr& msg);
    void performLoopClosure();
    bool detectLoopClosureDistance(int* latestId, int* closestId);
    bool detectLoopClosureExternal(int* latestId, int* closestId);
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum);
    void visualizeLoopClosure();
    void updateInitialGuess();
    void extractForLoopClosure();
    void extractNearby();
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);
    void extractSurroundingKeyFrames();
    void downsampleCurrentScan();
    void updatePointAssociateToMap();
    void cornerOptimization();
    void surfOptimization();
    void combineOptimizationCoeffs();
    bool LMOptimization(int iterCount);
    void scan2MapOptimization();
    void transformUpdate();
    bool saveFrame();
    void addOdomFactor();
    void addGPSFactor();
    void addLoopFactor();
    void saveKeyFramesAndFactor();
    void correctPoses();
    void updatePath(const PointTypePose& pose_in);
    void publishOdometry();
    void publishFrames();
};

#endif //MAP_OPTIMIZATION_MAPOPTIMIZATIONNODE_H