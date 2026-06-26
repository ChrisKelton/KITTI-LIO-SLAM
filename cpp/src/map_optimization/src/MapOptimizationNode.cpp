//
// Created by ckelton on 6/11/26.
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include "MapOptimizationNode.h"
#include "slam_utilities/map_utilities.h"

// Single-residual scan-matching factor for the GTSAM back-end (config useGtsamScanMatcher).
// residual = nᵀ (T · p) + d, with n the unit feature normal/direction in the map frame and p the
// point in the lidar/body frame. Covers both planar (point-to-plane) and edge (point-to-line along
// the perpendicular) correspondences, mirroring how the custom LMOptimization linearizes each as a
// 1-D residual along the direction stored in coeffSel.
class PointPlaneFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    gtsam::Point3  p_;   // point in the lidar/body frame
    gtsam::Vector3 n_;   // unit feature normal/direction, in the map frame
    double         d_;   // offset so residual == nᵀ(T·p) + d
public:
    PointPlaneFactor(gtsam::Key key, gtsam::Point3 p, gtsam::Vector3 n, double d,
                     const gtsam::SharedNoiseModel& model)
        : gtsam::NoiseModelFactorN<gtsam::Pose3>(model, key),
          p_(std::move(p)), n_(std::move(n)), d_(d) {}

    // GTSAM 4.2 custom-factor signature (boost::optional Jacobian).
    gtsam::Vector evaluateError(const gtsam::Pose3& T,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::Matrix36 Dpose;                          // ∂(T·p)/∂T  (3×6)
        const gtsam::Point3 q = T.transformFrom(p_, H ? &Dpose : nullptr);
        if (H) *H = n_.transpose() * Dpose;             // 1×6 analytic Jacobian
        return gtsam::Vector1(n_.dot(q) + d_);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PointPlaneFactor(*this)));
    }
};


MapOptimizationNode::MapOptimizationNode() : Node("map_optimization") {
    config = new Config;
    setup_config();
    initialize();
    setup_subpub();

    RCLCPP_INFO(this->get_logger(), "Setup MapOptimizationNode!");
}

MapOptimizationNode::~MapOptimizationNode() {
    delete config;
    delete isam2;
}


void MapOptimizationNode::setup_config() {
    config->numberOfCores = this->declare_parameter("numberOfCores", 2);

    config->gpsTopic = this->declare_parameter("gpsTopic", "odometry/gps");
    config->savePCD = this->declare_parameter("savePCD", true);  // TODO: Maybe default this to `false`
    config->savePCDDirectory = this->declare_parameter("savePCDDirectory", "/Downloads/LIO");
    config->useGtsamScanMatcher = this->declare_parameter("useGtsamScanMatcher", false);
    config->odometryFrame = this->declare_parameter("odometryFrame", "odom");
    config->lidarFrame = this->declare_parameter("lidarFrame", "base_link");

    config->mappingCornerLeafSize = this->declare_parameter("mappingCornerLeafSize", 0.2);
    config->mappingSurfLeafSize = this->declare_parameter("mappingSurfLeafSize", 0.2);
    config->mappingProcessInterval = this->declare_parameter("mappingProcessInterval", 0.15);

    config->N_SCAN       = static_cast<uint32_t>(this->declare_parameter<int>("N_SCAN",        64));
    config->Horizon_SCAN = static_cast<uint32_t>(this->declare_parameter<int>("Horizon_SCAN", 4096));

    config->globalMapVisualizationSearchRadius = this->declare_parameter("globalMapVisualizationSearchRadius", 1e3);
    config->globalMapVisualizationPoseDensity = this->declare_parameter("globalMapVisualizationPoseDensity", 10.0);
    config->globalMapVisualizationLeafSize = this->declare_parameter("globalMapVisualizationLeafSize", 1.0);

    config->surroundingKeyframeAddingDistThreshold = this->declare_parameter("surroundingKeyframeAddingDistThreshold", 1.0);
    config->surroundingKeyframeAddingAngleThreshold  = this->declare_parameter("surroundingKeyframeAddingAngleThreshold", 0.2);
    config->surroundingKeyframeDensity = this->declare_parameter("surroundingKeyframeDensity", 1.0);
    config->surroundingKeyframeSearchRadius = this->declare_parameter("surroundingKeyframeSearchRadius", 50.0);

    // TODO: Pay attention to these flags
    config->loopClosureEnableFlag = this->declare_parameter("loopClosureEnableFlag", true);
    config->loopClosureFrequency = this->declare_parameter("loopClosureFrequency", 1.0);
    config->surroundingKeyframeSize = this->declare_parameter("surroundingKeyframeSize", 50);
    config->historyKeyframeSearchRadius = this->declare_parameter("historyKeyframeSearchRadius", 10.0);
    config->historyKeyframeSearchTimeDiff = this->declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
    config->historyKeyframeSearchNum = this->declare_parameter("historyKeyframeSearchNum", 25);
    config->historyKeyframeFitnessScore = this->declare_parameter("historyKeyframeFitnessScore", 0.3);

    // TODO: Experiment with this
    config->useImuHeadingInitialization = this->declare_parameter("useImuHeadingInitialization", false);
    config->useGpsElevation = this->declare_parameter("useGpsElevation", false);
    config->gpsCovThreshold = this->declare_parameter("gpsCovThreshold", 2.0);
    config->poseCovThreshold = this->declare_parameter("poseCovThreshold", 25.0);

    config->edgeFeatureMinValidNum = this->declare_parameter("edgeFeatureMinValidNum", 10);
    config->surfFeatureMinValidNum = this->declare_parameter("surfFeatureMinValidNum", 100);

    config->imuRPYWeight = this->declare_parameter("imuRPYWeight", 0.01);
    config->z_tolerance = this->declare_parameter("z_tolerance", std::numeric_limits<float>::max());
    config->rotation_tolerance = this->declare_parameter("rotation_tolerance", std::numeric_limits<float>::max());
}

void MapOptimizationNode::initialize() {
    gtsam::ISAM2Params isam2_params;
    isam2_params.relinearizeThreshold = 0.1;
    isam2_params.relinearizeSkip = 1;
    isam2 = new gtsam::ISAM2(isam2_params);

    downsizeFilterCorner.setLeafSize(config->mappingCornerLeafSize, config->mappingCornerLeafSize, config->mappingCornerLeafSize);
    downsizeFilterSurf.setLeafSize(config->mappingSurfLeafSize, config->mappingSurfLeafSize, config->mappingSurfLeafSize);
    downsizeFilterICP.setLeafSize(config->mappingSurfLeafSize, config->mappingSurfLeafSize, config->mappingSurfLeafSize);
    downsizeFilterSurroundingKeyPoses.setLeafSize(config->surroundingKeyframeDensity, config->surroundingKeyframeDensity, config->surroundingKeyframeDensity);

    lastPoseAffineAvailable = false;
    lastPose6D = ZeroPointTypePose;

    allocate_memory();
}

void MapOptimizationNode::allocate_memory() {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());  // corner feature set from odomOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());  // surf feature set from odomOptimization
    laserCloudCornerLastDs.reset(new pcl::PointCloud<PointType>());  // downsampled corner feature set from odomOptimization
    laserCloudSurfLastDs.reset(new pcl::PointCloud<PointType>());  // downsampled surf feature set from odomOptimization

    laserCloudOriginal.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    auto size_ = config->N_SCAN * config->Horizon_SCAN;
    laserCloudOriginalCornerVec.resize(size_);
    coeffSelCornerVec.resize(size_);
    laserCloudOriginalCornerFlag.resize(size_);
    laserCloudOriginalSurfVec.resize(size_);
    coeffSelSurfVec.resize(size_);
    laserCloudOriginalSurfFlag.resize(size_);

    std::fill(laserCloudOriginalCornerFlag.begin(), laserCloudOriginalCornerFlag.end(), false);
    std::fill(laserCloudOriginalSurfFlag.begin(), laserCloudOriginalSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDs.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDs.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i) {
        transformTobeMapped[i] = 0;
    }

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
}

void MapOptimizationNode::setup_subpub() {
    pubKeyPoses = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
    pubLaserCloudSurround = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
    pubLaserOdometryGlobal = this->create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", 1);
    pubLaserOdometryIncremental = this->create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry_incremental", 1);
    pubPath = this->create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);

    subCloud = this->create_subscription<slam_msgs::msg::CloudInfo>("lio_sam/feature/cloud_info", 1, std::bind(&MapOptimizationNode::laserCloudInfoHandler, this, std::placeholders::_1));
    subGPS = this->create_subscription<nav_msgs::msg::Odometry>(config->gpsTopic, 200, std::bind(&MapOptimizationNode::gpsHandler, this, std::placeholders::_1));
    subLoop = this->create_subscription<std_msgs::msg::Float64MultiArray>("lio_loop/loop_closure_detection", 1, std::bind(&MapOptimizationNode::loopInfoHandler, this, std::placeholders::_1));

    srvSaveMap = this->create_service<slam_srvs::srv::SaveMap>("lio_sam/save_map",
        [this](const std::shared_ptr<slam_srvs::srv::SaveMap::Request> req, std::shared_ptr<slam_srvs::srv::SaveMap::Response> res) {
            saveMapService(*req, *res);
        });

    tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    pubHistoryKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    pubLoopConstraintEdge = this->create_publisher<visualization_msgs::msg::MarkerArray>("lio_sam/mapping/loop_closure_constraints", 1);

    pubRecentKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
    pubRecentKeyFrame = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

    pubSLAMInfo = this->create_publisher<slam_msgs::msg::CloudInfo>("lio_sam/mapping/slam_info", 1);
}

void MapOptimizationNode::laserCloudInfoHandler(const slam_msgs::msg::CloudInfo::ConstSharedPtr& msg) {
    // extract time stamp
    timeLaserInfoStamp = msg->header.stamp;
    timeLaserInfoCur = ROS_TIME(*msg);

    // extract info and feature cloud
    cloudInfo = *msg;
    pcl::fromROSMsg(msg->cloud_corner, *laserCloudCornerLast);
    pcl::fromROSMsg(msg->cloud_surface, *laserCloudSurfLast);

    std::lock_guard lock(mtx);

    static double timeLastProcessing = -1;
    if (timeLaserInfoCur - timeLastProcessing >= config->mappingProcessInterval) {
        RCLCPP_INFO(this->get_logger(), "Processing extracted feature cloud at time '%.9f'", timeLaserInfoCur);
        timeLastProcessing = timeLaserInfoCur;

        // Per-function runtime profiling. Times each stage of the mapping pipeline so we can see
        // which stage dominates the per-frame cost (scan-to-map registration on dense 64-beam
        // clouds is the usual suspect). Logs per-stage and total wall time in milliseconds.
        using profile_clock = std::chrono::steady_clock;
        const auto profile_start = profile_clock::now();
        auto profileStage = [&](const char* name, auto&& fn) {
            const auto t0 = profile_clock::now();
            fn();
            const double ms = std::chrono::duration<double, std::milli>(profile_clock::now() - t0).count();
            RCLCPP_INFO(this->get_logger(), "  [profile] %-26s %8.3f ms", name, ms);
        };

        profileStage("updateInitialGuess",        [&] { updateInitialGuess(); });
        profileStage("extractSurroundingKeyFrames", [&] { extractSurroundingKeyFrames(); });
        profileStage("downsampleCurrentScan",      [&] { downsampleCurrentScan(); });
        profileStage("scan2MapOptimization",       [&] { scan2MapOptimization(); });
        profileStage("saveKeyFramesAndFactor",     [&] { saveKeyFramesAndFactor(); });
        profileStage("correctPoses",               [&] { correctPoses(); });
        profileStage("publishOdometry",            [&] { publishOdometry(); });
        profileStage("publishFrames",              [&] { publishFrames(); });

        const double total_ms = std::chrono::duration<double, std::milli>(profile_clock::now() - profile_start).count();
        RCLCPP_INFO(this->get_logger(), "  [profile] %-26s %8.3f ms", "TOTAL laserCloudInfoHandler", total_ms);
    }

    aLoopIsClosed = false;
}

void MapOptimizationNode::gpsHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    gpsQueue.push_back(*msg);
}

void MapOptimizationNode::pointAssociateToMap(PointType const* const pt_in, PointType* const pt_out) {
    pt_out->x = transPointAssociateToMap(0, 0) * pt_in->x + transPointAssociateToMap(0, 1) * pt_in->y + transPointAssociateToMap(0, 2) * pt_in->z + transPointAssociateToMap(0, 3);
    pt_out->y = transPointAssociateToMap(1, 0) * pt_in->x + transPointAssociateToMap(1, 1) * pt_in->y + transPointAssociateToMap(1, 2) * pt_in->z + transPointAssociateToMap(1, 3);
    pt_out->z = transPointAssociateToMap(2, 0) * pt_in->x + transPointAssociateToMap(2, 1) * pt_in->y + transPointAssociateToMap(2, 2) * pt_in->z + transPointAssociateToMap(2, 3);
    pt_out->intensity = pt_in->intensity;
}

bool MapOptimizationNode::saveMapService(slam_srvs::srv::SaveMap::Request& req, slam_srvs::srv::SaveMap::Response& res) {
    std::string saveMapDirectory;

    RCLCPP_INFO(this->get_logger(), "****************************************************");
    RCLCPP_INFO(this->get_logger(), "Saving map to pcd files...");
    const char* homeEnv = std::getenv("HOME");
    if (homeEnv == nullptr) {
        RCLCPP_ERROR(this->get_logger(), "HOME environment variable not set, cannot resolve save destination");
        res.success = false;
        return false;
    }
    if (req.destination.empty()) {
        saveMapDirectory = homeEnv + config->savePCDDirectory;
    } else {
        saveMapDirectory = homeEnv + req.destination;
    }
    RCLCPP_INFO(this->get_logger(), "Save destination: %s", saveMapDirectory.c_str());
    // recreate the directory, dropping any previously saved map
    std::error_code ec;
    std::filesystem::remove_all(saveMapDirectory, ec);
    if (!std::filesystem::create_directories(saveMapDirectory, ec) && ec) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create save directory: %s", ec.message().c_str());
        res.success = false;
        return false;
    }
    // save key frame transformations
    pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDs(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDs(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < static_cast<int>(cloudKeyPoses3D->size()); ++i) {
        *globalCornerCloud += *transformPointCloud<PointType, PointTypePose>(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i], config->numberOfCores);
        *globalSurfCloud += *transformPointCloud<PointType, PointTypePose>(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i], config->numberOfCores);
        RCLCPP_INFO(this->get_logger(), "Processing feature cloud %d of %zu ...", i, cloudKeyPoses6D->size());
    }

    if (req.resolution != 0) {
        RCLCPP_INFO(this->get_logger(), R"(Save resolution: %f)", req.resolution);

        // downsample and save corner cloud
        downsizeFilterCorner.setInputCloud(globalCornerCloud);
        downsizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downsizeFilterCorner.filter(*globalCornerCloudDs);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDs);
        // downsample and save surf cloud
        downsizeFilterSurf.setInputCloud(globalSurfCloud);
        downsizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downsizeFilterSurf.filter(*globalSurfCloudDs);
        pcl::io::savePCDFile(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDs);
    } else {
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
    }

    // save global point cloud map
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;

    const int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
    res.success = ret == 0;

    downsizeFilterCorner.setLeafSize(config->mappingCornerLeafSize, config->mappingCornerLeafSize, config->mappingCornerLeafSize);
    downsizeFilterSurf.setLeafSize(config->mappingSurfLeafSize, config->mappingSurfLeafSize, config->mappingSurfLeafSize);

    RCLCPP_INFO(this->get_logger(), "****************************************************");
    RCLCPP_INFO(this->get_logger(), "Saving map to pcd files completed");

    return res.success;
}

void MapOptimizationNode::visualizeGlobalMapThread() {
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok()) {
        rate.sleep();
        publishGlobalMap();
    }

    if (!config->savePCD)
        return;

    slam_srvs::srv::SaveMap::Request req;
    slam_srvs::srv::SaveMap::Response res;

    if (!saveMapService(req, res)) {
        RCLCPP_WARN(this->get_logger(), "Failed to save map");
    }
}

void MapOptimizationNode::publishGlobalMap() {
    // if (pubLaserCloudSurround->get_subscription_count() == 0)
    //     return;

    if (cloudKeyPoses3D->points.empty())
        return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDs(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDs(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    // TODO: Could possibly speed this up: https://stackoverflow.com/questions/79099317/pclkdtreeflann-is-extremely-slow
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), config->globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
    mtx.unlock();

    for (int i = 0; i < static_cast<int>(pointSearchIndGlobalMap.size()); i++) {
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    }
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downsizeFilterGlobalMapKeyPoses;  // for global map visualization
    downsizeFilterGlobalMapKeyPoses.setLeafSize(config->globalMapVisualizationPoseDensity, config->globalMapVisualizationPoseDensity, config->globalMapVisualizationPoseDensity);
    downsizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downsizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDs);
    for (auto& pt : globalMapKeyPosesDs->points) {
        kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
        pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < static_cast<int>(globalMapKeyPosesDs->size()); ++i) {
        if (pointDistance(globalMapKeyPosesDs->points[i], cloudKeyPoses3D->back()) > config->globalMapVisualizationSearchRadius)
            continue;
        // thisKeyInd is a key indicating which pose we will use to transform the points at that key
        int thisKeyInd = static_cast<int>(globalMapKeyPosesDs->points[i].intensity);
        *globalMapKeyFrames += *transformPointCloud<PointType, PointTypePose>(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd], config->numberOfCores);
        *globalMapKeyFrames += *transformPointCloud<PointType, PointTypePose>(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd], config->numberOfCores);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downsizeFilterGlobalMapKeyFrames;  // for global map visualization
    downsizeFilterGlobalMapKeyFrames.setLeafSize(config->globalMapVisualizationLeafSize, config->globalMapVisualizationLeafSize, config->globalMapVisualizationLeafSize);
    downsizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downsizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDs);
    publishCloud<PointType>(pubLaserCloudSurround, globalMapKeyFramesDs, timeLaserInfoStamp, config->odometryFrame);
}

void MapOptimizationNode::loopClosureThread() {
    if (!config->loopClosureEnableFlag)
        return;

    rclcpp::Rate rate(config->loopClosureFrequency);
    while (rclcpp::ok()) {
        rate.sleep();
        performLoopClosure();
        visualizeLoopClosure();
    }
}

void MapOptimizationNode::loopInfoHandler(const std_msgs::msg::Float64MultiArray::ConstSharedPtr& msg) {
    std::lock_guard lock(mtxLoopInfo);
    if (msg->data.size() != 2)
        return;

    loopInfoVec.push_back(*msg);

    while (loopInfoVec.size() > 5)
        loopInfoVec.pop_front();
}

void MapOptimizationNode::performLoopClosure() {
    if (cloudKeyPoses3D->points.empty())
        return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    // Snapshot the keyframe clouds under the same lock so loopFindNearKeyframes() never reads them
    // while the subscriber thread is reallocating them via push_back.
    copy_cornerCloudKeyFrames = cornerCloudKeyFrames;
    copy_surfCloudKeyFrames = surfCloudKeyFrames;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (!detectLoopClosureExternal(&loopKeyCur, &loopKeyPre)) {
        if (!detectLoopClosureDistance(&loopKeyCur, &loopKeyPre)) {
            RCLCPP_INFO(this->get_logger(), "No loop closure detected...");
            return;
        } else {
            RCLCPP_INFO(this->get_logger(), "Detected Loop Closure Distance!");
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "Detected Loop Closure External!");
    }

    // extract cloud
    pcl::PointCloud<PointType>::Ptr curKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
        // Create a point cloud centered around loopKeyPre with `config->historyKeyframeSearchNum` clouds on either side
        // of it to create a larger point cloud more likely to match well using ICP.
        loopFindNearKeyframes(curKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, config->historyKeyframeSearchNum);
        if (curKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames->get_subscription_count() != 0)
            publishCloud<PointType>(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, config->odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(config->historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon((1e-6));
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(curKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (!icp.hasConverged() || icp.getFitnessScore() > config->historyKeyframeFitnessScore)
        return;

    // publish corrected cloud
    if (pubIcpKeyFrames->get_subscription_count() != 0) {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*curKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud<PointType>(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, config->odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    // world origin -> cur key frame origin -> corrected origin
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;  // pre-multiply -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(std::make_pair(loopKeyCur, loopKeyPre));
    // essentially the error between the transforms:
    //  - poseFrom: pose from world origin -> current key -> looped prev key (solved by ICP)
    //  - poseTo: pose from world origin -> looped prev key
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constraint
    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

bool MapOptimizationNode::detectLoopClosureDistance(int* latestId, int* closestId) {
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), config->historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop);

    for (int i = 0; i < static_cast<int>(pointSearchIndLoop.size()); ++i) {
        int id = pointSearchIndLoop[i];
        // If enough time has passed between points, then keep key
        if (std::abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > config->historyKeyframeSearchTimeDiff) {
            loopKeyPre = id;
            break;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestId = loopKeyCur;
    *closestId = loopKeyPre;

    return true;
}

bool MapOptimizationNode::detectLoopClosureExternal(int* latestId, int* closestId) {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard lock(mtxLoopInfo);
    if (loopInfoVec.empty())
        return false;

    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    if (std::abs(loopTimeCur - loopTimePre) < config->historyKeyframeSearchTimeDiff)
        return false;

    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2)
        return false;

    // latest key
    for (int i = cloudSize - 1; i >= 0; --i) {
        if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur) {
            loopKeyCur = std::round(copy_cloudKeyPoses6D->points[i].intensity);
        } else {
            break;
        }
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i) {
        if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre) {
            loopKeyPre = std::round(copy_cloudKeyPoses6D->points[i].intensity);
        } else {
            break;
        }
    }

    if (loopKeyCur == loopKeyPre)
        return false;

    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    *latestId = loopKeyCur;
    *closestId = loopKeyPre;

    return true;
}

void MapOptimizationNode::loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum) {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;
        // Use the snapshots taken under mtx in performLoopClosure(), not the live vectors.
        *nearKeyframes += *transformPointCloud<PointType>(copy_cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        *nearKeyframes += *transformPointCloud<PointType>(copy_surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframe
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downsizeFilterICP.setInputCloud(nearKeyframes);
    downsizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

void MapOptimizationNode::visualizeLoopClosure() {
    if (loopIndexContainer.empty())
        return;

    visualization_msgs::msg::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::msg::Marker markerNode;
    markerNode.header.frame_id = config->odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::msg::Marker::ADD;
    markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3;
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::msg::Marker markerEdge;
    markerEdge.header.frame_id = config->odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::msg::Marker::ADD;
    markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it!= loopIndexContainer.end(); ++it) {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::msg::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge->publish(markerArray);
}

void MapOptimizationNode::updateInitialGuess() {
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastImuTransformation;
    // initialization
    if (cloudKeyPoses3D->points.empty()) {
        transformTobeMapped[0] = cloudInfo.imu_roll_init;
        transformTobeMapped[1] = cloudInfo.imu_pitch_init;
        transformTobeMapped[2] = cloudInfo.imu_yaw_init;

        if (!config->useImuHeadingInitialization)
            transformTobeMapped[2] = 0;

        lastImuTransformation = getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);  // save imu before return
        return;
    }

    // use imu pre-integration estimation for pose guess
    static bool lastImuPreTransAvailable = false;
    // static bool motionModelAvailable = true;
    static Eigen::Affine3f lastImuPreTransformation;
    if (cloudInfo.odom_available) {
        Eigen::Affine3f transBack = getTransformation(cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
            cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
        if (!lastImuPreTransAvailable) {
            lastImuPreTransformation = transBack;
            lastImuPreTransAvailable = true;
        } else {
            Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuPreTransformation = transBack;

            lastImuTransformation = getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);  // save imu before return
            return;
        }

        // motionModelAvailable = false;
    } else if (lastPoseAffineAvailable) {
    // Constant-velocity motion-model fallback. Reached only when IMU pre-integration odometry is
    // unavailable (e.g. the IMU factor graph has not converged). Previously the guess carried just
    // the IMU rotation with zero translation, so for a fast-moving vehicle scan-to-map registration
    // started meters from the truth and the LM loop failed to converge. Here we assume the
    // inter-frame motion realized in the previous step repeats, giving registration a translation
    // prior. incrementalOdometryAffineFront is this frame's start pose (= the previous frame's
    // optimized pose); lastPoseAffine is the frame-before-that, so their delta is the last realized
    // inter-frame motion, which we replay.
    // if (motionModelAvailable) {
        Eigen::Affine3f motionIncre = lastPoseAffine.inverse() * incrementalOdometryAffineFront;

        // Clamp the replayed increment to physically plausible bounds. The constant-velocity replay
        // otherwise compounds a bad scan-match result into the next initial guess, launching the
        // pose outside the convergence basin (observed: pose/velocity runaway to z=-70m, |v|=57m/s).
        // Bound translation by a max speed and rotation by a max rate over the inter-frame dt.
        {
            constexpr double kMaxLinearSpeed  = 30.0;  // m/s   (~108 km/h, generous for KITTI)
            constexpr double kMaxAngularSpeed = 1.5;   // rad/s (per axis)
            float ix, iy, iz, iroll, ipitch, iyaw;
            pcl::getTranslationAndEulerAngles(motionIncre, ix, iy, iz, iroll, ipitch, iyaw);
            const double dt = (lastMotionTime < 0.0) ? 0.1 : std::max(timeLaserInfoCur - lastMotionTime, 1e-3);
            const float maxTrans = static_cast<float>(kMaxLinearSpeed * dt);
            const float maxRot   = static_cast<float>(kMaxAngularSpeed * dt);
            const float transNorm = std::sqrt(ix * ix + iy * iy + iz * iz);
            if (transNorm > maxTrans && transNorm > 1e-6f) {
                const float scale = maxTrans / transNorm;
                ix *= scale; iy *= scale; iz *= scale;
            }
            iroll  = std::clamp(iroll,  -maxRot, maxRot);
            ipitch = std::clamp(ipitch, -maxRot, maxRot);
            iyaw   = std::clamp(iyaw,   -maxRot, maxRot);
            motionIncre = getTransformation(ix, iy, iz, iroll, ipitch, iyaw);
        }

        Eigen::Affine3f transFinal = incrementalOdometryAffineFront * motionIncre;
        float cvX, cvY, cvZ, cvRoll, cvPitch, cvYaw;
        pcl::getTranslationAndEulerAngles(transFinal, cvX, cvY, cvZ, cvRoll, cvPitch, cvYaw);
        // Always adopt the predicted translation.
        transformTobeMapped[3] = cvX;
        transformTobeMapped[4] = cvY;
        transformTobeMapped[5] = cvZ;
        // Adopt the predicted rotation only when no IMU orientation is available; otherwise the
        // imu_available block below supplies a more reliable (gyro-based) rotation increment, and
        // we keep this constant-velocity translation underneath it.
        if (!cloudInfo.imu_available) {
            transformTobeMapped[0] = cvRoll;
            transformTobeMapped[1] = cvPitch;
            transformTobeMapped[2] = cvYaw;
        }
    }
    lastPoseAffine = incrementalOdometryAffineFront;
    lastMotionTime = timeLaserInfoCur;
    lastPoseAffineAvailable = true;
    // motionModelAvailable = true;

    // use imu incremental estimation for pose guess (only rotation)
    if (cloudInfo.imu_available) {
        Eigen::Affine3f transBack = getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
        Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);  // save imu before return
    }
}

void MapOptimizationNode::extractForLoopClosure() {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses-1; i >= 0; --i) {
        if (static_cast<int>(cloudToExtract->size()) <= config->surroundingKeyframeSize)
            cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    extractCloud(cloudToExtract);
}

void MapOptimizationNode::extractNearby() {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDs(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), config->surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < static_cast<int>(pointSearchInd.size()); ++i) {
        int id = pointSearchInd[i];
        surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    downsizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downsizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDs);
    for (auto& pt : surroundingKeyPosesDs->points) {
        kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
        pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    // also extract some latest key frames in case the robot rotates in one position
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses-1; i >= 0; --i) {
        if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0) {
           surroundingKeyPosesDs->push_back(cloudKeyPoses3D->points[i]);
        } else {
            break;
        }
    }

    extractCloud(surroundingKeyPosesDs);
}

void MapOptimizationNode::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    for (int i = 0; i < static_cast<int>(cloudToExtract->size()); ++i) {
        if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > config->surroundingKeyframeSearchRadius)
            continue;

        int thisKeyInd = static_cast<int>(cloudToExtract->points[i].intensity);
        if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
            // transformed cloud available
            *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
            *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
        } else {
            // transformed cloud not available
            pcl::PointCloud<PointType> laserCloudCornerTmp = *transformPointCloud<PointType>(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            pcl::PointCloud<PointType> laserCloudSurfTmp = *transformPointCloud<PointType>(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *laserCloudCornerFromMap += laserCloudCornerTmp;
            *laserCloudSurfFromMap += laserCloudSurfTmp;
            laserCloudMapContainer[thisKeyInd] = std::make_pair(laserCloudCornerTmp, laserCloudSurfTmp);
        }
    }

    // Downsample the surrounding corner key frames (or map)
    downsizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downsizeFilterCorner.filter(*laserCloudCornerFromMapDs);
    laserCloudCornerFromMapDsNum = laserCloudCornerFromMapDs->size();
    // Downsample the surrounding surf key frames (or map)
    downsizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downsizeFilterSurf.filter(*laserCloudSurfFromMapDs);
    laserCloudSurfFromMapDsNum = laserCloudSurfFromMapDs->size();

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
        laserCloudMapContainer.clear();
}

void MapOptimizationNode::extractSurroundingKeyFrames() {
    if (cloudKeyPoses3D->points.empty())
        return;

    extractNearby();
}

void MapOptimizationNode::downsampleCurrentScan() {
    // Downsample cloud from current scan
    laserCloudCornerLastDs->clear();
    downsizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downsizeFilterCorner.filter(*laserCloudCornerLastDs);
    laserCloudCornerLastDsNum = laserCloudCornerLastDs->size();

    laserCloudSurfLastDs->clear();
    downsizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downsizeFilterSurf.filter(*laserCloudSurfLastDs);
    laserCloudSurfLastDsNum = laserCloudSurfLastDs->size();
}

void MapOptimizationNode::updatePointAssociateToMap() {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

void MapOptimizationNode::cornerOptimization() {
    updatePointAssociateToMap();

    #pragma omp parallel for num_threads(config->numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDsNum; ++i) {
        // For every point in the downsampled corner point cloud, search for the 5 nearest neighbors, and if they are
        // all within a certain squared distance then find the mean corner and error of this new corner point.
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudCornerLastDs->points[i];
        // pointSel are now points in the global map, while pointOri are in the local map
        pointAssociateToMap(&pointOri, &pointSel);
        int numNeighbors = kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

        // TODO: Need to update this with 'lidarCurvatureFeatureExtractionNeighbors'
        if (numNeighbors == 5 && pointSearchSqDis[4] < 1.0) {
            // mean corner point
            float cx = 0, cy = 0, cz = 0;
            for (int j = 0; j < 5; j++) {
                cx += laserCloudCornerFromMapDs->points[pointSearchInd[j]].x;
                cy += laserCloudCornerFromMapDs->points[pointSearchInd[j]].y;
                cz += laserCloudCornerFromMapDs->points[pointSearchInd[j]].z;
            }
            cx /= 5; cy /= 5; cz /= 5;

            // mean l2-norm error
            float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
            for (int j = 0; j < 5; j++) {
                float ax = laserCloudCornerFromMapDs->points[pointSearchInd[j]].x - cx;
                float ay = laserCloudCornerFromMapDs->points[pointSearchInd[j]].y - cy;
                float az = laserCloudCornerFromMapDs->points[pointSearchInd[j]].z - cz;

                a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                a22 += ay * ay; a23 += ay * az;
                a33 += az * az;
            }
            a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

            // covariance error mat
            matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
            matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
            matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

            // calculate eigenvalues/eigenvectors
            // find in which direction the corner varies the most
            cv::eigen(matA1, matD1, matV1);
            // only care about variation in x,y direction
            if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                // original point in the world coordinates
                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                // move the averaged points up and down on the eigenvectors corresponding to axis-0.
                float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                // essentially interpolating the new corner point using precise values from the eigenvectors
                float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                          + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                           - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                           + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                // make sure there is sufficient variation in the direction of the 0th eigenvector
                float ld2 = a012 / l12;

                float s = 1 - 0.9 * std::fabs(ld2);

                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                coeff.intensity = s * ld2;

                if (s > 0.1) {
                    laserCloudOriginalCornerVec[i] = pointOri;
                    coeffSelCornerVec[i] = coeff;
                    laserCloudOriginalCornerFlag[i] = true;
                }
            }
        }
    }
}

void MapOptimizationNode::surfOptimization() {
    updatePointAssociateToMap();

    #pragma omp parallel for num_threads(config->numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDsNum; ++i) {
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudSurfLastDs->points[i];
        pointAssociateToMap(&pointOri, &pointSel);
        int numNeighbors = kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<float, 5, 3> matA0;
        Eigen::Matrix<float, 5, 1> matB0;
        Eigen::Vector3f matX0;

        matA0.setZero();
        matB0.fill(-1);
        matX0.setZero();

        if (numNeighbors == 5 && pointSearchSqDis[4] < 1.0) {
            for (int j = 0; j < 5; ++j) {
                matA0(j, 0) = laserCloudSurfFromMapDs->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfFromMapDs->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfFromMapDs->points[pointSearchInd[j]].z;
            }

            matX0 = matA0.colPivHouseholderQr().solve(matB0);

            float pa = matX0(0, 0);
            float pb = matX0(1, 0);
            float pc = matX0(2, 0);
            float pd = 1;

            float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps; pb /= ps; pc /= ps; pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < 5; ++j) {
                if (std::fabs(pa * laserCloudSurfFromMapDs->points[pointSearchInd[j]].x +
                    pb * laserCloudSurfFromMapDs->points[pointSearchInd[j]].y +
                    pc * laserCloudSurfFromMapDs->points[pointSearchInd[j]].z + pd) > 0.2) {
                    planeValid = false;
                    break;
                }
            }

            if (planeValid) {
                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1 - 0.9 * std::fabs(pd2) / std::sqrt(std::sqrt(pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;

                if (s > 0.1) {
                    laserCloudOriginalSurfVec[i] = pointOri;
                    coeffSelSurfVec[i] = coeff;
                    laserCloudOriginalSurfFlag[i] = true;
                }
            }
        }
    }
}

void MapOptimizationNode::combineOptimizationCoeffs() {
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDsNum; ++i) {
        if (laserCloudOriginalCornerFlag[i]) {
            laserCloudOriginal->push_back(laserCloudOriginalCornerVec[i]);
            coeffSel->push_back(coeffSelCornerVec[i]);
        }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDsNum; ++i) {
        if (laserCloudOriginalSurfFlag[i]) {
            laserCloudOriginal->push_back(laserCloudOriginalSurfVec[i]);
            coeffSel->push_back(coeffSelSurfVec[i]);
        }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriginalCornerFlag.begin(), laserCloudOriginalCornerFlag.end(), false);
    std::fill(laserCloudOriginalSurfFlag.begin(), laserCloudOriginalSurfFlag.end(), false);
}

bool MapOptimizationNode::LMOptimization(int iterCount)
{
	// This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
	// lidar <- camera      ---     camera <- lidar
	// x = z                ---     x = y
	// y = x                ---     y = z
	// z = y                ---     z = x
	// roll = yaw           ---     roll = pitch
	// pitch = roll         ---     pitch = yaw
	// yaw = pitch          ---     yaw = roll

	// lidar -> camera
	float srx = sin(transformTobeMapped[1]);
	float crx = cos(transformTobeMapped[1]);
	float sry = sin(transformTobeMapped[2]);
	float cry = cos(transformTobeMapped[2]);
	float srz = sin(transformTobeMapped[0]);
	float crz = cos(transformTobeMapped[0]);

	int laserCloudSelNum = laserCloudOriginal->size();
	if (laserCloudSelNum < 50) {
		return false;
	}

	cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
	cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
	cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
	cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
	cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
	cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

	PointType pointOri, coeff;

	for (int i = 0; i < laserCloudSelNum; i++) {
		// lidar -> camera
		pointOri.x = laserCloudOriginal->points[i].y;
		pointOri.y = laserCloudOriginal->points[i].z;
		pointOri.z = laserCloudOriginal->points[i].x;
		// lidar -> camera
		coeff.x = coeffSel->points[i].y;
		coeff.y = coeffSel->points[i].z;
		coeff.z = coeffSel->points[i].x;
		coeff.intensity = coeffSel->points[i].intensity;
		// in camera
		float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
				  + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
				  + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

		float ary = ((cry*srx*srz - crz*sry)*pointOri.x
				  + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
				  + ((-cry*crz - srx*sry*srz)*pointOri.x
				  + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

		float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
				  + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
				  + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
		// camera -> lidar
		matA.at<float>(i, 0) = arz;
		matA.at<float>(i, 1) = arx;
		matA.at<float>(i, 2) = ary;
		matA.at<float>(i, 3) = coeff.z;
		matA.at<float>(i, 4) = coeff.x;
		matA.at<float>(i, 5) = coeff.y;
		matB.at<float>(i, 0) = -coeff.intensity;
	}

	cv::transpose(matA, matAt);
	matAtA = matAt * matA;
	matAtB = matAt * matB;

	// Levenberg damping (λ·I form: an absolute addition to every diagonal). The eigenvalues of AᵀA
	// span ~1e6 down to ~1e2; the oscillation that kept plain Gauss-Newton from converging lives in
	// the weak directions (z, roll/pitch; eig ~5e2). Marquardt damping (λ·diag) scales by each DoF's
	// own curvature and so barely touches those weak directions — it was ineffective. A constant λ
	// sized to the weak eigenvalues damps them meaningfully (eig 5e2 + λ 3e2 ≈ halves the step) while
	// being negligible for the strong directions (eig 1e6 + 3e2 ≈ unchanged). Applied to a copy so
	// the eigenvalue/degeneracy diagnostics below still see the true (undamped) AᵀA.
	constexpr float kLMLambda = 300.0f;
	cv::Mat matAtADamped = matAtA.clone();
	for (int d = 0; d < 6; ++d)
		matAtADamped.at<float>(d, d) += kLMLambda;
	cv::solve(matAtADamped, matAtB, matX, cv::DECOMP_QR);

	if (iterCount == 0) {

		cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

		cv::eigen(matAtA, matE, matV);
		matV.copyTo(matV2);

		isDegenerate = false;
		float eignThre[6] = {100, 100, 100, 100, 100, 100};
		for (int i = 5; i >= 0; i--) {
			if (matE.at<float>(0, i) < eignThre[i]) {
				for (int j = 0; j < 6; j++) {
					matV2.at<float>(i, j) = 0;
				}
				isDegenerate = true;
			} else {
				break;
			}
		}
		matP = matV.inv() * matV2;
	}

	if (isDegenerate)
	{
		cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
		matX.copyTo(matX2);
		matX = matP * matX2;
	}

	transformTobeMapped[0] += matX.at<float>(0, 0);
	transformTobeMapped[1] += matX.at<float>(1, 0);
	transformTobeMapped[2] += matX.at<float>(2, 0);
	transformTobeMapped[3] += matX.at<float>(3, 0);
	transformTobeMapped[4] += matX.at<float>(4, 0);
	transformTobeMapped[5] += matX.at<float>(5, 0);

	float deltaR = std::sqrt(
						std::pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
						std::pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
						std::pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
	float deltaT = std::sqrt(
						std::pow(matX.at<float>(3, 0) * 100, 2) +
						std::pow(matX.at<float>(4, 0) * 100, 2) +
						std::pow(matX.at<float>(5, 0) * 100, 2));

	if (deltaR < 0.05 && deltaT < 0.05) {
		return true; // converged
	}
	return false; // keep optimizing
}

void MapOptimizationNode::scan2MapOptimization() {
    if (cloudKeyPoses3D->points.empty())
        return;

    if (laserCloudCornerLastDsNum > config->edgeFeatureMinValidNum && laserCloudSurfLastDsNum > config->surfFeatureMinValidNum) {
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDs);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDs);

        // Two interchangeable back-ends: the original hand-rolled Gauss-Newton/Levenberg solver, or
        // a GTSAM LevenbergMarquardtOptimizer over a single Pose3. Both re-associate correspondences
        // each iteration (cornerOptimization/surfOptimization) and write the result into
        // transformTobeMapped; select via the useGtsamScanMatcher config flag for A/B comparison.
        if (config->useGtsamScanMatcher)
            scan2MapOptimizationGtsam();
        else
            scan2MapOptimizationCustom();

        transformUpdate();
    } else {
        RCLCPP_WARN(this->get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDsNum, laserCloudSurfLastDsNum);
    }
}

void MapOptimizationNode::scan2MapOptimizationCustom() {
    // Guardrail against worst-case frames: a healthy registration converges (LMOptimization
    // returns true) in well under 10 iterations. When it does not converge — e.g. a poor
    // initial guess — the loop used to grind through all 30 iterations, producing multi-second
    // frames. 15 caps that cost while leaving ample room for normal convergence.
    constexpr int kMaxLMIterations = 15;
    for (int iterCount = 0; iterCount < kMaxLMIterations; iterCount++) {
        laserCloudOriginal->clear();
        coeffSel->clear();
        cornerOptimization();
        surfOptimization();
        combineOptimizationCoeffs();
        if (LMOptimization(iterCount))
            break;
    }
}

void MapOptimizationNode::scan2MapOptimizationGtsam() {
    using gtsam::symbol_shorthand::X;

    auto toPose3 = [&] {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                            gtsam::Point3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    };

    // Outer ICP loop: re-find correspondences at the latest pose, then let GTSAM's LM refine the
    // single Pose3 against that (fixed) correspondence set. GTSAM owns the damping/relinearization
    // that the custom path approximated with a fixed λ·I.
    constexpr int kMaxOuterIters = 15;
    for (int outer = 0; outer < kMaxOuterIters; ++outer) {
        laserCloudOriginal->clear();
        coeffSel->clear();
        cornerOptimization();
        surfOptimization();
        combineOptimizationCoeffs();

        const int nCorr = static_cast<int>(laserCloudOriginal->size());
        if (nCorr < 50)
            break;

        const gtsam::Pose3 Tcur = toPose3();
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial;
        initial.insert(X(0), Tcur);

        for (int i = 0; i < nCorr; ++i) {
            const auto& po = laserCloudOriginal->points[i];   // body-frame point
            const auto& cf = coeffSel->points[i];             // s·normal, s·signedDist

            const gtsam::Vector3 sn(cf.x, cf.y, cf.z);
            const double s = sn.norm();                       // per-point weight
            if (s < 1e-6)
                continue;
            const gtsam::Vector3 nrm = sn / s;                // unit normal/direction
            const double signedDist = cf.intensity / s;       // residual at Tcur

            const gtsam::Point3 p(po.x, po.y, po.z);
            const double d = signedDist - nrm.dot(Tcur.transformFrom(p));  // nᵀ(Tcur·p)+d == signedDist

            // sigma 1/s reproduces the original per-point weighting; Huber adds outlier robustness.
            const auto base = gtsam::noiseModel::Isotropic::Sigma(1, 1.0 / s);
            const auto robust = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1), base);
            graph.emplace_shared<PointPlaneFactor>(X(0), p, nrm, d, robust);
        }

        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(10);
        params.setRelativeErrorTol(1e-4);
        const gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial, params).optimize();
        const gtsam::Pose3 Topt = result.at<gtsam::Pose3>(X(0));

        const gtsam::Vector3 rpy = Topt.rotation().rpy();
        transformTobeMapped[0] = rpy.x();
        transformTobeMapped[1] = rpy.y();
        transformTobeMapped[2] = rpy.z();
        transformTobeMapped[3] = Topt.translation().x();
        transformTobeMapped[4] = Topt.translation().y();
        transformTobeMapped[5] = Topt.translation().z();

        // Outer-loop convergence: pose change between re-associations.
        const gtsam::Pose3 delta = Tcur.between(Topt);
        if (delta.rotation().rpy().norm() < 5e-4 && delta.translation().norm() < 5e-4)
            break;
    }
}

void MapOptimizationNode::transformUpdate() {
    if (cloudInfo.imu_available) {
        if (std::abs(cloudInfo.imu_pitch_init) < 1.4) {
            double imuWeight = config->imuRPYWeight;
            tf2::Quaternion imuQuaternion;
            tf2::Quaternion transformQuaternion;
            double rollMid, pitchMid, yawMid;

            // slerp roll
            transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
            imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
            tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[0] = rollMid;

            // slerp pitch
            transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
            imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
            tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[1] = pitchMid;
        }
    }

    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], config->rotation_tolerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], config->rotation_tolerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], config->z_tolerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
}

bool MapOptimizationNode::saveFrame() {
    if (cloudKeyPoses3D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween ,x, y, z, roll, pitch, yaw);

    if (std::abs(roll) < config->surroundingKeyframeAddingAngleThreshold &&
    std::abs(pitch) < config->surroundingKeyframeAddingAngleThreshold &&
    std::abs(yaw) < config->surroundingKeyframeAddingAngleThreshold &&
    std::sqrt(x*x + y*y + z*z) < config->surroundingKeyframeAddingDistThreshold) {
        return false;
    }

    return true;
}

void MapOptimizationNode::addOdomFactor() {
    if (cloudKeyPoses3D->points.empty()) {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished());
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    } else {
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

void MapOptimizationNode::addGPSFactor() {
    if (gpsQueue.empty())
        return;

    // wait for system intialized and settles down
    if (cloudKeyPoses3D->points.empty())
        return;
    if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
        return;

    // pose covariance small, no need to correct
    if (poseCovariance(3, 3) < config->poseCovThreshold && poseCovariance(4, 4) < config->poseCovThreshold)
        return;

    RCLCPP_INFO(this->get_logger(), "poseCovariance is large! Got '%.4f' & '%.4f'.", poseCovariance(3, 3), poseCovariance(4, 4));
    // last gps position
    static PointType lastGPSPoint;

    int gpsCnt = 0;
    while (!gpsQueue.empty()) {
        gpsCnt++;
        if (ROS_TIME(gpsQueue.front()) < timeLaserInfoCur - 0.2) {
            // message too old
            gpsQueue.pop_front();
        } else if (ROS_TIME(gpsQueue.front()) > timeLaserInfoCur + 0.2) {
            // message too new
            break;
        } else {
            nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
            gpsQueue.pop_front();

            // GPS too noisy, skip
            float noise_x = thisGPS.pose.covariance[0];
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];
            if (noise_x > config->gpsCovThreshold || noise_y > config->gpsCovThreshold)
                continue;

            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!config->useGpsElevation) {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // GPS not properly initialized (0, 0, 0
            if (std::abs(gps_x) < 1e-6 && std::abs(gps_y) < 1e-6)
                continue;

            // Add GPS every few meters
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                continue;

            lastGPSPoint = curGPSPoint;

            gtsam::Vector Vector3(3);
            Vector3 << std::max(noise_x, 1.0f), std::max(noise_y, 1.0f), std::max(noise_z, 1.0f);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            graph.add(gps_factor);

            aLoopIsClosed = true;

            RCLCPP_INFO(this->get_logger(), "Added Loop Closure GPS Factor at (%.5f, %.5f, %.5f)", gps_x, gps_y, gps_z);
            break;
        }
    }
    RCLCPP_INFO(this->get_logger(), "'%d' GPS messages iterated over.", gpsCnt);
}

void MapOptimizationNode::addLoopFactor() {
    if (loopIndexQueue.empty())
        return;

    for (int i = 0; i < static_cast<int>(loopIndexQueue.size()); ++i) {
        int indexFrom = loopIndexQueue[i].first;
        int indexTo = loopIndexQueue[i].second;
        RCLCPP_INFO(this->get_logger(), "Adding a loop constraint between indices: from '%d' to '%d'", indexFrom, indexTo);
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

void MapOptimizationNode::saveKeyFramesAndFactor() {
    if (!saveFrame())
        return;

    // odom factor
    addOdomFactor();

    // gps factor
    addGPSFactor();

    // loop factor
    addLoopFactor();

    // update iSAM
    isam2->update(graph, initialEstimate);
    isam2->update();

    if (aLoopIsClosed) {
        RCLCPP_INFO(this->get_logger(), "A loop is closed! Updating iSAM2 Bayes Net '5' times.");
        for (int i = 0; i < 5; ++i) {
            isam2->update();
        }
    }

    graph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam2->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size();  // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    poseCovariance = isam2->marginalCovariance(isamCurrentEstimate.size()-1);

    auto del_roll = thisPose6D.roll - lastPose6D.roll;
    auto del_pitch = thisPose6D.pitch - lastPose6D.pitch;
    auto del_yaw = thisPose6D.yaw - lastPose6D.yaw;
    auto del_x = thisPose6D.x - lastPose6D.x;
    auto del_y = thisPose6D.y - lastPose6D.y;
    auto del_z = thisPose6D.z - lastPose6D.z;
    RCLCPP_INFO(
        this->get_logger(),
        "Latest 6D Pose Update:\n\troll: %.5f + %.5f = %.5f\n\tpitch: %.5f + %.5f = %.5f\n\tyaw: %.5f + %.5f = %.5f\n\t\t(%.5f, %.5f, %.5f)\n\tx: %.5f + %.5f = %.5f\n\ty: %.5f + %.5f = %.5f\n\tz: %.5f + %.5f = %.5f\n\t\t(%.5f, %.5f, %.5f)",
        lastPose6D.roll, del_roll, thisPose6D.roll,
        lastPose6D.pitch, del_pitch, thisPose6D.pitch,
        lastPose6D.yaw, del_yaw, thisPose6D.yaw,
        thisPose6D.roll, thisPose6D.pitch, thisPose6D.yaw,
        lastPose6D.x, del_x, thisPose6D.x,
        lastPose6D.y, del_y, thisPose6D.y,
        lastPose6D.z, del_z, thisPose6D.z,
        thisPose6D.x, thisPose6D.y, thisPose6D.z
        );

    lastPose6D = thisPose6D;

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDs, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDs, *thisSurfKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
}

void MapOptimizationNode::correctPoses() {
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed) {
        RCLCPP_INFO(this->get_logger(), "Loop is Closed!");
        // clear map cache
        laserCloudMapContainer.clear();
        // clear path
        globalPath.poses.clear();
        // update key poses
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i) {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);
        }
    }
}

void MapOptimizationNode::updatePath(const PointTypePose& pose_in) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = rclcpp::Time(static_cast<int64_t>(pose_in.time * 1e9));
    pose_stamped.header.frame_id = config->odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf2::Quaternion q;
    q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

void MapOptimizationNode::publishOdometry()
{
	// Publish odometry for ROS (global)
	nav_msgs::msg::Odometry laserOdometryROS;
	laserOdometryROS.header.stamp = timeLaserInfoStamp;
	laserOdometryROS.header.frame_id = config->odometryFrame;
	laserOdometryROS.child_frame_id = "odom_mapping";
	laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
	laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
	laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
	tf2::Quaternion quatOdom;
	quatOdom.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
	laserOdometryROS.pose.pose.orientation = tf2::toMsg(quatOdom);
	pubLaserOdometryGlobal->publish(laserOdometryROS);

	// Publish TF
	geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
	trans_odom_to_lidar.header.stamp = timeLaserInfoStamp;
	trans_odom_to_lidar.header.frame_id = config->odometryFrame;
	trans_odom_to_lidar.child_frame_id = "lidar_link";
	trans_odom_to_lidar.transform.translation.x = transformTobeMapped[3];
	trans_odom_to_lidar.transform.translation.y = transformTobeMapped[4];
	trans_odom_to_lidar.transform.translation.z = transformTobeMapped[5];
	trans_odom_to_lidar.transform.rotation = tf2::toMsg(quatOdom);
	tfBroadcaster->sendTransform(trans_odom_to_lidar);

	// Publish odometry for ROS (incremental)
	static bool lastIncreOdomPubFlag = false;
	static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
	static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
	if (lastIncreOdomPubFlag == false)
	{
		lastIncreOdomPubFlag = true;
		laserOdomIncremental = laserOdometryROS;
		increOdomAffine = trans2Affine3f(transformTobeMapped);
	} else {
		Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
		increOdomAffine = increOdomAffine * affineIncre;
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
		if (cloudInfo.imu_available == true)
		{
			if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
			{
				double imuWeight = 0.1;
				tf2::Quaternion imuQuaternion;
				tf2::Quaternion transformQuaternion;
				double rollMid, pitchMid, yawMid;

				// slerp roll
				transformQuaternion.setRPY(roll, 0, 0);
				imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
				tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
				roll = rollMid;

				// slerp pitch
				transformQuaternion.setRPY(0, pitch, 0);
				imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
				tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
				pitch = pitchMid;
			}
		}
		laserOdomIncremental.header.stamp = timeLaserInfoStamp;
		laserOdomIncremental.header.frame_id = config->odometryFrame;
		laserOdomIncremental.child_frame_id = "odom_mapping";
		laserOdomIncremental.pose.pose.position.x = x;
		laserOdomIncremental.pose.pose.position.y = y;
		laserOdomIncremental.pose.pose.position.z = z;
		tf2::Quaternion quatIncremental;
		quatIncremental.setRPY(roll, pitch, yaw);
		laserOdomIncremental.pose.pose.orientation = tf2::toMsg(quatIncremental);
		if (isDegenerate)
			laserOdomIncremental.pose.covariance[0] = 1;
		else
			laserOdomIncremental.pose.covariance[0] = 0;
	}
	pubLaserOdometryIncremental->publish(laserOdomIncremental);
}

void MapOptimizationNode::publishFrames()
{
	if (cloudKeyPoses3D->points.empty())
		return;
	// publish key poses
	publishCloud<PointType>(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, config->odometryFrame);
	// Publish surrounding key frames
    pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
    *localMapOut += *laserCloudCornerFromMapDs;
    *localMapOut += *laserCloudSurfFromMapDs;
    publishCloud<PointType>(pubRecentKeyFrames, localMapOut, timeLaserInfoStamp, config->odometryFrame);
	// publish registered key frame
	if (pubRecentKeyFrame->get_subscription_count() != 0)
	{
		pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
		PointTypePose thisPose6D = trans2PointTypePose<PointTypePose>(transformTobeMapped);
		*cloudOut += *transformPointCloud<PointType>(laserCloudCornerLastDs,  &thisPose6D);
		*cloudOut += *transformPointCloud<PointType>(laserCloudSurfLastDs,    &thisPose6D);
		publishCloud<PointType>(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, config->odometryFrame);
	}
	// publish registered high-res raw cloud
	if (pubCloudRegisteredRaw->get_subscription_count() != 0)
	{
		pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
		pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
		PointTypePose thisPose6D = trans2PointTypePose<PointTypePose>(transformTobeMapped);
		*cloudOut = *transformPointCloud<PointType>(cloudOut,  &thisPose6D);
		publishCloud<PointType>(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, config->odometryFrame);
	}
	// publish path
	if (pubPath->get_subscription_count() != 0)
	{
		globalPath.header.stamp = timeLaserInfoStamp;
		globalPath.header.frame_id = config->odometryFrame;
		pubPath->publish(globalPath);
	}
	// publish SLAM information for 3rd-party usage
	static int lastSLAMInfoPubSize = -1;
	if (pubSLAMInfo->get_subscription_count() != 0)
	{
		if (lastSLAMInfoPubSize != static_cast<int>(cloudKeyPoses6D->size()))
		{
			slam_msgs::msg::CloudInfo slamInfo;
			slamInfo.header.stamp = timeLaserInfoStamp;
			pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
			*cloudOut += *laserCloudCornerLastDs;
			*cloudOut += *laserCloudSurfLastDs;
			slamInfo.key_frame_cloud = cloudToRosMsg<PointType>(cloudOut, timeLaserInfoStamp, config->lidarFrame);
			slamInfo.key_frame_poses = cloudToRosMsg<PointTypePose>(cloudKeyPoses6D, timeLaserInfoStamp, config->odometryFrame);
			slamInfo.key_frame_map = cloudToRosMsg<PointType>(localMapOut, timeLaserInfoStamp, config->odometryFrame);
			pubSLAMInfo->publish(slamInfo);
			lastSLAMInfoPubSize = static_cast<int>(cloudKeyPoses6D->size());
		}
	}
}
