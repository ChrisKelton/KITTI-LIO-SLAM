//
// Created by ckelton on 6/9/26.
//
#include <stdexcept>
#include <string>

#include "ImageProjectionNode.h"


// File-local helpers — not part of the public interface.
namespace {

bool hasField(const sensor_msgs::msg::PointCloud2& msg, const std::string& name) {
    return std::any_of(msg.fields.begin(), msg.fields.end(),
        [&](const auto& f) { return f.name == name; });
}

void validateFields(const sensor_msgs::msg::PointCloud2& msg) {
    std::vector<std::string> required = {"x", "y", "z", "intensity", "ring", "time"};
    for (const auto& field : required) {
        if (!hasField(msg, field)) {
            throw std::runtime_error("Unable to find field " + field);
        }
    }
}

// Only supports Velodyne sensor for right now.
void extractPointCloud(
    const sensor_msgs::msg::PointCloud2& msg,
    const pcl::PointCloud<LidarPoint>::Ptr& cloud
) {
    validateFields(msg);

    // Clear stale points: this cloud is only emptied in resetParameters(), which runs after a
    // *successful* pipeline pass. If deskewInfo() bails (e.g. waiting for IMU), the next scan would
    // otherwise be appended onto the previous one, corrupting ranges and the range image.
    cloud->clear();

    cloud->reserve(msg.width * msg.height);
    cloud->header.frame_id = msg.header.frame_id;
    cloud->height = 1;  // unorganized until ring-projection is done

    sensor_msgs::PointCloud2ConstIterator<float>    iter_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float>    iter_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float>    iter_z(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float>    iter_intensity(msg, "intensity");
    sensor_msgs::PointCloud2ConstIterator<uint16_t> iter_ring(msg, "ring");
    sensor_msgs::PointCloud2ConstIterator<float>    iter_time(msg, "time");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_ring, ++iter_time) {
        LidarPoint p;
        p.x = *iter_x; p.y = *iter_y; p.z = *iter_z;
        p.intensity = *iter_intensity;
        p.ring = *iter_ring;
        p.time = static_cast<double>(*iter_time);
        cloud->push_back(p);
    }

    cloud->width = cloud->size();
}

}  // namespace


// ***********************************************************
// Constructor & Destructor
// ***********************************************************
ImageProjectionNode::ImageProjectionNode()
    : Node("image_projection_node"), ringFlag(0), deskewFlag(0)
{
    config = std::make_unique<Config>();
    setup_config();
    setup_subpub();
    initialize();

    RCLCPP_INFO(this->get_logger(), "Setup ImageProjectionNode!");
}

ImageProjectionNode::~ImageProjectionNode() {}


// ***********************************************************
// Setup
// ***********************************************************
void ImageProjectionNode::setup_config() {
    config->lidarFrame       = this->declare_parameter("lidarFrame",        "base_link");
    config->imuTopic         = this->declare_parameter("imuTopic",          "imu_raw");
    config->odomTopic        = this->declare_parameter("odomTopic",         "odometry/imu");
    config->pointCloudTopic  = this->declare_parameter("pointCloudTopic",   "points_raw");

    config->N_SCAN       = this->declare_parameter<int>("N_SCAN",        64);
    config->Horizon_SCAN = this->declare_parameter<int>("Horizon_SCAN", 4096);

    // Default to identity (row-major 3x3). An empty default would leave extRotV with no storage,
    // and the Eigen::Map below would then read 9 doubles past the end of an empty buffer.
    const std::vector<double> identity3x3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    config->extRotV    = this->declare_parameter<std::vector<double>>("extRotV",    identity3x3);
    config->extRotRPYV = this->declare_parameter<std::vector<double>>("extRotRPYV", identity3x3);
    config->extRot  = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config->extRotV.data(),    3, 3);
    config->extRPY  = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(config->extRotRPYV.data(), 3, 3);
    config->extQRPY = Eigen::Quaterniond(config->extRPY).inverse();

    config->lidarMinRange    = this->declare_parameter<float>("lidarMinRange",  1.0);
    config->lidarMaxRange    = this->declare_parameter<float>("lidarMaxRange",  400.0);
    // Parameter name aligned with feature_extraction ("downsampleRate"); previously "downsample_rate",
    // so a shared YAML config silently applied to only one of the two nodes.
    config->downsampleRate   = this->declare_parameter<int>("downsampleRate", 1);
    // Must match the value used in featureExtraction for curvature computation.
    config->lidarCurvatureFeatureExtractionNeighbors =
        this->declare_parameter<int>("lidarCurvatureFeatureExtractionNeighbors", 5);
}

void ImageProjectionNode::setup_subpub() {
    subImu = this->create_subscription<sensor_msgs::msg::Imu>(
        config->imuTopic, 2000,
        std::bind(&ImageProjectionNode::imuHandler, this, std::placeholders::_1)
    );
    subOdom = this->create_subscription<nav_msgs::msg::Odometry>(
        config->odomTopic + "_incremental", 2000,
        std::bind(&ImageProjectionNode::odometryHandler, this, std::placeholders::_1)
    );
    subPointCloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        config->pointCloudTopic, 5,
        std::bind(&ImageProjectionNode::cloudHandler, this, std::placeholders::_1)
    );

    pubExtractedCloud  = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/deskew/cloud_deskewed", 1);
    pubPointCloudInfo  = this->create_publisher<slam_msgs::msg::CloudInfo>("lio_sam/deskew/cloud_info", 1);
}

void ImageProjectionNode::initialize() {
    allocateMemory();
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

void ImageProjectionNode::allocateMemory() {
    laserCloudIn.reset(new pcl::PointCloud<LidarPoint>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    extractedCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(config->N_SCAN * config->Horizon_SCAN);

    cloudInfo.start_ring_index.assign(config->N_SCAN, 0);
    cloudInfo.end_ring_index.assign(config->N_SCAN, 0);

    cloudInfo.point_col_ind.assign(config->N_SCAN * config->Horizon_SCAN, 0);
    cloudInfo.point_range.assign(config->N_SCAN * config->Horizon_SCAN, 0);

    resetParameters();
}

void ImageProjectionNode::resetParameters() {
    laserCloudIn->clear();
    extractedCloud->clear();
    rangeMat = cv::Mat(config->N_SCAN, config->Horizon_SCAN, CV_32F,
                       cv::Scalar::all(std::numeric_limits<float>::max()));

    imuPointerCur  = 0;
    firstPointFlag = true;
    odomDeskewFlag = false;

    for (int i = 0; i < queueLength; i++) {
        imuTime[i] = 0;
        imuRotX[i] = 0;
        imuRotY[i] = 0;
        imuRotZ[i] = 0;
    }
}


// ***********************************************************
// Subscription/Publisher handlers
// ***********************************************************
void ImageProjectionNode::imuHandler(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    sensor_msgs::msg::Imu thisImu = imuConverter(*msg, config->extRot, config->extQRPY, this->get_logger());

    std::lock_guard<std::mutex> lock1(imuLock);
    imuQueue.push_back(thisImu);
}

void ImageProjectionNode::odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    std::lock_guard<std::mutex> lock2(odomLock);
    odomQueue.push_back(*msg);
}

void ImageProjectionNode::cloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    // validateFields() throws if a required field is missing. An exception escaping a ROS2 callback
    // calls std::terminate, so catch it here and shut down cleanly instead.
    try {
        if (!cachePointCloud(msg)) {
            RCLCPP_INFO(this->get_logger(), "Not enough point clouds yet: %zu / %d", cloudQueue.size(), 2);
            return;
        }
    } catch (const std::runtime_error& ex) {
        RCLCPP_ERROR(this->get_logger(), "Failed to process point cloud: %s", ex.what());
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Imu Queue Size: %zu, Odometry Queue Size: %zu", imuQueue.size(), odomQueue.size());
    if (!deskewInfo()) {
        RCLCPP_INFO(this->get_logger(), "deskewInfo not prepared!");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "deskewInfo prepared!");

    projectPointCloud();
    cloudExtraction();
    publishClouds();
    resetParameters();
}


// ***********************************************************
// Data Preparation
// ***********************************************************
bool ImageProjectionNode::cachePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    cloudQueue.push_back(*msg);
    if (cloudQueue.size() <= 2)
        return false;

    currentCloudMsg = std::move(cloudQueue.front());
    cloudQueue.pop_front();
    // Must extract from currentCloudMsg (the scan we just dequeued), not msg (the scan that just
    // arrived, two frames newer). Using msg desynced the points from the header/timestamps below.
    extractPointCloud(currentCloudMsg, laserCloudIn);

    cloudHeader = currentCloudMsg.header;
    timeScanCur = ROS_TIME(currentCloudMsg);
    // Guard against an empty cloud before dereferencing back().
    if (laserCloudIn->points.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received an empty point cloud, skipping.");
        return false;
    }
    timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

    RCLCPP_INFO(this->get_logger(), "Current time scan: '%.9f'. End time scan: '%.9f'", timeScanCur, timeScanEnd);

    if (!laserCloudIn->is_dense) {
        RCLCPP_ERROR(this->get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
        rclcpp::shutdown();
    }

    // check ring channel
    // TODO: recover row index from elevation angle if ring channel is absent
    if (ringFlag == 0) {
        ringFlag = -1;
        for (int i = 0; i < static_cast<int>(currentCloudMsg.fields.size()); ++i) {
            if (currentCloudMsg.fields[i].name == "ring") {
                ringFlag = 1;
                break;
            }
        }
        if (ringFlag == -1) {
            RCLCPP_ERROR(this->get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
            rclcpp::shutdown();
        }
    }

    // check point time
    if (deskewFlag == 0) {
        deskewFlag = -1;
        for (auto& field : currentCloudMsg.fields) {
            if (field.name == "time") {
                deskewFlag = 1;
                break;
            }
        }
        if (deskewFlag == -1) {
            RCLCPP_WARN(this->get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }
    }

    return true;
}

bool ImageProjectionNode::deskewInfo() {
    std::lock_guard lock1(imuLock);
    std::lock_guard lock2(odomLock);

    // make sure IMU data avilable for the scan
    //
    // Want to make sure that we have all the IMU data within the scan time range
    //
    // For motion distortion correction, the IMU data must fully bracket the entire LiDAR scan. That requires.
    //     - IMU data starting at or before `timeScanCur` (scan start)
    //     - IMU data ending at or after `timeScanEnd` (scan end)
    // When the IMU data is entirely inside the scan window; i.e., `front() >= timeScanCur` && `back() < timeScanEnd`
    // we have the worst case. We would have no IMU coverage at either boundary of the scan where you need it most
    // for distortion correction.
    if (imuQueue.empty()
        || ROS_TIME(imuQueue.front()) > timeScanCur
        || ROS_TIME(imuQueue.back())  < timeScanEnd)
    {
        RCLCPP_DEBUG(this->get_logger(), "Waiting for IMU data...");
        return false;
    }

    imuDeskewInfo();
    odomDeskewInfo();
    return true;
}

void ImageProjectionNode::imuDeskewInfo() {
    cloudInfo.imu_available = false;

    // Get rid of out of date IMU messages
    while (!imuQueue.empty()) {
        if (ROS_TIME(imuQueue.front()) < timeScanCur - 0.01)
            imuQueue.pop_front();
        else
            break;
    }

    if (imuQueue.empty()) {
        RCLCPP_INFO(this->get_logger(), "No IMU data after filtering out old IMU messages!");
        return;
    }

    imuPointerCur = 0;
    for (int i = 0; i < (int)imuQueue.size(); ++i) {
        sensor_msgs::msg::Imu thisImuMsg  = imuQueue[i];
        double                currentImuTime = ROS_TIME(thisImuMsg);

        // get roll, pitch, and yaw estimation for initialization at the closest time to the scan start
        // initial orientation for IMU preintegration. Since we filter out older IMU messages that are older
        // than 0.01 seconds before the current scan start time, we won't have to run this many times.
        if (currentImuTime <= timeScanCur)
            imuRPY2rosRPY(thisImuMsg, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);

        if (currentImuTime > timeScanEnd + 0.01)
            break;

        if (imuPointerCur == 0) {
            imuRotX[0] = 0; imuRotY[0] = 0; imuRotZ[0] = 0;
            imuTime[0] = currentImuTime;
            ++imuPointerCur;
            continue;
        }

        double angular_x, angular_y, angular_z;
        imuAngular2rosAngular(thisImuMsg, angular_x, angular_y, angular_z);

        // integrate rotation
        double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
        imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
        imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
        imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
        imuTime[imuPointerCur] = currentImuTime;
        ++imuPointerCur;
    }

    --imuPointerCur;

    if (imuPointerCur <= 0)
        return;

    cloudInfo.imu_available = true;
}

void ImageProjectionNode::odomDeskewInfo() {
    cloudInfo.odom_available = false;

    while (!odomQueue.empty()) {
        if (ROS_TIME(odomQueue.front()) < timeScanCur - 0.01) {
            auto time_ = ROS_TIME(odomQueue.front());
            RCLCPP_INFO(this->get_logger(), "Removing Odometry message at time '%.9f'", time_);
            odomQueue.pop_front();
        } else {
            break;
        }
    }

    if (odomQueue.empty()) {
        RCLCPP_INFO(this->get_logger(), "No Odometry data after filtering out old Odometry messages!");
        return;
    }

    // Means we have no initial estimate of the odometry pose
    if (ROS_TIME(odomQueue.front()) > timeScanCur)
        return;

    // get start odometry at the beginning of the scan
    nav_msgs::msg::Odometry startOdomMsg;
    for (int i = 0; i < (int)odomQueue.size(); ++i) {
        startOdomMsg = odomQueue[i];
        if (ROS_TIME(startOdomMsg) >= timeScanCur)
            break;
    }

    tf2::Quaternion orientation;
    tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

    double roll, pitch, yaw;
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // Initial guess used in mapOptimization
    cloudInfo.initial_guess_x     = startOdomMsg.pose.pose.position.x;
    cloudInfo.initial_guess_y     = startOdomMsg.pose.pose.position.y;
    cloudInfo.initial_guess_z     = startOdomMsg.pose.pose.position.z;
    cloudInfo.initial_guess_roll  = roll;
    cloudInfo.initial_guess_pitch = pitch;
    cloudInfo.initial_guess_yaw   = yaw;

    cloudInfo.odom_available = true;

    // get end odometry at the end of the scan
    odomDeskewFlag = false;

    if (ROS_TIME(odomQueue.back()) < timeScanEnd)
        return;

    nav_msgs::msg::Odometry endOdomMsg;
    for (int i = 0; i < (int)odomQueue.size(); ++i) {
        endOdomMsg = odomQueue[i];
        if (ROS_TIME(endOdomMsg) >= timeScanEnd)
            break;
    }

    if (int(std::round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
        return;

    Eigen::Affine3f transBegin = getTransformation(
        startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y,
        startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    Eigen::Affine3f transEnd = getTransformation(
        endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y,
        endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

    // transBt is the relative transform between the odometry pose at scan start and scan end. It captures the
    // full motion of the sensor during the scan. For motion distortion correction (deskewing), LIO-SAM only uses
    // the translation component (`odomIncreX/Y/Z`) from this odometry delta.
    //     - The rotation component of motion distortion is already handled more accurately by IMU integration -
    //        gyroscopes measure angular velocity directly and at much higher frequency than odometry.
    //     - The translation component from odometry complements the IMU since acclerometers are noisier for
    //        recovering displacement.
    Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
    odomIncreX = transBt.translation().x();
    odomIncreY = transBt.translation().y();
    odomIncreZ = transBt.translation().z();

    odomDeskewFlag = true;
}


// ***********************************************************
// projectPointCloud()
// ***********************************************************
void ImageProjectionNode::projectPointCloud() {
    int cloudSize = laserCloudIn->points.size();

    // Angular resolution per column; static so it is computed once.
    static float ang_res_x = 360.0f / float(config->Horizon_SCAN);

    for (int i = 0; i < cloudSize; ++i) {
        PointType thisPoint;
        thisPoint.x         = laserCloudIn->points[i].x;
        thisPoint.y         = laserCloudIn->points[i].y;
        thisPoint.z         = laserCloudIn->points[i].z;
        thisPoint.intensity = laserCloudIn->points[i].intensity;

        const float range = pointDistance(thisPoint);
        if (range < config->lidarMinRange || range > config->lidarMaxRange)
            continue;

        const int rowIdn = laserCloudIn->points[i].ring;
        if (rowIdn < 0 || rowIdn >= config->N_SCAN)
            continue;

        if (rowIdn % config->downsampleRate != 0)
            continue;

        // Velodyne sensor
        // Compute raw azimuth angle using `atan2(x, y)` to measure the angle from the y-axis rather than the
        // x-axis, rotating the coordinate frame 90 degrees. Result is in (-180 degrees, 180 degrees].
        const float horizonAngle = std::atan2(thisPoint.x, thisPoint.y) * 180.0f / M_PI;
        // -90 corrects for the y-axis offset; negation + Horizon_SCAN/2 maps to column index.
        // TODO: verify scan rotation direction (cw vs ccw) for this sensor
        int columnIdn = -round((horizonAngle - 90.0f) / ang_res_x) + config->Horizon_SCAN / 2;
        if (columnIdn >= config->Horizon_SCAN)
            columnIdn -= static_cast<int>(config->Horizon_SCAN);

        if (columnIdn < 0 || columnIdn >= config->Horizon_SCAN)
            continue;

        // Already have a value at this (row, col)
        if (rangeMat.at<float>(rowIdn, columnIdn) != std::numeric_limits<float>::max())
            continue;

        thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

        rangeMat.at<float>(rowIdn, columnIdn) = range;
        fullCloud->points[columnIdn + rowIdn * config->Horizon_SCAN] = thisPoint;
    }
}

PointType ImageProjectionNode::deskewPoint(PointType* point, double relTime) {
    if (deskewFlag == -1 || cloudInfo.imu_available == false)
        return *point;

    double pointTime = timeScanCur + relTime;

    float rotXCur, rotYCur, rotZCur;
    findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

    float posXCur, posYCur, posZCur;
    findPosition(relTime, &posXCur, &posYCur, &posZCur);

    if (firstPointFlag) {
        transStartInverse = getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur).inverse();
        firstPointFlag = false;
    }

    const Eigen::Affine3f transFinal = getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
    Eigen::Affine3f transBt    = transStartInverse * transFinal;

    PointType newPoint;
    newPoint.x         = transBt(0,0)*point->x + transBt(0,1)*point->y + transBt(0,2)*point->z + transBt(0,3);
    newPoint.y         = transBt(1,0)*point->x + transBt(1,1)*point->y + transBt(1,2)*point->z + transBt(1,3);
    newPoint.z         = transBt(2,0)*point->x + transBt(2,1)*point->y + transBt(2,2)*point->z + transBt(2,3);
    newPoint.intensity = point->intensity;

    return newPoint;
}

// `const` here indicates this function is "read-only" and will not alter any of the object's internal state.
void ImageProjectionNode::findRotation(double pointTime, float* rotXCur, float* rotYCur, float* rotZCur) const {
    *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

    int imuPointerFront = 0;
    while (imuPointerFront < imuPointerCur) {
        if (pointTime < imuTime[imuPointerFront])
            break;
        ++imuPointerFront;
    }

    if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
        // Have an exact matching time
        *rotXCur = imuRotX[imuPointerFront];
        *rotYCur = imuRotY[imuPointerFront];
        *rotZCur = imuRotZ[imuPointerFront];
    } else {
        // Need to interpolate since we are not at an exact IMU time
        const int imuPointerBack = imuPointerFront - 1;
        double ratioFront = (pointTime           - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        double ratioBack  = (imuTime[imuPointerFront] - pointTime)          / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
        *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
        *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
    }
}

void ImageProjectionNode::findPosition(double relTime, float* posXCur, float* posYCur, float* posZCur) {
    *posXCur = 0; *posYCur = 0; *posZCur = 0;

    // Positional deskew has little benefit at walking speed, so skip when unavailable.
    if (cloudInfo.odom_available == false || odomDeskewFlag == false)
        return;

    float ratio = relTime / (timeScanEnd - timeScanCur);
    *posXCur = ratio * odomIncreX;
    *posYCur = ratio * odomIncreY;
    *posZCur = ratio * odomIncreZ;
}


// ***********************************************************
// cloudExtraction()
// ***********************************************************
void ImageProjectionNode::cloudExtraction() {
    int count = 0;
    for (int i = 0; i < config->N_SCAN; i++) {
        // Offset ensures each extracted point has enough neighbors for curvature computation.
        cloudInfo.start_ring_index[i] = count - 1 + config->lidarCurvatureFeatureExtractionNeighbors;

        for (int j = 0; j < config->Horizon_SCAN; j++) {
            if (rangeMat.at<float>(i, j) != std::numeric_limits<float>::max()) {
                // mark the point's column index for marking occlusion later
                cloudInfo.point_col_ind[count] = j;
                // save range info
                cloudInfo.point_range[count]   = rangeMat.at<float>(i, j);
                // save extracted cloud
                extractedCloud->push_back(fullCloud->points[j + i * config->Horizon_SCAN]);
                // size of extracted cloud
                count++;
            }
        }
        // Mirror the start-of-ring offset so both ends reserve the same neighbor margin for
        // curvature computation (was a hardcoded 5, which only matched the default neighbor count).
        cloudInfo.end_ring_index[i] = count - 1 - config->lidarCurvatureFeatureExtractionNeighbors;
    }
}


// ***********************************************************
// publishClouds()
// ***********************************************************
void ImageProjectionNode::publishClouds() {
    RCLCPP_INFO(this->get_logger(), "Publishing the deskewed cloud!");
    cloudInfo.header        = cloudHeader;
    cloudInfo.cloud_deskewed = publishCloud<PointType>(
        pubExtractedCloud, extractedCloud, rclcpp::Time(cloudHeader.stamp), config->lidarFrame);
    pubPointCloudInfo->publish(cloudInfo);
}
