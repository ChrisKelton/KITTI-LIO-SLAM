//
// Created by ckelton on 6/10/26.
//

#include <Eigen/Dense>
#include <pcl/common/transforms.h>

#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>

#include "TransformFusionNode.h"


namespace {

Eigen::Affine3f odom2affine(const nav_msgs::msg::Odometry& odom) {
    double roll, pitch, yaw;
    const double x = odom.pose.pose.position.x;
    const double y = odom.pose.pose.position.y;
    const double z = odom.pose.pose.position.z;
    tf2::Quaternion orientation;
    tf2::fromMsg(odom.pose.pose.orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    return getTransformation(x, y, z, roll, pitch, yaw);
}

}  // namespace


TransformFusionNode::TransformFusionNode()
    : Node("transform_fusion_node"), tfMap2Odom(*this), tfOdom2BaseLink(*this)
{
    config = new TransformFusionNodeConfig();
    setup_config();
    initialize();
    setup_subpub();
}

TransformFusionNode::~TransformFusionNode() {}

void TransformFusionNode::setup_config() {
    config->lidarFrame    = this->declare_parameter("lidarFrame",    "base_link");
    config->baselinkFrame = this->declare_parameter("baselinkFrame", "base_link");
    config->mapFrame      = this->declare_parameter("mapFrame",      "map");
    config->odometryFrame = this->declare_parameter("odometryFrame", "odom");
    config->odomTopic     = this->declare_parameter("odomTopic",     "odometry/imu");
}

void TransformFusionNode::initialize() {
    tfBuffer   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tfListener = std::make_unique<tf2_ros::TransformListener>(*tfBuffer);

    if (config->lidarFrame != config->baselinkFrame) {
        try {
            // Blocks up to 3 s waiting for the lidar→baselink static transform.
            lidar2Baselink = tfBuffer->lookupTransform(
                config->lidarFrame, config->baselinkFrame,
                tf2::TimePointZero, tf2::durationFromSec(3.0));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
        }
    }
}

void TransformFusionNode::setup_subpub() {
    subLaserOdometry = this->create_subscription<nav_msgs::msg::Odometry>(
        "lio_sam/mapping/odometry", 5,
        std::bind(&TransformFusionNode::lidarOdometryHandler, this, std::placeholders::_1));
    subImuOdometry = this->create_subscription<nav_msgs::msg::Odometry>(
        config->odomTopic + "_incremental", 2000,
        std::bind(&TransformFusionNode::imuOdometryHandler, this, std::placeholders::_1));

    pubImuOdometry = this->create_publisher<nav_msgs::msg::Odometry>(config->odomTopic, 2000);
    pubImuPath     = this->create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", 1);
}

void TransformFusionNode::lidarOdometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& odomMsg) {
    std::lock_guard lock(mtx);

    lidarOdomAffine = odom2affine(*odomMsg);
    lidarOdomTime   = ROS_TIME(*odomMsg);
}

void TransformFusionNode::imuOdometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr& odomMsg) {
    // Broadcast a fixed identity map→odom transform. The real correction is
    // applied by fusing lidar odometry with the incremental IMU odometry below.
    geometry_msgs::msg::TransformStamped map_to_odom;
    map_to_odom.header.stamp    = odomMsg->header.stamp;
    map_to_odom.header.frame_id = config->mapFrame;
    map_to_odom.child_frame_id  = config->odometryFrame;
    map_to_odom.transform.translation.x = 0.0;
    map_to_odom.transform.translation.y = 0.0;
    map_to_odom.transform.translation.z = 0.0;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, 0.0);
    map_to_odom.transform.rotation.x = q.x();
    map_to_odom.transform.rotation.y = q.y();
    map_to_odom.transform.rotation.z = q.z();
    map_to_odom.transform.rotation.w = q.w();
    tfMap2Odom.sendTransform(map_to_odom);

    std::lock_guard lock(mtx);

    imuOdomQueue.push_back(*odomMsg);

    if (lidarOdomTime == -1)
        return;

    while (!imuOdomQueue.empty()) {
        if (ROS_TIME(imuOdomQueue.front()) <= lidarOdomTime)
            imuOdomQueue.pop_front();
        else
            break;
    }

    imuOdomAffineFront = odom2affine(imuOdomQueue.front());
    imuOdomAffineBack  = odom2affine(imuOdomQueue.back());
    Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
    Eigen::Affine3f imuOdomAffineLast  = lidarOdomAffine * imuOdomAffineIncre;

    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

    // publish latest odometry
    nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
    laserOdometry.pose.pose.position.x = x;
    laserOdometry.pose.pose.position.y = y;
    laserOdometry.pose.pose.position.z = z;
    q.setRPY(roll, pitch, yaw);
    laserOdometry.pose.pose.orientation = tf2::toMsg(q);
    pubImuOdometry->publish(laserOdometry);

    // publish odom→baselink tf
    tf2::Transform tCur;
    tf2::fromMsg(laserOdometry.pose.pose, tCur);
    if (config->lidarFrame != config->baselinkFrame) {
        tf2::Transform lidar2Baselink_tf;
        tf2::fromMsg(lidar2Baselink.transform, lidar2Baselink_tf);
        tCur = tCur * lidar2Baselink_tf;
    }
    geometry_msgs::msg::TransformStamped odom_2_baselink;
    odom_2_baselink.header.stamp    = odomMsg->header.stamp;
    odom_2_baselink.header.frame_id = config->odometryFrame;
    odom_2_baselink.child_frame_id  = config->baselinkFrame;
    odom_2_baselink.transform       = tf2::toMsg(tCur);
    tfOdom2BaseLink.sendTransform(odom_2_baselink);

    // publish IMU path at 10 Hz
    static nav_msgs::msg::Path imuPath;
    static double last_path_time = -1;
    double imuTime = ROS_TIME(imuOdomQueue.back());
    if (imuTime - last_path_time > 0.1) {
        last_path_time = imuTime;
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp    = imuOdomQueue.back().header.stamp;
        pose_stamped.header.frame_id = config->odometryFrame;
        pose_stamped.pose            = laserOdometry.pose.pose;
        imuPath.poses.push_back(pose_stamped);
        while (!imuPath.poses.empty() && ROS_TIME(imuPath.poses.front()) < lidarOdomTime - 0.1)
            imuPath.poses.erase(imuPath.poses.begin());
        if (pubImuPath->get_subscription_count() != 0) {
            imuPath.header.stamp    = imuOdomQueue.back().header.stamp;
            imuPath.header.frame_id = config->odometryFrame;
            pubImuPath->publish(imuPath);
        }
    }
}
