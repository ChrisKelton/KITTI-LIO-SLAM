//
// Created by ckelton on 6/9/26.
//
#pragma once

#ifndef KITTI_LIO_SLAM_UTILITY_H
#define KITTI_LIO_SLAM_UTILITY_H

/* Include ordering convention (Google Style / ROS2 convention)
 *
 * 1. Corresponding header (for .cpp files - the header this .cpp implements)
 * 2. C system headers (<cmath>, <cstring>)
 * 3. C++ standard library headers (<string>, <vector>)
 * 4. Third-party library headers (Eigen, PCL, OpenCV)
 * 5. ROS2 headers (rclcpp, sensor_msgs, etc.)
 * 6. Local project headers ("utils/Config.h", "slam_utilities/utility.h")
 *
 */

#include <cmath>

#include <Eigen/Core>  // Matrix, Vector types
#include <Eigen/Geometry>  // Quaternion, transform

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


typedef pcl::PointXYZI PointType;


inline sensor_msgs::msg::Imu imuConverter(
    const sensor_msgs::msg::Imu& imu_in,
    const Eigen::Matrix3d& extRot,
    const Eigen::Quaterniond& extQRPY,
    const rclcpp::Logger& logger
) {
    sensor_msgs::msg::Imu imu_out;
    imu_out.header = imu_in.header;
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll, pitch, yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (std::sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1) {
        RCLCPP_ERROR(logger, "Invalid quaternion, please use a 9-axis IMU!");
        rclcpp::shutdown();
    }

    return imu_out;
}


// General tips on when to convert Imu orientation quaternion to euler angles:
//    1. Gravity alignment:
//        - Roll and pitch relative to gravity are directly observable from accelerometer data and are used to
//            initialize and correct the IMU preintegration. It's easier to isolate and work with individual rotation
//            axes in Euler form:
//
//    2. Extrinsic calibration application:
//        - Correcting IMU orientation is easier to verify and debug in RPY space - you can sanity check that a
//            90 degree mounting rotation shows up as exactly 90 degrees on the right axis.
//
//    3. IMU preintegration initialization:
//        - GTSAM's `PreintegrationParams` expects gravity direction in Euler form to set up the integration correctly.
//
//    Euler Angle Issues:
//        - Gimbal Lock: At pitch = +/- 90 degrees, roll and yaw become indistinguishable.
//        - Interpolation: Interpolating between two RPY representations can take the long way around.
//        - Discontinuities: Wrapping at +/- 180 degrees can cause sudden jumps.
//        - Not composable: Combining two rotations requires converting back to matrix/quaternion first.
template<typename T>
void imuRPY2rosRPY(const sensor_msgs::msg::Imu& thisImuMsg, T& rosRoll, T& rosPitch, T& rosYaw) {
    // Convert geometry_msgs::Quaternion to tf2::Quaternion
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg.orientation, orientation);

    // getRPY requires tf2Scalar& (double&), so use intermediates before assigning to T
    double roll, pitch, yaw;
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    rosRoll = static_cast<T>(roll);
    rosPitch = static_cast<T>(pitch);
    rosYaw = static_cast<T>(yaw);
}


template<typename T>
void imuAngular2rosAngular(const sensor_msgs::msg::Imu& thisImuMsg, T& angular_x, T& angular_y, T& angular_z) {
    angular_x = thisImuMsg.angular_velocity.x;
    angular_y = thisImuMsg.angular_velocity.y;
    angular_z = thisImuMsg.angular_velocity.z;
}


template<typename T>
double ROS_TIME(const T& msg) {
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
}


inline float pointDistance(const PointType& p) {
    return std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


inline float pointDistance(const PointType& p1, const PointType& p2) {
    return std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


template<typename T>
sensor_msgs::msg::PointCloud2 publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& thisPub,
    const typename pcl::PointCloud<T>::Ptr& thisCloud,
    const rclcpp::Time thisStamp,
    const std::string thisFrame
) {
    sensor_msgs::msg::PointCloud2 tmpCloud;
    pcl::toROSMsg(*thisCloud, tmpCloud);
    tmpCloud.header.stamp = thisStamp;
    tmpCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
        thisPub->publish(tmpCloud);
    return tmpCloud;
}


inline Eigen::Affine3f getTransformation(const float x, const float y, const float z, const float roll, const float pitch, const float yaw) {
    Eigen::Affine3f trans = Eigen::Affine3f::Identity();
    trans.translation() << x, y, z;
    trans.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
    trans.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
    trans.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    return trans;
}

#endif //KITTI_LIO_SLAM_UTILITY_H
