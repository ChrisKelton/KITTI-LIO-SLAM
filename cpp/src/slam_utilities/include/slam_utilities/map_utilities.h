//
// Created by ckelton on 6/12/26.
//
#pragma once

#ifndef SLAM_UTILITIES_MAP_UTILITIES_H
#define SLAM_UTILITIES_MAP_UTILITIES_H

#include <gtsam/geometry/Pose3.h>

#include "utility.h"


template<typename T0, typename T1>
pcl::PointCloud<T0>::Ptr transformPointCloud(pcl::PointCloud<T0>::Ptr cloudIn, T1* transformIn, int numberOfCores = 2) {
    pcl::PointCloud<T0>::Ptr cloudOut(new pcl::PointCloud<T0>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
        const auto& pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

template<typename T>
gtsam::Pose3 pclPointTogtsamPose3(T thisPoint) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

template<typename T>
Eigen::Affine3f pclPointToAffine3f(T thisPoint) {
    return getTransformation(thisPoint.x, thisPoint.y thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

Eigen::Affine3f trans2Affine3f(float transformIn[]) {
    return getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

template<typename T>
T trans2PointTypePose(float transformIn[]) {
    T thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

float constraintTransformation(float value, float limit) {
    if (value < -limit) {
        value = -limit;
    } else if (value > limit) {
        value = limit;
    }

    return value;
}

#endif //SLAM_UTILITIES_MAP_UTILITIES_H