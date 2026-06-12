//
// Created by ckelton on 6/2/26.
//

/*
Using brackets looks for system installed packages at /usr/include/ ...

Using quotes for package means to search locally for the include file and on any paths included by -I, if it is not
    found, then default back to looking like brackets looks. I.e., in /usr/include/, ...
*/


#include "LidarFeatureExtractionNode.h"
#include "slam_utilities/utility.h"


namespace {

struct by_value {
    bool operator()(smoothness_t const& left, smoothness_t const& right) {
        return left.value < right.value;
    }
};

}  // namespace


void LidarFeatureExtractionNode::setup_config() {
    config->lidarFrame = this->declare_parameter("lidarFrame", "base_link");

    config->N_SCAN = this->declare_parameter<int>("N_SCAN", 64);
    config->Horizon_SCAN = this->declare_parameter<int>("Horizon_SCAN", 4096);
    config->downsampleRate = this->declare_parameter<int>("downsampleRate", 1);
    // Values specific for KITTI dataset
    //     Most ground-truth in benchmark are maxed to 80m
    config->lidarMinRange = this->declare_parameter<float>("lidarMinRange", 1.0);
    config->lidarMaxRange = this->declare_parameter<float>("lidarMaxRange", 400.0);

    config->surfLeafSize = this->declare_parameter<float>("surfLeafSize", 0.4);
    config->edgeThreshold = this->declare_parameter<float>("edgeThreshold", 0.1);
    config->surfThreshold = this->declare_parameter<float>("surfThreshold", 0.1);

    config->lidarCurvatureFeatureExtractionNeighbors = static_cast<uint>(this->declare_parameter<int>("lidarCurvatureFeatureExtractionNeighbors", 5));
    config->pixelDiffTh = static_cast<uint>(this->declare_parameter<int>("pixelDiffTh", 10));
}

void LidarFeatureExtractionNode::initialize() {
    cloudSmoothness.resize(config->N_SCAN*config->Horizon_SCAN);
    downsizeFilter.setLeafSize(config->surfLeafSize, config->surfLeafSize, config->surfLeafSize);

    extractedCloud.reset(new pcl::PointCloud<PointType>());
    cornerCloud.reset(new pcl::PointCloud<PointType>());
    surfaceCloud.reset(new pcl::PointCloud<PointType>());

    const int cloudSize = config->N_SCAN * config->Horizon_SCAN;
    cloudCurvature.assign(cloudSize, 0.0f);
    cloudNeighborPicked.assign(cloudSize, 0);
    cloudLabel.assign(cloudSize, 0);
}

void LidarFeatureExtractionNode::setup_subpub() {
    subLaserCloudInfo = this->create_subscription<slam_msgs::msg::CloudInfo>(
        "lio_sam/deskew/cloud_info", 1,
        std::bind(&LidarFeatureExtractionNode::laserCloudInfoHandler, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribing: lio_sam/deskew/cloud_info");

    pubLaserCloudInfo = this->create_publisher<slam_msgs::msg::CloudInfo>("lio_sam/feature/cloud_info", 1);
    RCLCPP_INFO(this->get_logger(), "Publishing: lio_sam/feature/cloud_info");
    pubCornerPoints = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/feature/cloud_corner", 1);
    RCLCPP_INFO(this->get_logger(), "Publishing: lio_sam/feature/cloud_corner");
    pubSurfacePoints = this->create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/feature/cloud_surface", 1);
    RCLCPP_INFO(this->get_logger(), "Publishing: lio_sam/feature/cloud_surface");
}

void LidarFeatureExtractionNode::laserCloudInfoHandler(const slam_msgs::msg::CloudInfo::ConstSharedPtr& msg) {
    cloudInfo = *msg;
    cloudHeader = msg->header;
    pcl::fromROSMsg(msg->cloud_deskewed, *extractedCloud);

    calculateSmoothness();

    markOccludedPoints();

    extractFeatures();

    publishFeatureCloud();
}

void LidarFeatureExtractionNode::calculateSmoothness() {
    int cloudSize = extractedCloud->points.size();
    for (int i = config->lidarCurvatureFeatureExtractionNeighbors; i < cloudSize - config->lidarCurvatureFeatureExtractionNeighbors; ++i) {
        float diffRange = cloudInfo.point_range[i-config->lidarCurvatureFeatureExtractionNeighbors];
        for (int j = -(config->lidarCurvatureFeatureExtractionNeighbors)+1; j <= config->lidarCurvatureFeatureExtractionNeighbors; j++) {
            if (j == 0) {
                diffRange -= cloudInfo.point_range[i] * 10;
            } else {
                diffRange += cloudInfo.point_range[i+j];
            }
        }
        cloudCurvature[i] = diffRange * diffRange;

        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;
        // cloudSmoothness for sorting
        cloudSmoothness[i].value = cloudCurvature[i];
        cloudSmoothness[i].ind = i;
    }
}

void LidarFeatureExtractionNode::markOccludedPoints() {
    int cloudSize = extractedCloud->points.size();
    // mark occluded points and parallel beam points
    for (int i = config->lidarCurvatureFeatureExtractionNeighbors; i < cloudSize - config->lidarCurvatureFeatureExtractionNeighbors - 1; ++i) {
        // occluded points
        float depth1 = cloudInfo.point_range[i];
        float depth2 = cloudInfo.point_range[i+1];
        int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i]));

        // Mark out 'lidarCurvatureFeatureExtractionNeighbors' pixels to the left or right of the farthest range pixel
        if (columnDiff < config->pixelDiffTh) {
            // default is 10 pixel difference in range image
            if (depth1 - depth2 > 0.3) {
                for (int j = 0; j <= config->lidarCurvatureFeatureExtractionNeighbors; ++j) {
                    cloudNeighborPicked[i - j] = 1;
                }
            } else if (depth2 - depth1 > 0.3) {
                for (int j = 1; j <= config->lidarCurvatureFeatureExtractionNeighbors+1; ++j) {
                    cloudNeighborPicked[i + j] = 1;
                }
            }
        }
        // parallel beam
        float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
        float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));

        if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.2 * cloudInfo.point_range[i]) {
            cloudNeighborPicked[i] = 1;
        }
    }
}

void LidarFeatureExtractionNode::extractFeatures() {
    cornerCloud->clear();
    surfaceCloud->clear();

    pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surfaceCloudScanDs(new pcl::PointCloud<PointType>());

    // Extracting highest curvature points within neighborhood
    for (int i = 0; i < config->N_SCAN; ++i) {
        surfaceCloudScan->clear();
        for (int j = 0; j < config->lidarCurvatureFeatureExtractionNeighbors + 1; ++j) {
            int sp = (cloudInfo.start_ring_index[i] * ((config->lidarCurvatureFeatureExtractionNeighbors + 1) - j) + cloudInfo.end_ring_index[i] * j) / (config->lidarCurvatureFeatureExtractionNeighbors + 1);
            int ep = (cloudInfo.start_ring_index[i] * (config->lidarCurvatureFeatureExtractionNeighbors - j) + cloudInfo.end_ring_index[i] * (j + 1)) / config->lidarCurvatureFeatureExtractionNeighbors;

            if (sp >= ep)
                continue;

            std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--) {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > config->edgeThreshold) {
                    largestPickedNum++;
                    if (largestPickedNum <= 20) {
                        cloudLabel[ind] = 1;
                        cornerCloud->push_back(extractedCloud->points[ind]);
                    } else {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= config->lidarCurvatureFeatureExtractionNeighbors; ++l) {
                        int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));;
                        if (columnDiff > config->pixelDiffTh)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -(config->lidarCurvatureFeatureExtractionNeighbors); --l) {
                        int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));;
                        if (columnDiff > config->pixelDiffTh)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; ++k) {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < config->surfThreshold) {
                    cloudLabel[ind] = -1;
                    cloudNeighborPicked[ind] = 1;

                    for (int l = 1; l <= config->lidarCurvatureFeatureExtractionNeighbors; ++l) {
                        int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                        if (columnDiff > config->pixelDiffTh)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -(config->lidarCurvatureFeatureExtractionNeighbors); --l) {
                        int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                        if (columnDiff > config->pixelDiffTh)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; ++k) {
                if (cloudLabel[k] <= 0) {
                    surfaceCloudScan->push_back(extractedCloud->points[k]);
                }
            }
        }

        surfaceCloudScanDs->clear();
        downsizeFilter.setInputCloud(surfaceCloudScan);
        downsizeFilter.filter(*surfaceCloudScanDs);

        *surfaceCloud += *surfaceCloudScanDs;
    }
}

void LidarFeatureExtractionNode::freeCloudInfoMemory() {
    cloudInfo.start_ring_index.clear();
    cloudInfo.end_ring_index.clear();
    cloudInfo.point_col_ind.clear();
    cloudInfo.point_range.clear();
}

void LidarFeatureExtractionNode::publishFeatureCloud() {
    // free cloud info memory
    freeCloudInfoMemory();
    // save newly extracted features
    cloudInfo.cloud_corner = publishCloud<PointType>(pubCornerPoints, cornerCloud, cloudHeader.stamp, config->lidarFrame);
    cloudInfo.cloud_surface = publishCloud<PointType>(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, config->lidarFrame);
    // publish to mapOptimization
    pubLaserCloudInfo->publish(cloudInfo);
}
