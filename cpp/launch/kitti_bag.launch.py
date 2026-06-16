import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


WS_ROOT: str = str(Path(__file__).resolve().parent.parent)
SPLIT_TYPE: str = "training"  # ['training', 'testing']


# ---------------------------------------------------------------------------
# Parameters tuned for KITTI raw dataset 2011_09_26_drive_0022 (Velodyne HDL-64E
# + OXTS RT3003 INS). Structure / non-sensor defaults follow LIO-SAM
# (TixiaoShan/LIO-SAM, config/params.yaml); sensor specs and extrinsics are KITTI.
#
# Parameter *keys* match the strings each node passes to declare_parameter()
# (e.g. extRotV / extRotRPYV / extTransV, z_tolerance), which differ from the
# names in LIO-SAM's yaml (extrinsicRot, z_tollerance, ...).
#
# Sensor specs: Velodyne HDL-64E -> 64 rings; Horizon_SCAN/range match the C++
# node defaults the repo already uses for KITTI.
# ---------------------------------------------------------------------------
N_SCAN = 64
HORIZON_SCAN = 4096
DOWNSAMPLE_RATE = 1
LIDAR_MIN_RANGE = 1.0
LIDAR_MAX_RANGE = 80.0  # KITTI ground truth is capped near 80 m; raise toward ~120 for full HDL-64E range

# IMU -> LiDAR extrinsics from calib/2011_09_26/calib_imu_to_velo.txt
# (p_velo = R * p_imu + T). KITTI's OXTS and Velodyne axes are nearly aligned,
# so R is close to identity. extRPY uses the same R since the OXTS orientation is
# reported in the IMU frame.
EXTRINSICS = {
    'extRotV': [
         0.9999976,    0.0007553071, -0.002035826,
        -0.0007854027, 0.9998898,    -0.01482298,
         0.002024406,  0.01482454,    0.9998881,
    ],
    'extRotRPYV': [
         0.9999976,    0.0007553071, -0.002035826,
        -0.0007854027, 0.9998898,    -0.01482298,
         0.002024406,  0.01482454,    0.9998881,
    ],
    'extTransV': [-0.8086759, 0.3195559, -0.7997231],
}

# Must be identical in image_projection and feature_extraction: it sets the
# curvature neighbor window and therefore the ring start/end index padding.
CURVATURE_NEIGHBORS = 5

# Deskew / range-image projection: subscribes points_raw + imu + odom, publishes
# lio_sam/deskew/cloud_info and the deskewed cloud.
IMAGE_PROJECTION_PARAMS = {
    'lidarFrame': 'base_link',
    'imuTopic': 'imu_raw',
    'odomTopic': 'odometry/imu',
    'pointCloudTopic': 'points_raw',
    'N_SCAN': N_SCAN,
    'Horizon_SCAN': HORIZON_SCAN,
    'downsampleRate': DOWNSAMPLE_RATE,
    'lidarMinRange': LIDAR_MIN_RANGE,
    'lidarMaxRange': LIDAR_MAX_RANGE,
    'extRotV': EXTRINSICS['extRotV'],
    'extRotRPYV': EXTRINSICS['extRotRPYV'],
    'lidarCurvatureFeatureExtractionNeighbors': CURVATURE_NEIGHBORS,
}

# LOAM feature extraction: subscribes the deskewed cloud_info, publishes corner /
# surface features. surfLeafSize maps to LIO-SAM's odometrySurfLeafSize.
FEATURE_EXTRACTION_PARAMS = {
    'lidarFrame': 'base_link',
    'N_SCAN': N_SCAN,
    'Horizon_SCAN': HORIZON_SCAN,
    'downsampleRate': DOWNSAMPLE_RATE,
    'lidarMinRange': LIDAR_MIN_RANGE,
    'lidarMaxRange': LIDAR_MAX_RANGE,
    'surfLeafSize': 0.4,
    'edgeThreshold': 1.0,
    'surfThreshold': 0.1,
    'lidarCurvatureFeatureExtractionNeighbors': CURVATURE_NEIGHBORS,
    'pixelDiffTh': 10,
}

# The `imu` executable spins TWO nodes (ImuPreintegrationNode + TransformFusionNode),
# so this dict carries the parameters for both. launch_ros applies a dict to every
# node in the process via the `/**` wildcard, so each node picks up the keys it declares.
IMU_PARAMS = {
    # --- ImuPreintegrationNode ---
    'imuTopic': 'imu_raw',
    'odomTopic': 'odometry/imu',
    'odometryFrame': 'odom',
    'imuAccNoise': 3.9939570888238808e-03,
    'imuGyrNoise': 1.5636343949698187e-03,
    'imuAccBiasN': 6.4356659353532566e-05,
    'imuGyrBiasN': 3.5640318696367613e-05,
    'imuGravity': 9.80511,
    'extRotV': EXTRINSICS['extRotV'],
    'extRotRPYV': EXTRINSICS['extRotRPYV'],
    'extTransV': EXTRINSICS['extTransV'],
    'reset_graph_every_n_keyframes': 100,
    # --- TransformFusionNode ---
    'lidarFrame': 'base_link',
    'baselinkFrame': 'base_link',
    'mapFrame': 'map',
}

# iSAM2 back-end: scan-to-map optimization, key-frame factor graph, loop closure,
# map save service. gpsTopic maps to LIO-SAM's "odometry/gpsz".
MAP_OPTIMIZATION_PARAMS = {
    # scan2MapOptimization dominates per-frame cost (~98%); its OpenMP loops scale
    # near-linearly. Machine has 16 cores, so use 12 and leave headroom for the
    # other nodes.
    'numberOfCores': 12,
    # Use the GTSAM LevenbergMarquardtOptimizer scan-matching back-end (vs the hand-rolled solver).
    'useGtsamScanMatcher': True,
    # NOTE: publish_data.py emits NavSatFix on /oxts/gps, but the GPS handler here
    # subscribes to nav_msgs/Odometry. The GPS factor will not fire until a
    # navsat_transform-style bridge republishes GPS as Odometry on this topic.
    'gpsTopic': 'odometry/gps',
    'savePCD': False,
    'savePCDDirectory': '/Downloads/LOAM/',
    'odometryFrame': 'odom',
    'lidarFrame': 'base_link',
    # voxel filter sizes
    'mappingCornerLeafSize': 0.2,
    'mappingSurfLeafSize': 0.4,
    'mappingProcessInterval': 0.15,
    'N_SCAN': N_SCAN,
    'Horizon_SCAN': HORIZON_SCAN,
    # global map visualization
    'globalMapVisualizationSearchRadius': 1000.0,
    'globalMapVisualizationPoseDensity': 10.0,
    'globalMapVisualizationLeafSize': 1.0,
    # surrounding map
    'surroundingKeyframeAddingDistThreshold': 1.0,
    'surroundingKeyframeAddingAngleThreshold': 0.2,
    'surroundingKeyframeDensity': 2.0,
    'surroundingKeyframeSearchRadius': 50.0,
    # loop closure
    'loopClosureEnableFlag': True,
    'loopClosureFrequency': 1.0,
    'surroundingKeyframeSize': 50,
    'historyKeyframeSearchRadius': 15.0,
    'historyKeyframeSearchTimeDiff': 30.0,
    'historyKeyframeSearchNum': 25,
    'historyKeyframeFitnessScore': 0.3,
    # GPS
    'useImuHeadingInitialization': True,
    'useGpsElevation': False,
    'gpsCovThreshold': 2.0,
    'poseCovThreshold': 25.0,
    # LOAM feature thresholds
    'edgeFeatureMinValidNum': 10,
    'surfFeatureMinValidNum': 100,
    # robot motion constraints (LIO-SAM z_tollerance / rotation_tollerance)
    'imuRPYWeight': 0.01,
    'z_tolerance': 1000.0,
    'rotation_tolerance': 1000.0,
}


def generate_launch_description():
    bag_file = str(Path(WS_ROOT) / f"data/raw_data_bag")

    # Node names are omitted so the `/**` parameter wildcard applies cleanly; in
    # particular the imu process hosts two nodes and must not be force-renamed.
    image_projection_node = Node(
        package='image_projection',
        executable='image_projection',
        parameters=[IMAGE_PROJECTION_PARAMS],
        output='screen',
    )

    feature_extraction_node = Node(
        package='feature_extraction',
        executable='feature_extraction',
        parameters=[FEATURE_EXTRACTION_PARAMS],
        output='screen',
    )

    imu_node = Node(
        package='imu',
        executable='imu',
        parameters=[IMU_PARAMS],
        output='screen',
    )

    map_optimization_node = Node(
        package='map_optimization',
        executable='map_optimization',
        parameters=[MAP_OPTIMIZATION_PARAMS],
        output='screen',
    )

    # NOTE: the previous `world -> feature_extraction` static transform was a
    # leftover (its child is a node name, not a frame). LIO-SAM publishes map->odom
    # and odom->base_link dynamically from the imu/map_optimization nodes, so no
    # static world transform is needed. Re-enable and fix the frames if required.
    # world_tf_node = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='world_tf',
    #     arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
    #     output='log',
    # )



    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(WS_ROOT, 'rviz', 'kitti.rviz')],
        output='log',
        additional_env={'QT_QPA_PLATFORM': 'xcb'},
    )

    return LaunchDescription([
        image_projection_node,
        feature_extraction_node,
        imu_node,
        map_optimization_node,
        # world_tf_node,
        rviz_node,
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_file, "--loop", "-r", "0.1", "--clock"],
            output="screen"
        )
    ])
