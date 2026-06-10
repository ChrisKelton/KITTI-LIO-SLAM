Project to utilize KITTI raw_data to leverage grayscale/color images, lidar, GPS/IMU unit for SLAM and possibly extend to 3D object detection & tracking.


(Mono Grayscale Image + Lidar + GPS/IMU)
Architecture:
    1. High-Frequency State Propagation (IMU):
        - Pre-integration: IMU measurements (angular velocity, acceleration) are processed between consecutive image/
                            LiDAR frames to calculate relative motion constraints.
        - Bias Correction: The system continuously estimates and corrects IMU biases (accelerometer and gyroscope) by
                            using the slower but more accurate LiDAR/Vision poses as ground truth anchors.
        - Output: This provides smooth, high-rate pose predictions to handle rapid, aggressive movements.
    2. LiDAR Odometry (LO):
        - Feature Extraction: Extract geometric features such as edge points (from sharp corners/poles) and planar
                                patches (from walls, ceilings, or floors).
        - Registration: Perform point cloud registration (e.g., using Generalized ICP or Normal Distributions Transform)
                        between sequential scans to compute rigid-body transformations.
        - Drift Suppression: The dense point cloud provides metric scale and prevents the slow drift commonly seen in
                                pure visual odometry.
    3. Visual Odometry (VO) & Depth Association:
        - Feature Tracking: Extract and track visual features (e.g., ORB, FAST) across consecutive grayscale frames.
        - Depth Fusion: Project the 3D LiDAR point clouds into the camera plane to assign precise depth values to the
                            2D visual feature points.
        - Pose Estimation: Calculate camera ego-motion by minimizing the re-projection error of these 3D-2D point
                            correspondences.
    4. Global Optimization & GPS Integration:
        - Factor Graph Formulation: Construct a unified factor graph that ties all constraints together: IMU
                                    pre-integration, visual reprojection errors, and LiDAR point-to-plane/point-to-line
                                    distances.
        - GPS Constraints: Incorporate GPS position factors as global nodes. This provides absolute coordinates and
                            prevents accumulated drift over long trajectories.
        - Loop Closure: Use visual bag-of-words (e.g., DBoW2) or LiDAR place recognition (e.g., ScanContext) to detect
                            previously visited locations. Adding loop closure factors triggers graph optimization to
                            correct the entire trajectory globally.
