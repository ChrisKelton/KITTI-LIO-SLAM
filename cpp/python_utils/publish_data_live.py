"""
Publish KITTI data as ROS2 topics in real time.

Topics published:
    imu_raw                 sensor_msgs/msg/Imu                 Implemented
    odometry/gps            nav_msgs/msg/Odometry               Implemented (ENU, for mapOptimization)
    points_raw              sensor_msgs/msg/PointCloud2         Implemented (x, y, z, intensity, time, ring)
    /camera/gray/left       sensor_msgs/msg/Image               Not yet implemented
    /camera/gray/right      sensor_msgs/msg/Image               Not yet implemented
    /camera/color/left      sensor_msgs/msg/Image               Not yet implemented
    /camera/color/right     sensor_msgs/msg/Image               Not yet implemented

    /ground_truth/pose      geometry_msgs/msg/PoseStamped       Not yet implemented
    /ground_truth/path      nav_msgs/msg/Path                   Not yet implemented
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from navpy import lla2ned
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from tqdm import tqdm

from inspect_data import OxtsDataTs, OxtsData

RAW_DATA_PATH: Path = Path("/home/ckelton/data/KITTI/raw_data")

# Velodyne HDL-64E geometry (KITTI): 64 beams spanning ~+2 deg (top) to ~-24.8 deg (bottom).
# Used to reconstruct the per-point ring index, which KITTI does not store in the .bin files.
N_SCAN: int = 64
HDL64_FOV_UP_DEG: float = 2.0
HDL64_FOV_DOWN_DEG: float = -24.8


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        self.elapsed_time = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed_time


# NavSatFix
#
# position_covariance = pos_accuracy^2 = sigma^2 = np.eye(3) * pos_accuracy^2

def to_ros_time_from_float(ts_sec: float) -> RosTime:
    t = RosTime()
    t.sec = int(ts_sec)
    t.nanosec = int((ts_sec - int(ts_sec))* 1e9)
    return t


def to_ros_time_from_pd(ts_sec: pd.Timestamp) -> RosTime:
    t = RosTime()
    # t.sec = int((ts_sec.value - ts_sec.nanosecond) / 1e3)  # bug: divides by 1e3 (µs) instead of 1e9 (ns→s)
    # t.nanosec = ts_sec.nanosecond  # bug: only sub-µs ns (0-999), not full sub-second ns
    t.sec = ts_sec.value // 1_000_000_000
    t.nanosec = ts_sec.value % 1_000_000_000
    return t


def get_singular_dir(dirs_: list[Path], raise_error: bool = False, error_desc: str = "") -> Optional[Path]:
    """
    If dir does not exist, dir is empty, or dir has more than 1 path, then 'None' will be returned
    :param dirs_:
    :param raise_error:
    :param error_desc:
    :return:
    """
    if len(dirs_) == 0:
        if raise_error:
            raise RuntimeError(f"No data in dir list{': ' + error_desc if error_desc != '' else ''}")
        return None
    if len(dirs_) > 1:
        if raise_error:
            raise RuntimeError(f"More than one data in dir list{': ' + error_desc if error_desc != '' else ''}")
        return None
    return dirs_[0]


class KittiPublisher(Node):
    CameraNameToTopicNameMap: dict[str, str] = {
        "image_00": "gray/left",
        "image_01": "gray/right",
        "image_02": "color/left",
        "image_03": "color/right",
    }

    def __init__(
        self,
        data_path: Path,
        data_to_publish: tuple[str, ...] = ("oxts", "feature_extraction", "image_00", "image_01", "image_02", "image_03", "gt"),
        speed: float = 1.0,
        use_sync: bool = False,  # detect if is_sync automatically
    ):
        super().__init__('kitti_publisher')
        self.speed = speed
        self.data_to_publish = [data_type.lower() for data_type in data_to_publish]
        data_to_publish = self.data_to_publish
        data_to_publish_cnt: int = 0

        self.data_path = data_path
        self.calib_path = sorted(data_path.glob("**/calib"))
        self.calib_path = get_singular_dir(self.calib_path, raise_error=True, error_desc=f"'{data_path}/**/calib'")
        if not self.calib_path.exists():
            raise RuntimeError(f"No calibration path exists at '{self.calib_path}'")
        self.sync_data_path = sorted(data_path.glob("**/sync"))
        self.sync_data_path = get_singular_dir(self.sync_data_path, raise_error=True, error_desc=f"'{data_path}/**/sync'")
        if not self.sync_data_path.exists():
            raise NotImplementedError(f"Synchronized data not found at '{self.sync_data_path}'")
        self.unsync_data_path = sorted(data_path.glob("**/unsync"))
        self.unsync_data_path = get_singular_dir(self.unsync_data_path, raise_error=True, error_desc=f"'{data_path}/**/unsync'")
        if not self.unsync_data_path.exists():
            raise NotImplementedError(f"Unsynchronized data not found at '{self.unsync_data_path}'")

        is_sync: bool = use_sync
        if use_sync:
            main_data_path = self.sync_data_path
        else:
            main_data_path = self.unsync_data_path

        if "oxts" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.oxts_dir = sorted(main_data_path.glob("**/oxts"))
            self.oxts_dir = get_singular_dir(self.oxts_dir, raise_error=True, error_desc=f"'{main_data_path}/**/oxts'")
            if not self.oxts_dir.exists():
                raise RuntimeError(f"OXTS data does not exist at '{self.oxts_dir}'")
        if "feature_extraction" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.velodyne_pts_dir = sorted(main_data_path.glob("**/velodyne_points"))
            self.velodyne_pts_dir = get_singular_dir(self.velodyne_pts_dir, raise_error=True, error_desc=f"'{main_data_path}/**/velodyne_points'")
            if not self.velodyne_pts_dir.exists():
                raise RuntimeError(f"Velodyne points data does not exist at '{self.velodyne_pts_dir}'")
        image_dirs: list[Path] = []
        if "image_00" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.gray_left_image_dir = sorted(main_data_path.glob("**/image_00"))
            self.gray_left_image_dir = get_singular_dir(self.gray_left_image_dir, raise_error=True, error_desc=f"'{main_data_path}/**/image_00'")
            if not self.gray_left_image_dir.exists():
                raise RuntimeError(f"Gray left image data does not exist at '{self.gray_left_image_dir}'")
            image_dirs.append(self.gray_left_image_dir)
        if "image_01" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.gray_right_image_dir = sorted(main_data_path.glob("**/image_01"))
            self.gray_right_image_dir = get_singular_dir(self.gray_right_image_dir, raise_error=True, error_desc=f"'{main_data_path}/**/image_00'")
            if not self.gray_right_image_dir.exists():
                raise RuntimeError(f"Gray right image data does not exist at '{self.gray_right_image_dir}'")
            image_dirs.append(self.gray_right_image_dir)
        if "image_02" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.color_left_image_dir = sorted(main_data_path.glob("**/image_02"))
            self.color_left_image_dir = get_singular_dir(self.color_left_image_dir, raise_error=True, error_desc=f"'{main_data_path}/**/image_00'")
            if not self.color_left_image_dir.exists():
                raise RuntimeError(f"Color left image data does not exist at '{self.color_left_image_dir}'")
            image_dirs.append(self.color_left_image_dir)
        if "image_03" in self.data_to_publish:
            data_to_publish_cnt += 1
            self.color_right_image_dir = sorted(main_data_path.glob("**/image_03"))
            self.color_right_image_dir = get_singular_dir(self.color_right_image_dir, raise_error=True, error_desc=f"'{main_data_path}/**/image_00'")
            if not self.color_right_image_dir.exists():
                raise RuntimeError(f"Color right image data does not exist at '{self.color_right_image_dir}'")
            image_dirs.append(self.color_right_image_dir)
        data_to_publish_cnt_gt = len(data_to_publish) - 1 if "gt" in self.data_to_publish else len(data_to_publish)
        if data_to_publish_cnt != data_to_publish_cnt:
            raise ValueError(f"Unidentified data type to publish in '{self.data_to_publish}' [Note: 'gt' is not included in check]. Processed '{data_to_publish_cnt}' / '{data_to_publish_cnt_gt}' requests.")

        self.image_dirs = image_dirs
        default_qos_profile_for_is_sync: int = 100
        # Data for VIO Pipeline ingestion
        # self.pub_imu = self.create_publisher(Imu, '/oxts/imu', default_qos_profile_for_is_sync if is_sync else 2000)

        # Published to imuPreintegration.cpp & imageProjection.cpp
        self.pub_imu = self.create_publisher(Imu, 'imu_raw', default_qos_profile_for_is_sync if is_sync else 2000)

        # Published to mapOptimization.cpp as nav_msgs/Odometry (the type its gpsHandler subscribes
        # to) on the gpsTopic configured in the launch file.
        self.pub_gps = self.create_publisher(Odometry, 'odometry/gps', default_qos_profile_for_is_sync if is_sync else 2000)

        # Camera frames are triggered when the laser scanner spins around and is facing forward
        # self.pub_pcl = self.create_publisher(PointCloud2, '/feature_extraction/data_pcl', default_qos_profile_for_is_sync if is_sync else 100)
        # Published to imageProjection.cpp
        self.pub_pcl = self.create_publisher(PointCloud2, 'points_raw', default_qos_profile_for_is_sync if is_sync else 100)

        # Not leveraged yet
        self.pub_gray_left_img = self.create_publisher(Image, '/image/gray/left', default_qos_profile_for_is_sync if is_sync else 100)
        self.pub_gray_right_img = self.create_publisher(Image, '/image/gray/right', default_qos_profile_for_is_sync if is_sync else 100)
        self.pub_color_left_img = self.create_publisher(Image, '/image/color/left', default_qos_profile_for_is_sync if is_sync else 100)
        self.pub_color_right_img = self.create_publisher(Image, '/image/color/right', default_qos_profile_for_is_sync if is_sync else 100)

        # Ground-Truth data
        self.pub_gt_pose = self.create_publisher(PoseStamped, '/ground_truth/pose', default_qos_profile_for_is_sync if is_sync else 100)
        self.pub_gt_path = self.create_publisher(NavPath, '/ground_truth/path', default_qos_profile_for_is_sync if is_sync else 10)

        self.timers_: dict[str, Timer] = {}
        self.elapsed_times: dict[str, list[float]] = {}
        for data_name in self.data_to_publish:
            self.timers_[data_name] = Timer()
            self.elapsed_times[data_name] = []

        self._load()
        self.oxts_origin = self.oxts_df.iloc[0]
        self.gps_origin_latlonalt = (self.oxts_origin['lat'], self.oxts_origin['lon'], self.oxts_origin['alt'])
        self.min_ts: int = 0  # Need this to reduce the time scale in between [-2**31, (2**31)-1]
        self._build_timeline()

        self.gt_path_msg = NavPath()
        # 'map' so the ground-truth path overlays the SLAM map in RViz (the SLAM map origin and the
        # ENU origin both sit at the first frame). 'world' was not in the published TF tree.
        self.gt_path_msg.header.frame_id = "map"

        self.idx = 0
        self.start_wall: float | None = None
        self.start_data: float | None = None

        # TODO: Update with the fastest ticking value
        # 50ms tick - tight enough to keep up with the 10Hz synchronized data
        self.timer = self.create_timer(0.05, self._tick)

        # Progress report every 5s
        self.progress_timer = self.create_timer(5.0, self._report_progress)

        self.get_logger().info(
            f"Ready: {len(self.oxts_df)} Oxts Data | {len(self.velodyne_ts)} Velodyne Point Clouds | "
            f"{len(self.images.get('image_00', []))} Grayscale Left Images | "
            f"{len(self.images.get('image_01', []))} Grayscale Right Images | "
            f"{len(self.images.get('image_02', []))} Color Left Images | "
            f"{len(self.images.get('image_03', []))} Color Right Images | speed = {self.speed}x"
        )

    def _load(self):
        self.get_logger().info('Loading KITTI data...')
        t_knot = time.time()

        self.oxts_df: pd.DataFrame = OxtsData.get_default_dataframe()
        oxts_dataformat_txt_path = sorted(self.oxts_dir.glob("**/dataformat.txt"))
        oxts_dataformat_txt_path = get_singular_dir(oxts_dataformat_txt_path, raise_error=True, error_desc=f"'{self.oxts_dir}/**/dataformat.txt'")
        with open(str(oxts_dataformat_txt_path), 'r') as f:
            data_keys = [line.strip("\n").split(":")[0] for line in f.readlines()]
        oxts_timestamps_txt_path = sorted(self.oxts_dir.glob("**/timestamps.txt"))
        oxts_timestamps_txt_path = get_singular_dir(oxts_timestamps_txt_path, raise_error=True, error_desc=f"'{self.oxts_dir}/**/timestamps.txt'")
        with open(str(oxts_timestamps_txt_path), 'r') as f:
            timestamps = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
        oxts_data_dir = sorted(self.oxts_dir.glob("**/data"))
        oxts_data_dir = get_singular_dir(oxts_data_dir, raise_error=True, error_desc=f"'{self.oxts_dir}/**/data'")
        for ts, data_txt_file in tqdm(zip(timestamps, sorted(oxts_data_dir.glob("**/*.txt"))), desc="Loading OXTS Data", total=len(timestamps)):
            oxts_data = OxtsDataTs.from_txt_file(data_txt_file, data_keys, ts)
            self.oxts_df.loc[len(self.oxts_df)] = oxts_data.to_pandas_series()
        self.oxts_df.sort_values(by=["ts"], inplace=True)
        self.get_logger().info(f"Time to load OXTS data '{(time.time() - t_knot):.3f}s'")

        # Velodyne data
        t0 = time.time()
        self.velodyne_data: list[Path] = []
        self.velodyne_ts: list[pd.Timestamp] = []
        # Per-scan sweep start/end times, used to reconstruct per-point timestamps for deskewing.
        self.velodyne_ts_start: list[pd.Timestamp] = []
        self.velodyne_ts_end: list[pd.Timestamp] = []
        velodyne_ts_txt_path = sorted(self.velodyne_pts_dir.glob("**/timestamps.txt"))
        velodyne_ts_txt_path = get_singular_dir(velodyne_ts_txt_path, raise_error=True, error_desc=f"'{self.velodyne_pts_dir}/**/timestamps.txt")
        with open(str(velodyne_ts_txt_path), 'r') as f:
            timestamps = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
        velodyne_ts_start_txt_path = sorted(self.velodyne_pts_dir.glob("**/timestamps_start.txt"))
        velodyne_ts_start_txt_path = get_singular_dir(velodyne_ts_start_txt_path, raise_error=True, error_desc=f"'{self.velodyne_pts_dir}/**/timestamps_start.txt'")
        with open(str(velodyne_ts_start_txt_path), 'r') as f:
            timestamps_start = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
        velodyne_ts_end_txt_path = sorted(self.velodyne_pts_dir.glob("**/timestamps_end.txt"))
        velodyne_ts_end_txt_path = get_singular_dir(velodyne_ts_end_txt_path, raise_error=True, error_desc=f"'{self.velodyne_pts_dir}/**/timestamps_end.txt'")
        with open(str(velodyne_ts_end_txt_path), 'r') as f:
            timestamps_end = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
        velodyne_data_dir = sorted(self.velodyne_pts_dir.glob("**/data"))
        velodyne_data_dir = get_singular_dir(velodyne_data_dir, raise_error=True, error_desc=f"'{self.velodyne_pts_dir}/**/data'")
        # 'sync' KITTI stores velodyne as float32 .bin; 'extract'/unsync stores it as text .txt.
        velodyne_files = sorted(velodyne_data_dir.glob("**/*.bin"))
        if not velodyne_files:
            velodyne_files = sorted(velodyne_data_dir.glob("**/*.txt"))
        # `timestamps` is the camera-trigger time (mid-sweep); start/end bracket the full sweep and
        # are used to assign per-point times for deskewing.
        for ts, ts_start, ts_end, velo_file in tqdm(zip(timestamps, timestamps_start, timestamps_end, velodyne_files), desc="Loading Velodyne Point Clouds", total=len(timestamps)):
            self.velodyne_data.append(velo_file)
            self.velodyne_ts.append(ts)
            self.velodyne_ts_start.append(ts_start)
            self.velodyne_ts_end.append(ts_end)
        self.get_logger().info(f"Time to load Velodyne data '{(time.time() - t0):.3f}s'")

        # Images
        self.images: dict[str, list[np.ndarray]] = {}
        self.images_ts: dict[str, list[pd.Timestamp]] = {}
        for image_dir in self.image_dirs:
            t0 = time.time()
            image_ts_txt_path = sorted(image_dir.glob("**/timestamps.txt"))
            image_ts_txt_path = get_singular_dir(image_ts_txt_path, raise_error=True, error_desc=f"'{image_dir}/**/timestamps.txt'")
            with open(str(image_ts_txt_path), 'r') as f:
                timestamps = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
            image_data_dir = sorted(image_dir.glob("**/data"))
            image_data_dir = get_singular_dir(image_data_dir, raise_error=True, error_desc=f"'{image_dir}/**/data'")
            camera_name = image_dir.stem
            self.images.setdefault(camera_name, [])
            self.images_ts.setdefault(camera_name, [])
            for ts, image_path in tqdm(zip(timestamps, sorted(image_data_dir.glob("**/*.png"))), desc=f"Loading images for camera '{camera_name}'", total=len(timestamps)):
                if camera_name in ["image_00", "image_01"]:
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                elif camera_name in ["image_02", "image_03"]:
                    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                else:
                    raise NotImplementedError(f"Unrecognized camera '{camera_name}'")
                self.images[camera_name].append(image)
                self.images_ts[camera_name].append(ts)
            self.get_logger().info(f"Time to load camera data from '{camera_name}': '{(time.time() - t0):.3f}s'")

        self.get_logger().info(f"Total time to load all data: '{(time.time() - t_knot):.3f}s'")

    def _build_timeline(self):
        events = (
            [(self.oxts_df.iloc[i]["ts"], 'oxts', i) for i in range(len(self.oxts_df))] +
            [(ts.value, 'feature_extraction', i) for i, ts in enumerate(self.velodyne_ts)] +
            [(ts.value, 'image_00', i) for i, ts in enumerate(self.images_ts.get("image_00", []))] +
            [(ts.value, 'image_01', i) for i, ts in enumerate(self.images_ts.get("image_01", []))] +
            [(ts.value, 'image_02', i) for i, ts in enumerate(self.images_ts.get("image_02", []))] +
            [(ts.value, 'image_03', i) for i, ts in enumerate(self.images_ts.get("image_03", []))]
        )
        if "gt" in self.data_to_publish:
            events += [(self.oxts_df.iloc[i]["ts"], 'gt', i) for i in range(len(self.oxts_df))]
        events.sort(key=lambda e: e[0])
        self.min_ts = events[0][0]
        # Velodyne messages are stamped at their sweep start, which is earlier than the trigger
        # time used in the timeline. Fold those into min_ts so published stamps stay non-negative.
        if self.velodyne_ts_start:
            self.min_ts = min(self.min_ts, min(ts.value for ts in self.velodyne_ts_start))
        self.timeline = events
        self.get_logger().info(f"Timeline built: {len(self.timeline)} events")

    def _tick(self) -> None:
        if self.idx >= len(self.timeline):
            self.get_logger().info("Finished - all events published")
            self.timer.cancel()
            self.progress_timer.cancel()
            return

        now = time.time_ns()
        if self.start_wall is None:
            self.start_wall = now
            self.start_data = self.timeline[0][0]

        # Drain all events that should have fired by now
        while self.idx < len(self.timeline):
            ts, kind, i = self.timeline[self.idx]
            data_elapsed = (ts - self.start_data) / self.speed
            wall_elapsed = now - self.start_wall
            if data_elapsed > wall_elapsed:
                break

            stamp = to_ros_time_from_pd(pd.Timestamp(ts - self.min_ts))

            self.timers_[kind].start()
            if kind == "oxts":
                self._pub_oxts(i, stamp)
            elif kind == "feature_extraction":
                self._pub_velodyne(i, stamp)
            elif "image" in kind.lower():
                self._pub_image(i, stamp, camera_name=kind)
            else:
                self._pub_gt(i, stamp)
            self.elapsed_times[kind].append(self.timers_[kind].stop())

            self.idx += 1

    def _report_progress(self):
        if self.idx == 0 or self.start_data is None:
            return
        pct = 100.0 * self.idx / len(self.timeline)
        ts_now = self.timeline[min(self.idx, len(self.timeline) - 1)][0]
        data_elapsed = ts_now - self.start_data
        # wall_elapsed = time.time() - self.start_wall
        wall_elapsed = time.time_ns() - self.start_wall
        total_data = self.timeline[-1][0] - self.start_data
        remaining_data = total_data - data_elapsed
        # eta = remaining_data / self.speed - wall_elapsed + (time.time() - self.start_wall)
        eta = remaining_data / self.speed - wall_elapsed + (time.time_ns() - self.start_wall)
        self.get_logger().info(
            f"Progress: {pct:.1f}%   ({self.idx}/{len(self.timeline)} events)    "
            f"ETA: {eta:.1f}s"
        )
        self.get_logger().info("Mean elapsed times to process data types:")
        for data_type, times in self.elapsed_times.items():
            mean_time = float(np.mean(times))
            self.get_logger().info(f"\t{data_type}: {(mean_time * 1e3):.2f}ms / {(1 / mean_time):.2f}Hz")

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _pub_oxts(self, idx: int, stamp: RosTime):
        series_ = self.oxts_df.iloc[idx]

        ax, ay, az = series_["ax"], series_["ay"], series_["az"]
        wx, wy, wz = series_["wx"], series_["wy"], series_["wz"]
        # Get rotation matrix from (roll, pitch, yaw) data to get orientation of IMU
        Rx = self.rotx(series_["roll"])
        Ry = self.roty(series_["pitch"])
        Rz = self.rotz(series_["yaw"])
        R_ = Rz.dot(Ry.dot(Rx))
        quaternion = R.from_matrix(R_).as_quat()
        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = "imu"
        imu_msg.orientation.x = quaternion[0]
        imu_msg.orientation.y = quaternion[1]
        imu_msg.orientation.z = quaternion[2]
        imu_msg.orientation.w = quaternion[3]
        imu_msg.angular_velocity.x = wx
        imu_msg.angular_velocity.y = wy
        imu_msg.angular_velocity.z = wz
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        # GPS as nav_msgs/Odometry to match mapOptimization's gpsHandler, which reads
        # pose.pose.position and the x/y/z variances on the diagonal of pose.covariance.
        # lat/lon/alt are converted to a local ENU position relative to the first fix, matching
        # the SLAM map origin.
        lat, lon, alt = series_["lat"], series_["lon"], series_["alt"]
        pos_accuracy = series_["pos_accuracy"]
        ned_x, ned_y, ned_z = lla2ned(lat, lon, alt, *self.gps_origin_latlonalt, latlon_unit='deg', alt_unit='m', model='wgs84')
        # ROS is ENU; KITTI lla2ned gives NED.
        enu_x, enu_y, enu_z = ned_y, ned_x, -ned_z

        gps_msg = Odometry()
        gps_msg.header.stamp = stamp
        gps_msg.header.frame_id = "odom"
        gps_msg.child_frame_id = "gps"
        gps_msg.pose.pose.position.x = float(enu_x)
        gps_msg.pose.pose.position.y = float(enu_y)
        gps_msg.pose.pose.position.z = float(enu_z)
        gps_msg.pose.pose.orientation.w = 1.0  # orientation unused by the GPS factor; keep it valid
        # 6x6 row-major covariance; the consumer reads indices 0 (x), 7 (y), 14 (z).
        covariance = [0.0] * 36
        covariance[0] = float(pos_accuracy ** 2)
        covariance[7] = float(pos_accuracy ** 2)
        covariance[14] = float(pos_accuracy ** 2)
        gps_msg.pose.covariance = covariance

        self.pub_imu.publish(imu_msg)
        self.pub_gps.publish(gps_msg)


    @staticmethod
    def _read_velodyne(path: Path) -> np.ndarray:
        # Both formats store rows of [x, y, z, reflectance]. 'sync' is float32 binary;
        # 'extract'/unsync is whitespace-separated text.
        if path.suffix == ".bin":
            return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return np.loadtxt(str(path), dtype=np.float32).reshape(-1, 4)

    def _pub_velodyne(self, idx: int, stamp: RosTime):
        points = self._read_velodyne(self.velodyne_data[idx])

        # KITTI .bin files carry no per-point timestamp, so reconstruct it from the azimuth angle
        # and the sweep's start/end times. This is the standard KITTI convention (cf. KISS-ICP):
        # yaw = -atan2(y, x), normalized to [0, 1] over one revolution, scaled by the sweep
        # duration. t = 0 at sweep start (rear of the vehicle), t = dt at sweep end.
        ts_start = self.velodyne_ts_start[idx]
        ts_end = self.velodyne_ts_end[idx]
        dt_scan = max((ts_end.value - ts_start.value) * 1e-9, 1e-6)  # seconds (~0.1 at 10 Hz)
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        frac = np.clip(0.5 * (yaw / np.pi + 1.0), 0.0, 1.0)
        point_time = (frac * dt_scan).astype(np.float32)

        # Reconstruct the ring (laser/beam index; 0 = bottom beam) from the elevation angle, since
        # KITTI .bin files carry no ring channel. HDL-64E beams are only approximately uniform in
        # elevation, so this uniform binning can occasionally collide adjacent beams; a measured
        # per-beam angle table would be more precise.
        depth_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        elev_deg = np.degrees(np.arctan2(points[:, 2], depth_xy))
        fov = HDL64_FOV_UP_DEG - HDL64_FOV_DOWN_DEG
        ring = np.round((elev_deg - HDL64_FOV_DOWN_DEG) / fov * (N_SCAN - 1))
        ring = np.clip(ring, 0, N_SCAN - 1).astype(np.uint16)

        # Sort by time so the consumer's "last point == scan end" assumption holds (it derives the
        # scan end time from the final point) and the earliest point defines the deskew reference.
        order = np.argsort(point_time, kind="stable")
        xyzi = points[order]
        point_time = point_time[order]
        ring = ring[order]

        # Pack as x, y, z, intensity, time (float32) + ring (uint16) -> 22 bytes/point. The C++
        # consumer reads each field by name/offset, so this mixed-type layout is fine.
        point_dtype = np.dtype({
            'names':   ['x', 'y', 'z', 'intensity', 'time', 'ring'],
            'formats': ['<f4', '<f4', '<f4', '<f4', '<f4', '<u2'],
            'offsets': [0, 4, 8, 12, 16, 20],
            'itemsize': 22,
        })
        cloud = np.empty(xyzi.shape[0], dtype=point_dtype)
        cloud['x'] = xyzi[:, 0]
        cloud['y'] = xyzi[:, 1]
        cloud['z'] = xyzi[:, 2]
        cloud['intensity'] = xyzi[:, 3]
        cloud['time'] = point_time
        cloud['ring'] = ring

        msg = PointCloud2()
        # Stamp at the sweep start: imageProjection treats header.stamp as the scan start, and the
        # per-point `time` values above are offsets relative to it.
        msg.header.stamp = to_ros_time_from_pd(pd.Timestamp(ts_start.value - self.min_ts))
        msg.header.frame_id = 'feature_extraction'
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 22
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud.tobytes()
        msg.is_dense = True
        self.pub_pcl.publish(msg)

    def _pub_image(self, idx: int, stamp: RosTime, camera_name: str):
        image = self.images[camera_name][idx]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = camera_name
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.step = image.shape[1]
        msg.data = image.tobytes()
        if camera_name in ["image_00", "image_01"]:
            msg.encoding = "mono8"
        else:
            msg.encoding = "rgb8"
        if camera_name == "image_00":
            self.pub_gray_left_img.publish(msg)
        elif camera_name == "image_01":
            self.pub_gray_left_img.publish(msg)
        elif camera_name == "image_02":
            self.pub_gray_left_img.publish(msg)
        else:
            self.pub_gray_left_img.publish(msg)

    def _pub_gt(self, idx: int, stamp: RosTime):
        series_ = self.oxts_df.iloc[idx]

        lat, lon, alt = series_["lat"], series_["lon"], series_["alt"]
        ned_x, ned_y, ned_z = lla2ned(lat, lon, alt, *self.gps_origin_latlonalt, latlon_unit='deg', alt_unit='m', model='wgs84')
        # ROS operates in ENU (East-North-Up) Coordinate Frame, so change from NED (North-East-Down).
        enu_x, enu_y, enu_z = ned_y, ned_x, -ned_z

        # Get orientation
        Rx = self.rotx(series_["roll"])
        Ry = self.roty(series_["pitch"])
        Rz = self.rotz(series_["yaw"])
        R_ = Rz.dot(Ry.dot(Rx))
        quaternion = R.from_matrix(R_).as_quat()

        # Create PoseStamped ROS message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = enu_x
        pose_msg.pose.position.y = enu_y
        pose_msg.pose.position.z = enu_z
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.pub_gt_pose.publish(pose_msg)

        self.gt_path_msg.header.stamp = stamp
        self.gt_path_msg.poses.append(pose_msg)
        self.pub_gt_path.publish(self.gt_path_msg)


def main():
    parser = argparse.ArgumentParser(description="Publish KITTI data as ROS2 topics")
    parser.add_argument(
        "--data",
        default=str(RAW_DATA_PATH),
        required=False,
        help="Path to raw KITTI dataset with subfolders: '/calib', '/sync', '/tracklets' (/tracklets not enforced for now)",
    )
    # TODO: Add "--gt" (if using unsynchronized data)
    parser.add_argument(
        "--speed",
        default=1.0,
        type=float,
        required=False,
        help="Playback speed multiplier",
    )
    args = parser.parse_args()

    data_to_publish = ("oxts", "feature_extraction")
    rclpy.init()
    node = KittiPublisher(data_path=Path(args.data), speed=args.speed, data_to_publish=data_to_publish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
