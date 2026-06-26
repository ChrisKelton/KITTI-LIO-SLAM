"""
Write KITTI raw data into a rosbag2 bag for offline playback with the SLAM pipeline.

Instead of publishing live (in parallel with the launched nodes), this serializes every
message into a bag once. Replay it with:

    ros2 bag play <output_dir>

while the SLAM launch file is running. The bag records exactly the topics the pipeline
consumes (plus optional ground truth):

    /imu_raw          sensor_msgs/msg/Imu          (frame: imu)
    /points_raw       sensor_msgs/msg/PointCloud2  (x, y, z, intensity, time, ring; frame: feature_extraction)
    /odometry/gps     nav_msgs/msg/Odometry        (ENU position for mapOptimization's GPS factor)
    /ground_truth/pose  geometry_msgs/msg/PoseStamped  (optional, --no-ground-truth to skip)
    /ground_truth/path  nav_msgs/msg/Path              (optional, written once at the end)

Sensor: Velodyne HDL-64E. KITTI .bin/.txt store only x,y,z,reflectance, so per-point
`time` (from the sweep start/end timestamps) and `ring` (from the elevation angle) are
reconstructed here, matching what imageProjection expects.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rosbag2_py
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from navpy import lla2ned
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu, PointCloud2, PointField
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from tqdm import tqdm

from inspect_data import OxtsDataTs, OxtsData
from inspect_tracking_data import extract_tracking_labels, load_calib_txt_file, convert_labels_from_cam_to_velo_coordinates

TRACKING_DATA_PATH: Path = Path("/home/ckelton/data/KITTI/tracking")
OUTPUT_BAG_PATH: Path = Path(f"/home/ckelton/repos/misc/KITTI-LIO-SLAM/cpp/data")

# Velodyne HDL-64E geometry (KITTI): 64 beams spanning ~+2 deg (top) to ~-24.8 deg (bottom).
# Used to reconstruct the per-point ring index, which KITTI does not store in the point files.
N_SCAN: int = 64
HDL64_FOV_UP_DEG: float = 2.0
HDL64_FOV_DOWN_DEG: float = -24.8

# Fully-resolved topic names the SLAM nodes subscribe to.
TOPIC_IMU: str = "/imu_raw"
TOPIC_POINTS: str = "/points_raw"
TOPIC_GPS: str = "/odometry/gps"
TOPIC_GT_POSE: str = "/ground_truth/pose"
TOPIC_GT_PATH: str = "/ground_truth/path"
TOPIC_DET3D: str = "/detections3d"

# Frame the ground truth path/pose is expressed in (the SLAM map frame, so it overlays in RViz).
GT_FRAME: str = "map"
LABEL_CLASSES: list[str] = ["car", "pedestrian"]
LABEL_CLASSES = [str_.lower() for str_ in LABEL_CLASSES]


def to_ros_time_from_pd(ts_sec: pd.Timestamp) -> RosTime:
    t = RosTime()
    t.sec = ts_sec.value // 1_000_000_000
    t.nanosec = ts_sec.value % 1_000_000_000
    return t


def get_singular_dir(dirs_: list[Path], raise_error: bool = False, error_desc: str = "") -> Optional[Path]:
    """Return the only directory in the list, else None (or raise)."""
    if len(dirs_) == 0:
        if raise_error:
            raise RuntimeError(f"No data in dir list{': ' + error_desc if error_desc != '' else ''}")
        return None
    if len(dirs_) > 1:
        if raise_error:
            raise RuntimeError(f"More than one data in dir list{': ' + error_desc if error_desc != '' else ''}")
        return None
    return dirs_[0]


class KittiBagWriter:
    OXTS_DATA_KIND: str = "oxts"
    VELO_DATA_KIND: str = "velodyne"
    LABELS_DATA_KIND: str = "labels"
    DATA_KINDS: list[str] = [OXTS_DATA_KIND, VELO_DATA_KIND]

    def __init__(
        self,
        data_path: Path,
        write_ground_truth: bool = True,
        data_to_skip: Optional[str | list[str]] = None,
        split: str = "training",
        sequence_number: int = 0,
        msg_frequency_hz: float = 10.0,
    ):
        if split not in ["training", "testing"]:
            raise ValueError(f"'split' must be in ['training', 'testing']. Got '{split}'")
        self.split = split
        if sequence_number < 0 or sequence_number > 20:
            raise ValueError(f"'sequence_number' must be in [0, 20]. Got '{sequence_number}'")
        self.sequence = sequence_number
        self.msg_frequency_hz = msg_frequency_hz

        self.data_path = data_path
        self.write_ground_truth = write_ground_truth

        if data_to_skip is not None:
            if isinstance(data_to_skip, str):
                data_to_skip = [data_to_skip]
            data_to_skip = [data_.lower() for data_ in data_to_skip]
            for data_ in data_to_skip:
                if data_ not in self.DATA_KINDS:
                    raise ValueError(f"Unrecognized data kind in data_to_skip: '{data_}'. Please choose from: '{self.DATA_KINDS}'")
        else:
            data_to_skip = []
        self.data_to_skip = data_to_skip

        main_data_path = data_path
        self.oxts_dir = get_singular_dir(
            sorted(main_data_path.glob(f"**/oxts/{split}")),
            raise_error=True if self.use_oxts_data or self.write_ground_truth else False,
            error_desc=f"'{main_data_path}/**/oxts/{split}'",
        )
        self.velodyne_pts_dir = None
        if self.use_velo_data:
            self.velodyne_pts_dir = get_singular_dir(
                sorted(main_data_path.glob(f"**/velodyne/{split}")), raise_error=True,
                error_desc=f"'{main_data_path}/**/velodyne/{split}'")
        self.labels_dir = None
        if self.use_labels_data:
            self.labels_dir = get_singular_dir(
                sorted(main_data_path.glob(f"**/labels/{split}")), raise_error=True,
                       error_desc=f"'{main_data_path}/**/labels/{split}'")
        self.calib_dir = get_singular_dir(sorted(main_data_path.glob(f"**/calib/{split}")), raise_error=False)

        self._load()
        if self.oxts_df is not None:
            self.gps_origin_latlonalt = (
                self.oxts_df.iloc[0]["lat"], self.oxts_df.iloc[0]["lon"], self.oxts_df.iloc[0]["alt"])
        else:
            self.gps_origin_latlonalt = (0.0, 0.0, 0.0)
        self._build_timeline()

        # Accumulated ground-truth path, written once at the end (writing the growing path every
        # step would bloat the bag quadratically).
        self._gt_path = NavPath()
        self._gt_path.header.frame_id = GT_FRAME

    @property
    def use_oxts_data(self) -> bool:
        return self.OXTS_DATA_KIND not in self.data_to_skip

    @property
    def use_velo_data(self) -> bool:
        return self.VELO_DATA_KIND not in self.data_to_skip

    @property
    def use_labels_data(self) -> bool:
        return self.LABELS_DATA_KIND not in self.data_to_skip

    @property
    def sequence_str(self) -> str:
        return f"{self.sequence:04d}"

    @property
    def msg_frequency_s(self) -> float:
        return 1.0 / self.msg_frequency_hz

    @staticmethod
    def generate_spaced_timestamps(range_: int, frequency_s: float) -> list[pd.Timestamp]:
        return [pd.Timestamp(idx * frequency_s, unit='s') for idx in range(range_)]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        print("Loading KITTI data...")

        use_artificial_ts: bool = False
        self.oxts_df = None
        if self.write_ground_truth or self.VELO_DATA_KIND not in self.data_to_skip:
            # OXTS (IMU + GPS + ground truth)
            self.oxts_df: pd.DataFrame = OxtsData.get_default_dataframe()
            dataformat_path = get_singular_dir(sorted(self.oxts_dir.glob("**/dataformat.txt")), raise_error=True)
            with open(str(dataformat_path), "r") as f:
                data_keys = [line.strip("\n").split(":")[0] for line in f.readlines()]

            oxts_data_dir = get_singular_dir(sorted(self.oxts_dir.glob("**/oxts")), raise_error=True)
            oxts_sequence_txtpath = sorted(oxts_data_dir.glob(f"**/{self.sequence_str}.txt"))
            if len(oxts_sequence_txtpath) != 1:
                raise ValueError(f"Found not just 1 sequence txtpath '{oxts_sequence_txtpath}'")
            oxts_sequence_txtpath = oxts_sequence_txtpath[0]
            oxts_data_unprocessed: list[str] = []
            with open(str(oxts_sequence_txtpath), 'r') as f:
                oxts_data_unprocessed = [line.strip("\n").strip(" ") for line in f.readlines()]

            oxts_ts_path = get_singular_dir(sorted(self.oxts_dir.glob("**/timestamps.txt")), raise_error=False)
            if oxts_ts_path is not None:
                with open(str(oxts_ts_path), "r") as f:
                    timestamps = [pd.Timestamp(line.strip("\n")) for line in f.readlines()]
            else:
                use_artificial_ts = True
                timestamps = self.generate_spaced_timestamps(len(oxts_data_unprocessed), self.msg_frequency_s)

            for ts, unprocessed_data in tqdm(zip(timestamps, oxts_data_unprocessed), desc="OXTS", total=len(timestamps)):
                oxts_data = OxtsDataTs.from_str(unprocessed_data, data_keys, ts)
                self.oxts_df.loc[len(self.oxts_df)] = oxts_data.to_pandas_series()
            self.oxts_df.sort_values(by=["ts"], inplace=True)

        self.velodyne_data: list[Path] = []
        self.velodyne_ts: list[pd.Timestamp] = []
        self.velodyne_ts_start: list[pd.Timestamp] = []
        self.velodyne_ts_end: list[pd.Timestamp] = []
        if self.use_velo_data:
            # Velodyne: store file paths + sweep timestamps (lazy-read in the write loop).
            velodyne_data_dir = get_singular_dir(sorted(self.velodyne_pts_dir.glob(f"**/velodyne/{self.sequence_str}")), raise_error=True)
            # 'sync' KITTI stores velodyne as float32 .bin; 'extract'/unsync stores it as text .txt.
            velodyne_files = sorted(velodyne_data_dir.glob("**/*.bin")) or sorted(velodyne_data_dir.glob("**/*.txt"))

            if use_artificial_ts:
                timestamps = self.generate_spaced_timestamps(len(velodyne_files), self.msg_frequency_s)
                timestamps_end = timestamps_start = timestamps
            else:
                def _read_ts(name: str) -> list[pd.Timestamp]:
                    path = get_singular_dir(sorted(self.velodyne_pts_dir.glob(f"**/{name}")), raise_error=False)
                    with open(str(path), "r") as f:
                        return [pd.Timestamp(line.strip("\n")) for line in f.readlines()]

                timestamps = _read_ts("timestamps.txt")
                timestamps_start = _read_ts("timestamps_start.txt")
                timestamps_end = _read_ts("timestamps_end.txt")

            for ts, ts_start, ts_end, velo_file in zip(timestamps, timestamps_start, timestamps_end, velodyne_files):
                self.velodyne_data.append(velo_file)
                self.velodyne_ts.append(ts)
                self.velodyne_ts_start.append(ts_start)
                self.velodyne_ts_end.append(ts_end)

        self.cam_to_velo: Optional[np.ndarray] = None
        self.label_data: Optional[pd.DataFrame] = None
        if self.use_labels_data:
            labels_data_dir = get_singular_dir(sorted(self.labels_dir.glob(f"**/label*")), raise_error=True)
            labels_sequence_path = labels_data_dir / f"{self.sequence_str}.txt"
            self.label_data = extract_tracking_labels(labels_sequence_path, LABEL_CLASSES)

            if self.calib_dir is not None:
                calib_data_dir = get_singular_dir(sorted(self.calib_dir.glob(f"**/calib")), raise_error=False)
                if calib_data_dir is not None:
                    calib_sequence_path = calib_data_dir / f"{self.sequence_str}.txt"
                    if calib_sequence_path.exists():
                        calibration_mats = load_calib_txt_file(calib_sequence_path)
                        T = calibration_mats.get("Tr_velo_cam")
                        if T.shape != (3, 4):
                            print(f"Unexpected shape for 'Tr_velo_cam' matrix shape '{T.shape}'. Expected '(3, 4)'. Skipping coordinate transformation from camera -> velodyne.")
                        elif T is None:
                            print(f"No 'Tr_velo_cam' matrix found at sequence '{self.calib_dir}'. Got the following calibration types: '{list(calibration_mats.keys())}'")
                        else:
                            R_rect = calibration_mats.get("R_rect")
                            if R_rect is None:
                                print(f"No 'R_rect' matrix found at sequence '{self.calib_dir}'. Assuming 'Tr_velo_cam' matrix encompasses full transformation from camera to velodyne coordinate frame.")
                            else:
                                T = R_rect @ T
                            R = T[:3, :3]
                            t = T[:3, 3]
                            R_inv = R.T
                            t_inv = R_inv @ -t
                            self.cam_to_velo = np.column_stack([R_inv, t_inv])
                    else:
                        print(f"No calibration data found at sequence '{self.calib_dir}'. Assuming 3D Detection data is in velodyne frame coordinates already.")
                else:
                    print(f"No calibration data found at '{self.calib_dir}'. Assuming 3D Detection data is in velodyne frame coordinates already.")
            else:
                print("No calibration data found. Assuming 3D Detection data is in velodyne frame coordinates already.")

            if self.cam_to_velo is not None:
                self.label_data = convert_labels_from_cam_to_velo_coordinates(self.label_data, self.cam_to_velo)

    def _build_timeline(self) -> None:
        oxts_df = self.oxts_df if self.use_oxts_data else []
        events: list[tuple[int, str, int | list[int]]] = (
            [(int(oxts_df.iloc[i]["ts"]), "oxts", i) for i in range(len(oxts_df))] +
            [(ts.value, "velodyne", i) for i, ts in enumerate(self.velodyne_ts)]
        )
        if self.use_labels_data:
            for frame_id in np.unique(self.label_data["frame"]):
                rows = self.label_data.loc[self.label_data["frame"] == frame_id]
                events_idx = list(rows.index)
                events.append((pd.Timestamp(frame_id * self.msg_frequency_s, unit='s').value, "label", events_idx))
        if self.write_ground_truth:
            events += [(int(self.oxts_df.iloc[i]["ts"]), "gt", i) for i in range(len(self.oxts_df))]
        events.sort(key=lambda e: e[0])

        if len(events) == 0:
            raise RuntimeError("No events stored, nothing to do...")

        self.timeline = events
        # All stamps are offset by min_ts so they start near zero (keeps RosTime.sec small and
        # preserves double precision in the consumer's sec + nanosec arithmetic). Velodyne messages
        # are stamped at their sweep start, which is earlier than the trigger time in the timeline,
        # so fold those in to keep every published stamp non-negative.
        self.min_ts = min(e[0] for e in events)
        if self.velodyne_ts_start:
            self.min_ts = min(self.min_ts, min(ts.value for ts in self.velodyne_ts_start))

    @staticmethod
    def _read_velodyne(path: Path) -> np.ndarray:
        # Both formats store rows of [x, y, z, reflectance]. 'sync' is float32 binary;
        # 'extract'/unsync is whitespace-separated text.
        if path.suffix == ".bin":
            return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        return np.loadtxt(str(path), dtype=np.float32).reshape(-1, 4)

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------
    @staticmethod
    def rotx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _orientation_quat(self, series_) -> np.ndarray:
        R_ = self.rotz(series_["yaw"]).dot(self.roty(series_["pitch"]).dot(self.rotx(series_["roll"])))
        return R.from_matrix(R_).as_quat()  # [x, y, z, w]

    def _enu(self, series_) -> tuple[float, float, float]:
        ned_x, ned_y, ned_z = lla2ned(
            series_["lat"], series_["lon"], series_["alt"], *self.gps_origin_latlonalt,
            latlon_unit="deg", alt_unit="m", model="wgs84")
        # ROS is ENU; lla2ned returns NED.
        return float(ned_y), float(ned_x), float(-ned_z)

    def _imu_msg(self, idx: int, stamp: RosTime) -> Imu:
        series_ = self.oxts_df.iloc[idx]
        q = self._orientation_quat(series_)
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = "imu"
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        msg.angular_velocity.x = float(series_["wx"])
        msg.angular_velocity.y = float(series_["wy"])
        msg.angular_velocity.z = float(series_["wz"])
        msg.linear_acceleration.x = float(series_["ax"])
        msg.linear_acceleration.y = float(series_["ay"])
        msg.linear_acceleration.z = float(series_["az"])
        return msg

    def _gps_msg(self, idx: int, stamp: RosTime) -> Odometry:
        # nav_msgs/Odometry to match mapOptimization's gpsHandler, which reads pose.pose.position
        # and the x/y/z variances on the diagonal of pose.covariance.
        series_ = self.oxts_df.iloc[idx]
        enu_x, enu_y, enu_z = self._enu(series_)
        pos_accuracy = float(series_["pos_accuracy"])
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "gps"
        msg.pose.pose.position.x = enu_x
        msg.pose.pose.position.y = enu_y
        msg.pose.pose.position.z = enu_z
        msg.pose.pose.orientation.w = 1.0  # orientation unused by the GPS factor; keep it valid
        covariance = [0.0] * 36  # 6x6 row-major; consumer reads indices 0 (x), 7 (y), 14 (z)
        covariance[0] = covariance[7] = covariance[14] = pos_accuracy ** 2
        msg.pose.covariance = covariance
        return msg

    def _velodyne_msg(self, idx: int) -> PointCloud2:
        points = self._read_velodyne(self.velodyne_data[idx])
        ts_start = self.velodyne_ts_start[idx]
        ts_end = self.velodyne_ts_end[idx]

        # Per-point time within the sweep (KITTI stores none). Standard KITTI convention
        # (cf. KISS-ICP): yaw = -atan2(y, x), normalized to [0, 1] over one revolution, scaled by
        # the sweep duration. t = 0 at sweep start (vehicle rear), t = dt at sweep end.
        dt_scan = max((ts_end.value - ts_start.value) * 1e-9, 1e-6)
        yaw = -np.arctan2(points[:, 1], points[:, 0])
        frac = np.clip(0.5 * (yaw / np.pi + 1.0), 0.0, 1.0)
        point_time = (frac * dt_scan).astype(np.float32)

        # Ring (laser/beam index; 0 = bottom beam) from elevation angle. HDL-64E beams are only
        # approximately uniform, so this uniform binning may collide adjacent beams; a measured
        # per-beam angle table would be more precise.
        depth_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        elev_deg = np.degrees(np.arctan2(points[:, 2], depth_xy))
        fov = HDL64_FOV_UP_DEG - HDL64_FOV_DOWN_DEG
        ring = np.clip(np.round((elev_deg - HDL64_FOV_DOWN_DEG) / fov * (N_SCAN - 1)), 0, N_SCAN - 1).astype(np.uint16)

        # Sort by time so the consumer's "last point == scan end" assumption holds and the earliest
        # point defines the deskew reference frame.
        order = np.argsort(point_time, kind="stable")
        xyzi = points[order]
        point_time = point_time[order]
        ring = ring[order]

        # x, y, z, intensity, time (float32) + ring (uint16) -> 22 bytes/point.
        point_dtype = np.dtype({
            "names": ["x", "y", "z", "intensity", "time", "ring"],
            "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
            "offsets": [0, 4, 8, 12, 16, 20],
            "itemsize": 22,
        })
        cloud = np.empty(xyzi.shape[0], dtype=point_dtype)
        cloud["x"], cloud["y"], cloud["z"], cloud["intensity"] = xyzi[:, 0], xyzi[:, 1], xyzi[:, 2], xyzi[:, 3]
        cloud["time"] = point_time
        cloud["ring"] = ring

        msg = PointCloud2()
        # Stamp at the sweep start: imageProjection treats header.stamp as the scan start, and the
        # per-point `time` values are offsets relative to it.
        msg.header.stamp = to_ros_time_from_pd(pd.Timestamp(ts_start.value - self.min_ts))
        msg.header.frame_id = "feature_extraction"
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
        return msg

    def _gt_pose_msg(self, idx: int, stamp: RosTime) -> PoseStamped:
        series_ = self.oxts_df.iloc[idx]
        enu_x, enu_y, enu_z = self._enu(series_)
        q = self._orientation_quat(series_)
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = GT_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = enu_x, enu_y, enu_z
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return msg

    def _3d_detection_msg(self, idx: int, stamp: RosTime) -> Detection3D:
        label = self.label_data.iloc[idx]

        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = "detection"

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = label.type
        # TODO: Maybe add some randomness to the probability score
        hypothesis.hypothesis.score = 1.0
        detection.results.append(hypothesis)

        detection.bbox.center.position.x = label.x
        detection.bbox.center.position.y = label.y
        detection.bbox.center.position.z = label.z

        # TODO: Add orientation
        detection.bbox.center.orientation.x = 0.0
        detection.bbox.center.orientation.y = 0.0
        detection.bbox.center.orientation.z = 0.0
        detection.bbox.center.orientation.w = 1.0  # Facing straight forward

        detection.bbox.size.x = label.width
        detection.bbox.size.y = label.length
        detection.bbox.size.z = label.height

        return detection

    def _3d_detection_array_msg(self, idxs: list[int], stamp: RosTime) -> Detection3DArray:
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "detections"

        for idx in idxs:
            msg.detections.append(self._3d_detection_msg(idx, stamp))
        return msg

    # ------------------------------------------------------------------
    # Bag writing
    # ------------------------------------------------------------------
    def write(self, output: Path, storage_id: str = "sqlite3") -> None:
        if output.exists():
            raise SystemExit(f"Output bag '{output}' already exists; remove it or choose another path.")

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(output), storage_id=storage_id),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        topics = [
            (TOPIC_IMU, "sensor_msgs/msg/Imu"),
            (TOPIC_POINTS, "sensor_msgs/msg/PointCloud2"),
            (TOPIC_GPS, "nav_msgs/msg/Odometry"),
            (TOPIC_DET3D, "detections3d"),
        ]
        if self.write_ground_truth:
            topics += [(TOPIC_GT_POSE, "geometry_msgs/msg/PoseStamped"),
                       (TOPIC_GT_PATH, "nav_msgs/msg/Path")]
        for name, type_str in topics:
            writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=type_str, serialization_format="cdr"))

        for ts, kind, i in tqdm(self.timeline, desc="Writing bag"):
            # Bag record time and message header share the same min_ts offset; only inter-message
            # deltas matter for `ros2 bag play`.
            ns = int(ts - self.min_ts)
            stamp = to_ros_time_from_pd(pd.Timestamp(ns))

            if kind == "oxts":
                writer.write(TOPIC_IMU, serialize_message(self._imu_msg(i, stamp)), ns)
                writer.write(TOPIC_GPS, serialize_message(self._gps_msg(i, stamp)), ns)
            elif kind == "velodyne":
                writer.write(TOPIC_POINTS, serialize_message(self._velodyne_msg(i)), ns)
            elif kind == "gt":
                pose = self._gt_pose_msg(i, stamp)
                writer.write(TOPIC_GT_POSE, serialize_message(pose), ns)
                self._gt_path.poses.append(pose)
            elif kind == "label":
                detections = self._3d_detection_array_msg(i, stamp)
                writer.write(TOPIC_DET3D, serialize_message(detections), ns)

        # Write the complete ground-truth path once, at the final timestamp.
        if self.write_ground_truth and self._gt_path.poses:
            last_ns = int(self.timeline[-1][0] - self.min_ts)
            self._gt_path.header.stamp = to_ros_time_from_pd(pd.Timestamp(last_ns))
            writer.write(TOPIC_GT_PATH, serialize_message(self._gt_path), last_ns)

        # Destroying the writer finalizes metadata.yaml and the storage file.
        del writer
        print(f"Wrote bag to '{output}'. Play with: ros2 bag play {output}")


def main():
    parser = argparse.ArgumentParser(description="Write KITTI data to a rosbag2 bag for offline SLAM playback")
    parser.add_argument("--data", default=str(TRACKING_DATA_PATH),
                        help="Path to the tracking KITTI dataset")
    parser.add_argument("--output", default=None,
                        help="Output bag directory (default: ./<dataset_name>_bag)")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"],
                        help="rosbag2 storage backend")
    parser.add_argument("--no-ground-truth", action="store_true",
                        help="Skip writing the /ground_truth topics")
    parser.add_argument("--data-to-skip", default=None, nargs="+", type=str)
    args = parser.parse_args()

    data_path = Path(args.data)
    output = Path(args.output) if args.output else OUTPUT_BAG_PATH / f"{data_path.resolve().name}_bag"

    writer = KittiBagWriter(
        data_path=data_path,
        write_ground_truth=not args.no_ground_truth,
        data_to_skip=args.data_to_skip,
    )
    writer.write(output=output, storage_id=args.storage_id)


if __name__ == "__main__":
    main()
