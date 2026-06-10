import argparse
from pathlib import Path

import numpy as np
import rclpy
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, PointField

KITTI_DATA_DIR = Path("/home/ckelton/data/KITTI/3dObjectDetection/object_velodyne")
TOPIC = "feature_extraction/data_pcl"
FRAME_ID = "feature_extraction"
FRAME_RATE_HZ = 10.0


def make_point_cloud2_msg(points: np.ndarray, stamp_ns: int) -> PointCloud2:
    """Build a PointCloud2 from an (N, 4) float32 array of (x, y, z, intensity)."""
    msg = PointCloud2()

    secs = stamp_ns // 1_000_000_000
    nanosecs = stamp_ns % 1_000_000_000
    msg.header.stamp = Time(sec=int(secs), nanosec=int(nanosecs))
    msg.header.frame_id = FRAME_ID

    msg.height = 1
    msg.width = points.shape[0]

    msg.fields = [
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = points.astype(np.float32).tobytes()
    msg.is_dense = True

    return msg


def main():
    parser = argparse.ArgumentParser(description="Generate a ROS2 bag from KITTI feature_extraction .bin files.")
    parser.add_argument(
        "--split",
        choices=["training", "testing"],
        default="training",
        help="KITTI data split to use (default: training)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=KITTI_DATA_DIR,
        help=f"Root KITTI feature_extraction directory (default: {KITTI_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output bag path (default: <data-dir>/<split>.bag)",
    )
    args = parser.parse_args()

    velodyne_dir = args.data_dir / args.split / "feature_extraction"
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Velodyne directory not found: {velodyne_dir}")

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {velodyne_dir}")

    default_output_dir = Path(__file__).parent.parent / "data"
    output_path = args.output or default_output_dir / f"{args.split}.bag"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Split:   {args.split}")
    print(f"Frames:  {len(bin_files)}")
    print(f"Output:  {output_path}")

    rclpy.init()

    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=str(output_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    writer.open(storage_options, converter_options)

    writer.create_topic(rosbag2_py.TopicMetadata(
        name=TOPIC,
        type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr",
    ))

    frame_interval_ns = int(1_000_000_000 / FRAME_RATE_HZ)

    for i, bin_file in enumerate(bin_files):
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        stamp_ns = i * frame_interval_ns
        msg = make_point_cloud2_msg(points, stamp_ns)
        writer.write(TOPIC, serialize_message(msg), stamp_ns)

        if (i + 1) % 100 == 0 or (i + 1) == len(bin_files):
            print(f"  Written {i + 1}/{len(bin_files)} frames")

    del writer
    rclpy.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
