import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros


# ==========================================
# 1. ODOMETRY TF BROADCASTER
# ==========================================
class OdomCameraTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_camera_tf_broadcaster')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Odometry,
            '/lio_sam/mapping/odometry',  # <-- Update to your topic
            self.odom_callback,
            odom_qos)

        self.get_logger().info("SETUP ODOMETRY CAMERA TF BROADCASTER")

    def odom_callback(self, msg: Odometry):
        t = TransformStamped()
        # t.header.stamp = msg.header.stamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = msg.header.frame_id if msg.header.frame_id else 'map'
        t.child_frame_id = 'odom_camera_frame'  # Unique frame name

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


# ==========================================
# 2. PATH TF BROADCASTER
# ==========================================
class PathCameraTFBroadcaster(Node):
    def __init__(self):
        super().__init__('path_camera_tf_broadcaster')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            # durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(
            Path,
            '/lio_sam/imu/path',  # <-- Update to your topic
            self.path_callback,
            path_qos)

        self.get_logger().info("SETUP PATH CAMERA TF BROADCASTER")

    def path_callback(self, msg: Path):
        if not msg.poses:
            return

        latest_pose = msg.poses[-1]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = msg.header.frame_id if msg.header.frame_id else 'map'
        t.child_frame_id = 'path_camera_frame'  # Unique frame name

        self.get_logger().info(f"Updating Path camera at time '{t}'")

        t.transform.translation.x = latest_pose.pose.position.x
        t.transform.translation.y = latest_pose.pose.position.y
        t.transform.translation.z = latest_pose.pose.position.z

        t.transform.rotation = latest_pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


# ==========================================
# 3. MAIN RUNTIME WITH EXECUTOR
# ==========================================
def main(args=None):
    rclpy.init(args=args)

    # Instantiate both node components
    odom_node = OdomCameraTFBroadcaster()
    path_node = PathCameraTFBroadcaster()

    # Use a MultiThreadedExecutor to process callbacks in parallel threads
    executor = MultiThreadedExecutor()
    executor.add_node(odom_node)
    executor.add_node(path_node)

    try:
        # Spin both nodes simultaneously
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown sequence
        odom_node.destroy_node()
        path_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
