import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import TransformStamped
import tf2_ros

class PathCameraTFBroadcaster(Node):
    def __init__(self):
        super().__init__('path_camera_tf_broadcaster')

        # Initialize the transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribe to your pipeline's path topic
        self.subscription = self.create_subscription(
            NavPath,
            "/ground_truth/path",
            self.path_callback,
            10,
        )

        self.get_logger().info("CREATED NODE.")

    def path_callback(self, msg: NavPath):
        # Ensure the path is not empty
        if not msg.poses:
            self.get_logger().info(f"No poses in msg!")
            return

        # Get the latest pose on the path (or change index to target different points along the path)
        latest_pose = msg.poses[-1]

        # Create the transform message
        t = TransformStamped()

        # Read the frame id from the path header (e.g., 'map' or 'world')
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = msg.header.frame_id if msg.header.frame_id else 'map'
        t.child_frame_id = 'moving_camera_frame'

        self.get_loger().info(f"time: '{t.header.stamp}'")
        # Set translation (position)
        t.transform.translation.x = latest_pose.pose.position.x
        t.transform.translation.y = latest_pose.pose.position.y
        t.transform.translation.z = latest_pose.pose.position.z

        # Set rotation (orientation)
        t.transform.rotation.x = latest_pose.pose.orientation.x
        t.transform.rotation.y = latest_pose.pose.orientation.y
        t.transform.rotation.z = latest_pose.pose.orientation.z
        t.transform.rotation.w = latest_pose.pose.orientation.w

        # Send the transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PathCameraTFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
