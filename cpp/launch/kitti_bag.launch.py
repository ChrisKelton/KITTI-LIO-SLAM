import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


WS_ROOT: str = os.path.expanduser(Path(__file__).parent.parent)
SPLIT_TYPE: str = "training"  # ['training', 'testing']

def generate_launch_description():
    # bag_file = str(Path(WS_ROOT) / f"data/{SPLIT_TYPE}.bag")

    velodyne_node = Node(
        package='velodyne_backend',
        executable='feature_extraction',
        name='feature_extraction',
        output='screen',
    )

    world_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_tf',
        arguments=['--frame-id', 'world', '--child-frame-id', 'feature_extraction'],
        output='log',
    )

    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', os.path.join(WS_ROOT, 'rviz', 'kitti.rviz')],
    #     output='log',
    #     additional_env={'QT_QPA_PLATFORM': 'xcb'},
    # )

    return LaunchDescription([
        world_tf_node,
        velodyne_node,
        # rviz_node,
        # ExecuteProcess(
        #     cmd=["ros2", "bag", "play", bag_file, "--loop", "-r", "1.0", "--clock"],
        #     output="screen"
        # )
    ])
