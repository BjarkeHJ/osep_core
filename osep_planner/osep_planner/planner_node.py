#!/usr/bin/env python3

"""
TODO:
    - Create Adjacency representation for the skeleton points in SkeletonState
    - Generate 4DOF Viewpoints for 

"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, PoseArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

from .skeleton_state import SkeletonState
from .viewpoint_manager import ViewpointManager

class PlannerNode(Node):
    def __init__(self):
        super().__init__('OsepPlannerNode')

        self.declare_parameter('skeleton_topic', 'osep/tsdf/skeleton')
        self.declare_parameter('viewpoint_topic', 'osep/planner/viewpoints')

        skeleton_topic = self.get_parameter('skeleton_topic').get_parameter_value().string_value
        viewpoint_topic = self.get_parameter('viewpoint_topic').get_parameter_value().string_value

        qos = rclpy.qos.QoSProfile(depth=10)
        self.skel_sub = self.create_subscription(PointCloud2, skeleton_topic, self.skeleton_callback, qos)
        self.vpts_pub = self.create_publisher(PoseArray, viewpoint_topic, qos)

        self.skel = SkeletonState()
        self.vpman = ViewpointManager(self.skel)

    @staticmethod
    def _pointcloud2_to_xyz_rgb(msg: PointCloud2):
        arr = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)
        if arr.dtype.names:
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)
            rgb_u32 = arr["rgb"].view(np.uint32)
        else:
            xyz = arr[:, 0:3].astype(np.float32, copy=False)
            rgb_u32 = arr[:, 3].view(np.uint32)
        return xyz, rgb_u32

    def skeleton_callback(self, msg):
        xyz, rgb = self._pointcloud2_to_xyz_rgb(msg)
        if xyz.size == 0:
            return
        
        self.get_logger().info(f"Received cloud with shape: {xyz.shape}")
        self.skel.update_skeleton(xyz)
        s = self.skel.get_size()
        print(f"Skeleton Updated: Skeleton Size: {s}")
        
        self.vpman.update_viewpoints()
        return


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()