#!/usr/bin/env python3

"""
TODO:
    - Create Adjacency representation for the skeleton points in SkeletonState (Type 3 issue -> Junction does not show up atm??)
    - Generate 4DOF Viewpoints for (Integrate the map structure for occupancy check)
    - Plan Path among valid viewpoints

"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

from .skeleton_state import SkeletonState
from .viewpoint_manager import ViewpointManager
from .planner import PathPlanner

class PlannerNode(Node):
    def __init__(self):
        super().__init__('OsepPlannerNode')

        self.declare_parameter('skeleton_topic', 'osep/tsdf/skeleton')
        self.declare_parameter('viewpoint_topic', 'osep/planner/viewpoints')
        self.declare_parameter('safe_distance', 10.0)

        skeleton_topic = self.get_parameter('skeleton_topic').get_parameter_value().string_value
        viewpoint_topic = self.get_parameter('viewpoint_topic').get_parameter_value().string_value
        safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value

        qos = rclpy.qos.QoSProfile(depth=10)
        self.skel_sub = self.create_subscription(PointCloud2, skeleton_topic, self.skeleton_callback, qos)
        self.vpts_pub = self.create_publisher(PoseArray, viewpoint_topic, qos)

        self.edges_pub = self.create_publisher(MarkerArray, "osep/planner/edges", qos)

        self.fixed_frame = "odom"
        
        self.skel = SkeletonState()
        self.vpman = ViewpointManager(self.skel, safe_distance)
        self.planner = PathPlanner(self.vpman.viewpoints)

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

    def publish_edge_markers(self):
        if not self.skel.skelver:
            return
        adj = self.skel.adjacency
        if adj is None or adj.size == 0:
            return

        vids = np.array(sorted(self.skel.skelver.keys()), dtype=np.int32)
        if adj.shape != (vids.size, vids.size):
            self.get_logger().warn(
                f"Adjacency shape {adj.shape} does not match number of vertices {vids.size}"
            )
            return

        # positions aligned to vids order
        pos = np.vstack([self.skel.skelver[int(v)].pos for v in vids])

        m = Marker()
        m.header.frame_id = getattr(self, "fixed_frame", "map")
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "skeleton_edges"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05  # line width (m)
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime = Duration(seconds=0).to_msg()  # 0 = forever

        # Build edge list: only upper triangle to avoid duplicates
        edges = np.column_stack(np.triu(adj != 0, k=1).nonzero())
        for i, j in edges:
            p1 = Point(x=float(pos[i, 0]), y=float(pos[i, 1]), z=float(pos[i, 2]))
            p2 = Point(x=float(pos[j, 0]), y=float(pos[j, 1]), z=float(pos[j, 2]))
            m.points.extend([p1, p2])

        marr = MarkerArray()
        marr.markers.append(m)
        self.edges_pub.publish(marr)

    def publish_viewpoints(self):
        pa = PoseArray()
        pa.header.frame_id = self.fixed_frame
        pa.header.stamp = self.get_clock().now().to_msg()

        vpts = list(self.vpman.viewpoints.values())

        for vp in vpts:
            if getattr(vp, "visited", False):
                continue
            if hasattr(vp, "valid") and not vp.valid:
                continue

            p = Pose()
            p.position.x = float(vp.pos[0])
            p.position.y = float(vp.pos[1])
            p.position.z = float(vp.pos[2])
            
            half = 0.5 * float(vp.yaw)
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = np.sin(half)
            p.orientation.w = np.cos(half)

            pa.poses.append(p)
        
        self.vpts_pub.publish(pa)

    def skeleton_callback(self, msg):
        xyz, rgb = self._pointcloud2_to_xyz_rgb(msg)
        if xyz.size == 0:
            return
        
        self.get_logger().info(f"Received cloud with shape: {xyz.shape}")
        self.skel.update_skeleton(xyz, rgb)
        self.publish_edge_markers() # Publishes edge connection between skeleton vertices for visualization...

        s = self.skel.get_size()
        print(f"Skeleton Size: {s}")
        
        self.vpman.update_viewpoints()
        vpts = self.vpman.get_viewpoints()
        n = len(vpts)
        print(f"Number of viewpoints: {n}")
        self.publish_viewpoints() # Publishes all generated viewpoints for visualization...

        return

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()