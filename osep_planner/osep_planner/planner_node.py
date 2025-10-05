#!/usr/bin/env python3

"""
TODO:
    - Create Adjacency representation for the skeleton points in SkeletonState (Type 3 issue -> Junction does not show up atm??)
    - Plan Path among valid viewpoints
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

from .skeleton_state import SkeletonState
from .viewpoint_manager import ViewpointManager
from .planner import PathPlanner
from .occ_map import OCCMap

class PlannerNode(Node):
    def __init__(self):
        super().__init__('OsepPlannerNode')

        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('drone_frame', 'base_link')
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.drone_frame = self.get_parameter('drone_frame').get_parameter_value().string_value

        self.declare_parameter('skeleton_topic', 'osep/tsdf/skeleton')
        self.declare_parameter('viewpoint_topic', 'osep/planner/viewpoints')
        self.declare_parameter('map_topic', 'osep/tsdf/static_pointcloud')
        self.declare_parameter('path_topic', 'osep/viewpoints')
        self.declare_parameter('adjusted_topic', 'osep/viewpoints_adjusted')
        skeleton_topic = self.get_parameter('skeleton_topic').get_parameter_value().string_value
        viewpoint_topic = self.get_parameter('viewpoint_topic').get_parameter_value().string_value
        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        adjusted_topic = self.get_parameter('adjusted_topic').get_parameter_value().string_value

        self.declare_parameter('safe_distance', 15.0)
        self.declare_parameter('voxel_size', 1.0)
        self.declare_parameter('camera_hfov', 60.0)
        self.declare_parameter('camera_vfov', 30.0)
        self.declare_parameter('camera_range', 20.0)
        safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value
        voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        cam_hfov = self.get_parameter('camera_hfov').get_parameter_value().double_value
        cam_vfov = self.get_parameter('camera_vfov').get_parameter_value().double_value
        cam_range = self.get_parameter('camera_range').get_parameter_value().double_value

        qos = rclpy.qos.QoSProfile(depth=10)
        self.skel_sub = self.create_subscription(PointCloud2, skeleton_topic, self.skeleton_callback, qos)
        self.map_sub = self.create_subscription(PointCloud2, map_topic, self.map_callback, qos)
        self.vpts_pub = self.create_publisher(PoseArray, viewpoint_topic, qos)
        self.edges_pub = self.create_publisher(MarkerArray, "osep/planner/edges", qos)
        self.path_pub = self.create_publisher(Path, path_topic, qos)
        self.adjusted_sub = self.create_subscription(Path, adjusted_topic, self.adjusted_callback, qos)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0).to_msg())
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.drone_position: np.ndarray = None

        self.skel = SkeletonState()
        self.occ_map = OCCMap(voxel_size, cam_hfov, cam_vfov, cam_range)
        self.vpman = ViewpointManager(self.skel, self.occ_map, safe_distance)
        self.planner = PathPlanner(self.skel, self.occ_map, self.vpman, self.drone_position, max_horizon=10)
        
        self.get_logger().info("PlannerNode Initialized")

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
        m.header.frame_id = getattr(self, "global_frame", "odom")
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
        pa.header.frame_id = self.global_frame
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

    def get_drone_position(self):
        tgt = self.global_frame
        src = self.drone_frame

        pin = PoseStamped()
        pin.header.frame_id = src
        pin.header.stamp =  Time()
        pin.pose.orientation.w = 1.0

        try:
            if not self.tf_buffer.can_transform(
                tgt, src, Time(), timeout=Duration(seconds=0.2)
            ):
                self.get_logger().warn(f"TF not available: {src} -> {tgt}")
                return None
            
            pout = self.tf_buffer.transform(
                pin, tgt, timeout=Duration(seconds=0.2)
            )

            return np.array(
                [pout.pose.position.x, pout.pose.position.y, pout.pose.position.z],
                dtype=np.float32
            )
        
        except TransformException as ex:
            self.get_logger().warn(f"TF exception while transforming {src} -> {tgt}: {ex}")
            return None


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

    def map_callback(self, msg):
        xyz, _ = self._pointcloud2_to_xyz_rgb(msg)
        if xyz.size == 0:
            return
        # insert centers in map
        self.occ_map.insert_centers(xyz)

    def adjusted_callback(self, msg):
        return

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()