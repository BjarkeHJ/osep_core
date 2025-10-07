#!/usr/bin/env python3

"""
TODO:
    - Create Adjacency representation for the skeleton points in SkeletonState (Type 3 issue -> Junction does not show up atm??)
    - Plan Path among valid viewpoints
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
        self.declare_parameter('camera_hfov', 50.0)
        self.declare_parameter('camera_vfov', 40.0)
        self.declare_parameter('camera_range', 20.0)
        self.declare_parameter('reach_distance', 2.0)
        safe_distance = float(self.get_parameter('safe_distance').get_parameter_value().double_value)
        voxel_size = float(self.get_parameter('voxel_size').get_parameter_value().double_value)
        cam_hfov = float(self.get_parameter('camera_hfov').get_parameter_value().double_value)
        cam_vfov = float(self.get_parameter('camera_vfov').get_parameter_value().double_value)
        cam_range = float(self.get_parameter('camera_range').get_parameter_value().double_value)
        self.reach_distance = float(self.get_parameter('reach_distance').get_parameter_value().double_value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.skel_sub = self.create_subscription(PointCloud2, skeleton_topic, self.skeleton_callback, qos)
        self.map_sub = self.create_subscription(PointCloud2, map_topic, self.map_callback, qos)
        self.vpts_pub = self.create_publisher(PoseArray, viewpoint_topic, qos)
        self.edges_pub = self.create_publisher(MarkerArray, "osep/planner/edges", qos)
        self.path_pub = self.create_publisher(Path, path_topic, qos)
        self.adjusted_sub = self.create_subscription(Path, adjusted_topic, self.adjusted_callback, qos)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0).to_msg())
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.plan_timer = self.create_timer(0.1, self.plan_tick) # 10 hz?

        self.drone_position: np.ndarray = None

        self.skel = SkeletonState()
        self.occ_map = OCCMap(voxel_size, cam_hfov, cam_vfov, cam_range)
        self.vpman = ViewpointManager(self.skel, self.occ_map, safe_distance)
        self.planner = PathPlanner(self.skel, self.occ_map, self.vpman, self.drone_position, max_horizon=5)
        
        self._adjusted: bool = True
        self._init_mode: bool = True
        self._init_points = [np.array([0.0, 0.0, 110.0], dtype=np.float32), 
                             np.array([120.0, 0.0, 140.0], dtype=np.float32)]

        self.prev_dist_to_tgt = np.inf
        self.track_count = 0

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

    @staticmethod
    def _quat_to_yaw(qx, qy, qz, qw) -> float:
        return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))

    def publish_edge_markers(self):
        if not self.skel.skelver:
            return
        # adj = self.skel.adjacency
        v1 = self.skel.current_version()
        adj, vids = self.skel.get_adjacency_and_vids()
        v2 = self.skel.current_version()
        if v1 != v2:
            return

        if adj is None or adj.size == 0 or vids is None:
            return

        # vids = np.array(sorted(self.skel.skelver.keys()), dtype=np.int32)
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

        # vpts = list(self.vpman.viewpoints.values())
        vpts = self.vpman.get_viewpoints() # return only valid viewpoints

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

    def publish_path(self):
        if self.planner is None or self.planner.path is None:
            return
        ids = list(self.planner.path.viewpoints)
        if not ids:
            return
        
        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for vptid in ids:
            vp = self.vpman.viewpoints.get(int(vptid))
            if vp is None or not vp.valid:
                continue
            ps = PoseStamped()
            ps.header.frame_id = self.global_frame
            ps.header.stamp = path.header.stamp
            ps.pose.position.x = float(vp.pos[0])
            ps.pose.position.y = float(vp.pos[1])
            ps.pose.position.z = float(vp.pos[2])
            halfyaw = 0.5 * float(vp.yaw)
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = np.sin(halfyaw)
            ps.pose.orientation.w = np.cos(halfyaw)

            path.poses.append(ps)
        
        self.path_pub.publish(path)
        self.get_logger().info(f"Published New Path - Path lenght: {len(path.poses)}")
    
    def publish_init_path(self):
        if not self._init_points:
            return
        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for p in self._init_points:
            ps = PoseStamped()
            ps.header.frame_id = self.global_frame
            ps.header.stamp = path.header.stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)

    def get_drone_position(self):
        tgt = self.global_frame
        src = self.drone_frame

        try:
            if not self.tf_buffer.can_transform(tgt, src, Time(), timeout=Duration(seconds=0.2)):
                self.get_logger().warn(f"TF not available: {src} -> {tgt}")
                return None
            
            ts = self.tf_buffer.lookup_transform(tgt, src, Time(), timeout=Duration(seconds=0.2))
            t = ts.transform.translation
            return np.array([t.x, t.y, t.z], dtype=np.float32)
        except TransformException as ex:
            self.get_logger().warn(f"TF exception while transforming {src} -> {tgt}: {ex}")
            return None

    def prune_current_path(self):
        if self.planner is None or self.planner.path is None or not self.planner.path.viewpoints:
            return
        keep = []
        for vptid in self.planner.path.viewpoints:
            vp = self.vpman.viewpoints.get(int(vptid))
            if vp is not None and vp.valid and not getattr(vp, "visited", False):
                keep.append(int(vptid))
        if len(keep) != len(self.planner.path.viewpoints):
            self.planner.path.viewpoints = keep
            self.planner.path.path_len = len(keep)

    def pop_if_reached(self) -> bool:
        if self.planner is None or self.planner.path is None:
            return False
        if not self.planner.path.viewpoints:
            return False
        if self.drone_position is None:
            return False
        
        head_vptid = int(self.planner.path.viewpoints[0])
        vp = self.vpman.viewpoints.get(head_vptid)
        if vp is None or not vp.valid:
            self.planner.path.viewpoints.pop(0)
            self.planner.path.path_len = len(self.planner.path.viewpoints)
            return True
        
        dist = float(np.linalg.norm(vp.pos.astype(np.float32) - self.drone_position.astype(np.float32)))
        if dist > self.prev_dist_to_tgt:
            self.track_count += 1
            # self.prev_dist_to_tgt = dist
        elif dist < self.reach_distance:
            self.prev_dist_to_tgt = dist
        
        if self.track_count > 5:
            self.track_count = 0
            self.prev_dist_to_tgt = np.inf
            self.vpman.mark_visited(head_vptid)
            try:
                newseen = self.occ_map.mark_visible_from(vp.pos, vp.yaw, commit=True)
            except Exception:
                newseen = None
                self.get_logger().warn(f"Could Not Mark Voxels Seen")
                pass

            self.planner.path.viewpoints.pop(0)
            self.planner.path.path_len = len(self.planner.path.viewpoints)
            self.get_logger().info(f"Viewpoint Reached - New Voxels Seen: {newseen if newseen is not None else 'None'}\n Total seen voxels: {len(self.occ_map._seen)} / {len(self.occ_map._occ)}")
            self.get_logger().info(f"Total seen voxels: {len(self.occ_map._seen)} / {len(self.occ_map._occ)}")
            self.get_logger().info(f"Total visited viewpoints: {self.vpman.get_n_visited()} / {len(self.vpman.get_viewpoints())}")
            return True
        
        elif self.track_count > 0 and self.prev_dist_to_tgt > dist:
            self.track_count = 0
        else:
            self.prev_dist_to_tgt = dist

        # if dist <= self.reach_distance:
        #     self.vpman.mark_visited(head_vptid)
        #     try:
        #         newseen = self.occ_map.mark_visible_from(vp.pos, vp.yaw, commit=True)
        #     except Exception:
        #         newseen = None
        #         self.get_logger().warn(f"Could Not Mark Voxels Seen")
        #         pass

        #     self.planner.path.viewpoints.pop(0)
        #     self.planner.path.path_len = len(self.planner.path.viewpoints)
        #     self.get_logger().info(f"Viewpoint Reached - New Voxels Seen: {newseen if newseen is not None else 'None'}\n Total seen voxels: {len(self.occ_map._seen)} / {len(self.occ_map._occ)}")
        #     self.get_logger().info(f"Total seen voxels: {len(self.occ_map._seen)} / {len(self.occ_map._occ)}")
        #     self.get_logger().info(f"Total visited viewpoints: {self.vpman.get_n_visited()} / {len(self.vpman.get_viewpoints())}")
        #     return True
        
        return False

    def pop_init_if_reached(self) -> bool:
        if not self._init_points or self.drone_position is None:
            return False
        head = self._init_points[0]
        dist = float(np.linalg.norm(head.astype(np.float32) - self.drone_position.astype(np.float32)))
        if dist <= self.reach_distance:
            self._init_points.pop(0)
            return True
        return False

    def skeleton_callback(self, msg):
        xyz, rgb = self._pointcloud2_to_xyz_rgb(msg)
        if xyz.size == 0:
            return
        
        print(f"Received cloud with shape: {xyz.shape}")
        
        self.skel.update_skeleton(xyz, rgb)
        self.publish_edge_markers() # Publishes edge connection between skeleton vertices for visualization...

        print(f"Skeleton Version: {self.skel.current_version()}")

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

    def adjusted_callback(self, msg: Path):
        if self._init_mode:
            self._adjusted = True
            return

        if self.planner is None or self.planner.path is None or not self.planner.path.viewpoints:
            self._adjusted = True
            return
        
        ids = self.planner.path.viewpoints
        if len(msg.poses) != len(ids):
            self.get_logger().warn(f"Adjusted path length {len(msg.poses)} != current path lenght {len(ids)}; Ignoring")
            self._adjusted = True
            return
        
        invalid_ids = []
        for i, ps in enumerate(msg.poses):
            vptid = int(ids[i])
            vp = self.vpman.viewpoints.get(vptid)
            if vp is None:
                continue

            if (ps.header.frame_id is None) or (ps.header.frame_id == ""):
                vp.valid = False
                invalid_ids.append(vptid)
                continue

            p = ps.pose.position
            q = ps.pose.orientation
            new_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            new_yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)

            if np.linalg.norm(vp.pos - new_pos) > 1e-6 or abs(vp.yaw - new_yaw) > 1e-6:
                vp.pos = new_pos
                vp.yaw = new_yaw

        if invalid_ids:
            self.prune_current_path()
        
        self._adjusted = True
        return

    def plan_tick(self):
        self.drone_position = self.get_drone_position()
        if self.drone_position is None:
            return
        
        # Init Phase
        if self._init_mode:
            if self.skel.get_size() > 0:
                self.get_logger().info("Autonomy Takeover!")
                self._init_mode = False
                self._adjusted = True
                return
            
            popped = True
            while popped:
                popped = self.pop_init_if_reached()
            if not self._init_points:
                self._init_mode = False
                self._adjusted = True
                return
            
            self.publish_init_path()
            return

        # Autonomy Phase
        popped = True
        while popped:
            popped = self.pop_if_reached()

        if self.planner and self.planner.path and not self.planner.path.viewpoints:
            self._adjusted = True

        if not self._adjusted:
            self.get_logger().warn("Waiting for adjusted Viewpoints")
            return # waiting for adjusted viewpoints

        self.planner.drone_pos = self.drone_position
        self.prune_current_path()

        try:
            committed = self.planner.generate_path()
        except Exception as ex:
            self.get_logger().warn(f"planner.generate_path() error: {ex}")
            return
        
        if committed:
            self.publish_path()
            self._adjusted = False
            
def main(args=None):
    rclpy.init(args=args)
    planner = PlannerNode()
    rclpy.spin(planner)
    planner.destroy_node()