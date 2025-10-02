#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree as KDTree

# State Containers
@dataclass
class Vertex:
    vid: int
    pos: np.ndarray
    # nbs: np.ndarray # Neighboring vertices
    last_update: int # when this was last updated

@dataclass
class Viewpoint:
    vid: int
    pos: np.ndarray
    yaw: float
    visited: bool

class SkeletonState:
    def __init__(self, merge_radius=3.0, update_th = 1.0):
        self.skelver: dict[int, Vertex] = {}
        self.viewpoints: dict[int, Viewpoint] = {}
        
        self._merge_r2 = merge_radius**2
        self._update_th = update_th
        self._next_id = 0

        self._pos_cache = None # (M,3)
        self._id_cache = None # (M,)
        self._kdtree = None

        self._version = 0

    def get_skeleton(self) -> np.ndarray:
        if not self.skelver:
            return np.empty((0, 4), dtype=np.float64)
        # rows = [[v.vid, v.pos[0], v.pos[1], v.pos[2]] for v in self.skelver.values()]
        vids = sorted(self.skelver.keys())
        rows = [[vid, *self.skelver[vid].pos] for vid in vids]
        return np.array(rows, dtype=np.float64)

    def get_size(self) -> int:
        return len(self.skelver)
    
    def current_version(self) -> int:
        return self._version

    def get_vertices_updated_since(self, since_version: int) -> list[Vertex]:
        return [v for v in self.skelver.values() if v.last_update > since_version]

    def update_skeleton(self, points_xyz: np.ndarray):
        if points_xyz is None or points_xyz.size == 0:
            return

        self._version += 1 # update version

        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        self._build_search_index()

        if self._pos_cache is None or self._pos_cache.shape[0] == 0:
            # No existing skeleton -> Spawn for all new
            for p in pts:
                self._create_vertex(p)
            self._invalidate_index()
            return

        r = float(np.sqrt(self._merge_r2))

        # For each point, get ALL candidate vertex indices within r
        idx_lists = self._kdtree.query_ball_point(pts, r=r)

        # Build candidate (dist, point_idx, vert_idx) tuples
        cand = []
        for i, idxs in enumerate(idx_lists):
            if not idxs:
                continue
            P = pts[i]
            V = self._pos_cache[idxs]  # (k,3)
            d = np.linalg.norm(V - P, axis=1)
            for dist, j_local in zip(d, idxs):
                cand.append((float(dist), i, int(j_local)))

        # Sort by distance (greedy minimum matching)
        cand.sort(key=lambda t: t[0])

        matched_points = set()
        matched_verts = set()

        # Assign greedily: each point & vertex at most once
        for dist, i, j in cand:
            if i in matched_points or j in matched_verts:
                continue
            vid = int(self._id_cache[j])
            self._update_vertex(vid, pts[i])   # pass single (3,) point
            matched_points.add(i)
            matched_verts.add(j)

        # Any unassigned points become new vertices
        for i, p in enumerate(pts):
            if i not in matched_points:
                self._create_vertex(p)

        self._invalidate_index()

    
    def _create_vertex(self, p: np.ndarray) -> int:
        vid = self._next_id
        self._next_id += 1 #increment next id
        self.skelver[vid] = Vertex(vid=vid, 
                                   pos=np.array(p, dtype=np.float64), 
                                   last_update=self._version) # creates new entry with key vid
        return vid

    def _update_vertex(self, vid: int, P: np.ndarray) -> None:
        old = self.skelver[vid].pos
        if np.linalg.norm(old - P) >= self._update_th:
            self.skelver[vid].pos = P # overwrites the position of existing vertex with P
            self.skelver[vid].last_update = self._version
        return 

    def _build_search_index(self) -> None:
        if not self.skelver:
            self._pos_cache = None
            self._id_cache = None
            self._kdtree = None
            return
        
        vids = sorted(self.skelver.keys())
        pos = np.vstack([self.skelver[v].pos for v in vids]) # (M,3)
        self._pos_cache = pos
        self._id_cache = np.array(vids, dtype=np.int32)
        self._kdtree = KDTree(pos) if pos.shape[0] >= 1 else None

    def _invalidate_index(self) -> None:
        self._pos_cache = None
        self._id_cache = None
        self._kdtree = None
