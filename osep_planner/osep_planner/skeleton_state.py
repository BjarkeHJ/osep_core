#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import csr_matrix as csr

# State Containers
@dataclass
class Vertex:
    vid: int
    pos: np.ndarray
    type: int
    seg: int # segment/branch id
    # nbs: np.ndarray # Neighboring vertices
    last_update: int # when this was last updated

class SkeletonState:
    def __init__(self, search_radius=3.0, update_th = 2.0):
        self.skelver: dict[int, Vertex] = {}
        self.adjacency: np.ndarray = np.array([])
        self.branches: dict[int, np.ndarray] = {}
        self.branch_ids: np.ndarray = np.array([])
        
        self._search_r2 = search_radius**2
        self._update_th = update_th
        self._next_id = 0

        self._pos_cache = None # (M,3)
        self._id_cache = None # (M,)
        self._kdtree = None

        self._version = 0

        self._color2seg: dict[int, int] = {}
        self._next_seg_id: int = 0

        self._deleted_vids: list[int] = []

        self._adjacency_vids = None

    def get_skeleton(self) -> np.ndarray:
        if not self.skelver:
            return np.empty((0, 4), dtype=np.float64)
        vids = sorted(self.skelver.keys())
        rows = [[vid, *self.skelver[vid].pos] for vid in vids]
        return np.array(rows, dtype=np.float64)

    def get_adjacency_and_vids(self):
        return self.adjacency, getattr(self, "_adjacency_vids", None)

    def get_deleted_vids(self) -> list[int]:
        return self._deleted_vids

    def get_size(self) -> int:
        return len(self.skelver)
    
    def current_version(self) -> int:
        return self._version

    def get_vertices_updated_since(self, since_version: int) -> list[Vertex]:
        return [v for v in self.skelver.values() if v.last_update > since_version]
    
    def get_vertex_ids_updated_since(self, since_version: int) -> list[int]:
        return [vid for vid, v in self.skelver.items() if v.last_update > since_version]

    def update_skeleton(self, points_xyz: np.ndarray, rgb_u32: np.ndarray):
        self._deleted_vids = []
        
        if points_xyz is None or points_xyz.size == 0:
            return
        
        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        n_in = pts.shape[0]

        if rgb_u32 is None or rgb_u32.size != n_in:
            segs_in = np.full((n_in), -1, dtype=np.int32)
        else:
            rgb_keys = np.vectorize(self._rgbkey_from_u32)(rgb_u32.astype(np.uint32, copy=False))
            segs_in = np.array([self._seg_from_rgbkey(k) for k in rgb_keys], dtype=np.int32)

        self._version += 1 # update version
        self._build_search_index()

        touched_vids: set[int] = set()

        if self._pos_cache is None or self._pos_cache.shape[0] == 0:
            # No existing skeleton -> Spawn for all new
            for p, seg in zip(pts, segs_in):
                vid_new = self._create_vertex(p, seg)
                touched_vids.add(vid_new)

            self._deleted_vids = self._prune_unseen(touched_vids)
            self._post_update_branches()
            self._update_adjacency()
            self._set_vertex_type()
            return

        # For each point, get ALL candidate vertex indices within r
        r = float(np.sqrt(self._search_r2))
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
            self._update_vertex(vid, pts[i], segs_in[i])   # pass single (3,) point
            matched_points.add(i)
            matched_verts.add(j)
            touched_vids.add(vid)

        # Any unassigned points become new vertices
        for i, p in enumerate(pts):
            if i not in matched_points:
                vid_new = self._create_vertex(p, segs_in[i])
                touched_vids.add(vid_new)

        self._deleted_vids = self._prune_unseen(touched_vids)
        self._post_update_branches()
        self._update_adjacency()
        self._set_vertex_type()

        print(f"Number of segments: {len(self.branch_ids)}")
        print(f"Branch IDs: {self.branch_ids}")

    def _create_vertex(self, p: np.ndarray, seg: int) -> int:
        vid = self._next_id
        self._next_id += 1 #increment next id
        self.skelver[vid] = Vertex(vid=vid, 
                                   pos=np.array(p, dtype=np.float64),
                                   type=0,
                                   seg=int(seg),
                                   last_update=self._version) # creates new entry with key vid
        return vid

    def _update_vertex(self, vid: int, P: np.ndarray, seg_in: int) -> None:
        v = self.skelver[vid]
        if np.linalg.norm(v.pos - P) >= self._update_th:
            v.pos = P # overwrites the position of existing vertex with P
            v.last_update = self._version
        
        # Update if unknown
        if v.seg < 0 and seg_in >= 0:
            v.seg = int(seg_in)

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

    def _post_update_branches(self) -> None:
        if not self.skelver:
            self.branches = {}
            self.branch_ids = np.array([], dtype=np.int32)
            return
        vids = np.array(sorted(self.skelver.keys()), dtype=np.int32)
        segs = np.array([self.skelver[int(vid)].seg for vid in vids], dtype=np.int32)
        unique_segs, inverse = np.unique(segs, return_inverse=True)

        branches_clean = {}
        valid_branch_ids = []
        for i, seg in enumerate(unique_segs):
            segs_vids = vids[inverse == i]
            if segs_vids.size == 0:
                continue # skip empty branhces
            branches_clean[int(seg)] = segs_vids
            valid_branch_ids.append(int(seg))

        self.branches = branches_clean
        self.branch_ids = np.array(valid_branch_ids, dtype=np.int32)

    def _prune_unseen(self, touched_vids: set[int]) -> list[int]:
        if not self.skelver:
            return []
        deleted = []
        for vid in list(self.skelver.keys()):
            if vid not in touched_vids:
                deleted.append(vid)
                del self.skelver[vid]
        return deleted
        
    @staticmethod
    def _rgbkey_from_u32(u : np.uint32) -> int:
        # Mast to lower 24 bits
        return int(u & np.uint32(0x00FFFFFF))

    def _quantize_rgb_key(self, key: int, step: int = 16) -> int:
        r = (key >> 16) & 0xFF
        g = (key >> 8) & 0xFF
        b = (key) & 0xFF
        rq = (r // step) * step
        gq = (g // step) * step
        bq = (b // step) * step
        return (rq << 16) | (gq << 8) | bq

    def _seg_from_rgbkey(self, key: int) -> int:
        key = self._quantize_rgb_key(key, step = 16)
        seg = self._color2seg.get(key, None)
        if seg is None:
            seg = self._next_seg_id
            self._color2seg[key] = seg
            self._next_seg_id += 1
        return seg

    def _update_adjacency(self, k: int = 2) -> None:
        """
        Build adjacency matrix by connecting each point in each branch to its k nearest neighbors.
        Edge points (endpoints) are only connected to their single closest neighbor.
        Args:
            k (int): Number of neighbors to connect for each non-endpoint (default: 2)
        """
        n = len(self.skelver)
        if n <= 1:
            self.adjacency = np.zeros((n, n), dtype=np.float64)
            self._adjacency_vids = np.array(sorted(self.skelver.keys()), dtype=np.int32) if n == 1 else None
            return
        self._build_search_index()
        vids = self._id_cache
        P = self._pos_cache
        id2row = {int(v): i for i, v in enumerate(vids)}
        W = np.zeros((n, n), dtype=np.float64)
        for seg in self.branch_ids:
            if seg < 0:
                continue
            seg_vids = self.branches.get(int(seg))
            if seg_vids is None or seg_vids.size == 0:
                continue
            rows = np.array([id2row[int(v)] for v in seg_vids], dtype=np.int32)
            m = rows.size
            if m <= 1:
                continue
            Pseg = P[rows]
            tree = KDTree(Pseg)
            # Find k+1 neighbors for all points (including self)
            dists, idxs = tree.query(Pseg, k=min(k+1, m))
            # For each point, compute sum of distances to k nearest neighbors (excluding self)
            sum_dists = dists[:, 1:k+1].sum(axis=1)
            # Endpoints: those with largest and smallest sum of distances (or just the two extremes)
            if m > 2:
                endpoint_indices = [int(np.argmin(sum_dists)), int(np.argmax(sum_dists))]
            else:
                endpoint_indices = list(range(m))
            for i in range(m):
                ni = rows[i]
                if i in endpoint_indices:
                    # Only connect to single closest neighbor (not self)
                    j = idxs[i, 1]
                    nj = rows[j]
                    d = dists[i, 1]
                    if ni != nj and (W[ni, nj] == 0.0 or d < W[ni, nj]):
                        W[ni, nj] = d
                        W[nj, ni] = d
                else:
                    # Connect to k nearest neighbors (not self)
                    for jidx in range(1, min(k+1, m)):
                        j = idxs[i, jidx]
                        nj = rows[j]
                        d = dists[i, jidx]
                        if ni != nj and (W[ni, nj] == 0.0 or d < W[ni, nj]):
                            W[ni, nj] = d
                            W[nj, ni] = d
        self._adjacency_vids = vids.copy()
        self.adjacency = np.minimum(W, W.T)

    def _set_vertex_type(self):
        if self.adjacency.size == 0:
            for v in self.skelver.values():
                v.type = 0
            return
        
        vids = self._id_cache
        deg = (self.adjacency > 0).sum(axis=1).astype(int)

        for i, vid in enumerate(vids):
            v = self.skelver[int(vid)]
            d = int(deg[i])
            if d == 1: # leaf
                v.type = 1
            elif d == 2: # branch
                v.type = 2
            elif d > 2: # junction
                v.type = 3
            else: 
                v.type = 0            