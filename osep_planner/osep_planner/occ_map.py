#!/usr/bin/env python3

import numpy as np
from scipy.spatial import cKDTree as KDTree

class OCCMap:
    def __init__(self, voxel_size: float, cam_hfov: float, cam_vfov: float, cam_range: float, kdtree_rebuild_stride: int = 5):
        self.voxel_size = float(voxel_size)
        self.inv_voxel = 1.0 / self.voxel_size

        self.horizontal_fov = cam_hfov
        self.vertical_fov = cam_vfov
        self.max_range = cam_range

        self._occ = set()
        self._centers = np.empty((0,3), dtype=np.float32)
        self._kdtree = None
        self._insert_batches_since_rebuild = 0
        self._kdtree_rebuild_stride = int(kdtree_rebuild_stride)

        self._seen: set[tuple[int, int, int]] = set()
        self._needs_rescore: bool = False

    def insert_centers(self, centers_xyz: np.ndarray) -> None:
        if centers_xyz.size == 0:
            return
        ijk = self._idx_from_centers(centers_xyz)
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            return
        
        uniq = np.unique(ijk, axis=0)
        before = len(self._occ)
        self._occ.update(map(tuple, uniq))
        inserted = len(self._occ) - before
        
        if inserted > 0:
            self._insert_batches_since_rebuild += 1
            if (self._kdtree is None) or (self._insert_batches_since_rebuild >= self._kdtree_rebuild_stride):
                self._rebuild_kdtree()
            self._needs_rescore = True
    
    def clear(self):
        self._occ.clear()
        self._centers = np.empty((0, 3), dtype=np.float32)
        self._kdtree = None
        self._insert_batches_since_rebuild = 0
        self._needs_rescore = True
    
    def is_occupied(self, p: np.ndarray) -> bool:
        return self._voxel_index_of_point(p) in self._occ
    
    def distance_to_structure(self, p: np.ndarray, default_far: float = np.inf) -> float:
        if self._kdtree is None or self._centers.shape[0] == 0:
            return default_far
        
        d, _ = self._kdtree.query(np.asarray(p, dtype=np.float32), k=1)
        return float(d)
    
    def distance_to_structure_along(self, p0: np.ndarray, p1: np.ndarray) -> float:
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)

        L = float(np.linalg.norm(p1 - p0))
        if L == 0.0:
            return 0.0 if self.is_occupied(p0) else np.inf
    
        t_hit, _ = self._ray_cast(p0, p1)
        return (t_hit * L) if (t_hit is not None) else np.inf

    def is_safe(self, p: np.ndarray, margin: float) -> bool:
        return self.distance_to_structure(p) >= float(margin)
    
    def line_of_sight(self, p0: np.ndarray, p1: np.ndarray, clearance: float = 0.0, step_scale: float = 1.0) -> bool:
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)

        t_hit, _ = self._ray_cast(p0, p1)
        if t_hit is not None:
            return False # occupied voxel encountered in direct path
        
        if clearance > 0.0 and self._kdtree is not None and self._centers.shape[0] > 0:
            seg = p1 - p0
            L = float(np.linalg.norm(seg))
            if L == 0.0:
                return self.distance_to_structure(p0) >= clearance
            dir = seg / L
            ds = max(1e-3, self.voxel_size * float(step_scale))
            n = int(np.ceil(L / ds)) + 1
            t = np.linspace(0.0, L, n, dtype=np.float32)
            samples = p0[None, :]+ t[:, None] * dir[None, :]
            d, _ = self._kdtree.query(samples, k=1, workers=-1)
            if np.any(d < clearance):
                return False
        return True
    
    def mark_visible_from(self, cam_pos: np.ndarray, yaw: float, commit: bool = False) -> int:
        if self._kdtree is None or self._centers.shape[0] == 0:
            return 0
        
        max_range = self.max_range
        hfov = self.horizontal_fov
        vfov = self.vertical_fov

        cam = np.asarray(cam_pos, dtype=np.float32)

        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        f = np.array([c, s, 0.0], dtype=np.float32) #foward
        r = np.array([-s, c, 0.0], dtype=np.float32) # right
        u = np.array([0.0, 0.0, 1.0], dtype=np.float32) # up

        h_half = float(hfov) * 0.5
        v_half = float(vfov) * 0.5

        idxs = self._kdtree.query_ball_point(cam, r=max_range)
        if not idxs:
            return 0

        dists = np.linalg.norm(self._centers[idxs] - cam[None, :], axis=1)
        order = np.argsort(dists)
        potential_new: set[tuple[int, int, int]] = set()

        budget = len(order)
        for j in order[:budget]:
            target = self._centers[idxs[j]]
            vec = target - cam
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-6 or dist > max_range:
                continue

            x = float(vec @ f)
            y = float(vec @ r)
            z = float(vec @ u)
            if x <= 0.0:
                continue
            az = np.arctan2(y, x)
            el = np.arctan2(z, x)
            if abs(az) > h_half or abs(el) > v_half:
                continue

            _, hit_ijk = self._ray_cast(cam, target)
            if hit_ijk is None:
                continue

            if (hit_ijk not in self._seen) and (hit_ijk not in potential_new):
                potential_new.add(hit_ijk)

        # commit to the map if seen - Used when viewpoint reached by uav
        if commit and potential_new:
            before = len(self._seen)
            self._seen.update(potential_new)
            if len(self._seen) > before:
                self._needs_rescore = True

        new_seen = len(potential_new)
        return new_seen

    def _rebuild_kdtree(self):
        if len(self._occ) == 0:
            self._centers = np.empty((0,3), dtype=np.float32)
            self._kdtree = None
        else:
            ijk = np.array(list(self._occ), dtype=np.int32)
            self._centers = self._index_to_center(ijk).astype(np.float32, copy=False)
            self._kdtree = KDTree(self._centers) if self._centers.shape[0] > 0 else None
        self._insert_batches_since_rebuild = 0
    
    def _ray_cast(self, p0: np.ndarray, p1: np.ndarray):
        v0 = p0 * self.inv_voxel
        v1 = p1 * self.inv_voxel

        ix, iy, iz = np.floor(v0).astype(np.int32)
        ex, ey, ez = np.floor(v1).astype(np.int32)

        tx, ty, tz = v1 - v0
        sx = 1 if tx > 0 else (-1 if tx < 0 else 0)
        sy = 1 if ty > 0 else (-1 if ty < 0 else 0)
        sz = 1 if tz > 0 else (-1 if tz < 0 else 0)

        invx = abs(1.0 / tx) if tx != 0.0 else np.inf
        invy = abs(1.0 / ty) if ty != 0.0 else np.inf
        invz = abs(1.0 / tz) if tz != 0.0 else np.inf

        def frac(a):
            return a - np.floor(a)
        
        tMaxX = (1.0 - frac(v0[0])) * invx if sx > 0 else (frac(v0[0]) * invx if sx < 0 else np.inf)
        tMaxY = (1.0 - frac(v0[1])) * invy if sy > 0 else (frac(v0[1]) * invy if sy < 0 else np.inf)
        tMaxZ = (1.0 - frac(v0[2])) * invz if sz > 0 else (frac(v0[2]) * invz if sz < 0 else np.inf)

        if (ix, iy, iz) in self._occ:
            return 0.0, (ix, iy, iz)
        
        t = 0.0
        max_steps = int(3 * (abs(ex - ix) + abs(ey - iy) + abs(ez - iz) + 3)) + 8
        steps = 0
        while (ix != ex or iy != ey or iz != ez) and steps < max_steps:
            if tMaxX <= tMaxY and tMaxX <= tMaxZ:
                t = tMaxX
                ix += sx
                tMaxX += invx
            elif tMaxY <= tMaxX and tMaxY <= tMaxZ:
                t = tMaxY
                iy += sy
                tMaxY += invy
            else:
                t = tMaxZ
                iz += sz
                tMaxZ += invz
            
            if (ix, iy, iz) in self._occ:
                return float(min(max(t, 0.0), 1.0)), (ix, iy, iz) # hit 0/1, parameter along direction

            steps += 1
        return None, None
    
    def _occ_seen_count(self) -> int:
        return len(self._seen)
    
    def _occ_total_count(self) -> int:
        return len(self._occ)
    
    def _get_seen_occ_centers(self) -> np.ndarray:
        if not self._seen:
            return np.empty((0,3), dtype=np.float32)
        ijk = np.array(list(self._seen), dtype=np.int32)
        return self._index_to_center(ijk).astype(np.float32, copy=False)
    
    @staticmethod
    def _round_int(x):
        return np.floor(x + 0.5).astype(np.int32)

    def _idx_from_centers(self, pts: np.ndarray) -> np.ndarray:
        return self._round_int(pts * self.inv_voxel - 0.5)

    def _index_to_center(self, ijk: np.ndarray) -> np.ndarray:
        return (ijk.astype(np.float32) + 0.5) * self.voxel_size

    def _voxel_index_of_point(self, p: np.ndarray) -> tuple[int, int, int]:
        v = np.asarray(p, dtype=np.float32) * self.inv_voxel
        ijk = np.floor(v).astype(np.int32)
        return (int(ijk[0]), int(ijk[1]), int(ijk[2]))
