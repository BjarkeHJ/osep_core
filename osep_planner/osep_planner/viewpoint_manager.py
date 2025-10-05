#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class Viewpoint:
    vptid: int #unique viewpoint id
    vid: int #parent vertex id
    pos: np.ndarray
    yaw: float
    score: float = 0.0
    visited: bool = False
    valid: bool = True
    u_coeff: np.ndarray | None = None

class ViewpointManager:
    def __init__(self, skel_state, occ_map, safe_dist):
        self.skeleton = skel_state # reference to SkeletonState
        self.occ_map = occ_map # Current occupancy map

        self.viewpoints: dict[int, Viewpoint] = {} # vptid -> Viewpoint
        self._ids_by_vid: dict[int, list[int]] = {} # vid -> [vptid, ...]
        self._next_vptid = 0

        self._last_skel_version = -1 # for synchronization
        self._safe_dist = safe_dist

        self._prev_type_by_vid: dict[int, int] = {} # Last known vertex type
        self._frame_by_vid: dict[int, list[np.ndarray]] = {} # list of frame coefficients per vertex

    def get_viewpoints(self) -> list:
        return [vp for vp in self.viewpoints.values() if vp.valid]

    def mark_visited(self, vptid: int) -> None:
        # Should propagate into occ_map... 
        vp = self.viewpoints.get(vptid)
        if vp and vp.valid:
            vp.visited = True

    def update_viewpoints(self) -> None:
        if self.occ_map is None:
            print("No Occupancy Map - Exiting")
            return

        cur_ver = self.skeleton.current_version()
        if cur_ver == self._last_skel_version:
            return # no version change
        
        # Deleted viewpoints corresponding to deleted vertices
        deleted_vids = getattr(self.skeleton, "get_deleted_vids", lambda: [])()
        print(f"Number of deleted vertices: {len(deleted_vids)}")
        for dvid in deleted_vids:
            vptids = self._ids_by_vid.pop(dvid, [])
            for dvpt in vptids:
                self.viewpoints.pop(dvpt, None)
            self._prev_type_by_vid.pop(dvid, None)
            self._frame_by_vid.pop(dvid, None)

        # Update and/or generate viewpoints for updated/new vertice
        changed_vids = self.skeleton.get_vertex_ids_updated_since(self._last_skel_version)
        print(f"Number of updated vertices: {len(changed_vids)}")
        
        for vid in changed_vids:
            v = self.skeleton.skelver.get(vid)
            if v is None:
                continue
            
            prev_type = self._prev_type_by_vid.get(vid, None)
            type_changed = (prev_type is not None) and (prev_type != v.type)
            have_bucket = vid in self._ids_by_vid and len(self._ids_by_vid[vid]) > 0
            have_frame = vid in self._frame_by_vid and len(self._frame_by_vid[vid]) > 0

            if v.type < 1:
                # Invalid
                old = self._ids_by_vid.pop(vid, [])
                for oid in old:
                    self.viewpoints.pop(oid, None)
                self._frame_by_vid.pop(vid, None)
                self._prev_type_by_vid[vid] = v.type
                continue

            if (not have_bucket) or type_changed or (not have_frame):
                # spawn fresh viewpoints
                new_vpts, u_coeffs = self._make_viewpoint(vid)
                self._replace_for_vid(vid, new_vpts)
                self._frame_by_vid[vid] = u_coeffs
            else:
                # Update in place
                self._update_in_place_for_vid(vid)
            
            self._prev_type_by_vid[vid] = v.type
        
        self._check_viewpoints() # Should be fixed to move viewpoints backwards OR trigger resample?

        self._last_skel_version = cur_ver # update stored version 

    def _make_viewpoint(self, vid: int):
        v = self.skeleton.skelver[vid]
        if v.type < 1:
            return [], []
        
        MAX_TRIES = 10
        step = max(0.5 * self.occ_map.voxel_size, 0.5)

        vpos = v.pos
        z_ref = float(vpos[2])
        safe_r = float(self._safe_dist)
        that, n1, n2 = self._build_local_frame(vid) # tanget, normal1, normal2

        out_vpts = []
        u_coeffs = []

        def vpt_gen(u: np.ndarray, r: float, z: float):
            # Appends viewpoint to out_vpts based on direction vector u, safe distance r, and reference z value
            nonlocal out_vpts
            u = self._normalize(u)
            if u.shape != (3,) or float(np.linalg.norm(u) < 1e-8):
                return
            
            dtfs = 0.0
            d = safe_r + dtfs
            p = np.array([vpos[0] + d * u[0], vpos[1] + d * u[1], z], dtype=np.float64)
            tries = 0 # implement adjustments via octree or the likes
            while tries < MAX_TRIES and not self._is_position_safe(p, 0.9 * safe_r):
                p += step * u
                tries += 1

            if not self._is_position_safe(p, 0.9 * safe_r):
                return # give up

            yaw = self._yaw_to_face(p, vpos)
            vp = Viewpoint(
                vptid = -1,
                vid = int(vid),
                pos = p,
                yaw = float(yaw),
                visited = False,
                valid = True,
                u_coeff = u
            )
            out_vpts.append(vp)
            u_coeffs.append(u)

        if abs(float(that[2])) > max(abs(float(that[0])), abs(float(that[1]))):
            for u in (n1, n2, -n1, -n2):
                vpt_gen(u, safe_r, z_ref)
        else:
            vpt_gen(n1, safe_r, z_ref)
            vpt_gen(-n1, safe_r, z_ref)
            if v.type == 1:
                u1 = self._normalize(-that + n1)
                u2 = self._normalize(-that - n1)
                for u in (-that, u1, u2):
                    vpt_gen(u, safe_r, z_ref)

        return out_vpts, u_coeffs

    def _update_in_place_for_vid(self, vid: int) -> None:
        v = self.skeleton.skelver.get(vid)
        vpos = v.pos # Position of vertex (updated)
        z_ref = float(vpos[2])
        safe_r = float(self._safe_dist)

        ids = list(self._ids_by_vid.get(vid, []))
        for vptid in ids:
            vp = self.viewpoints.get(vptid)
            if not vp or not vp.valid:
                continue
            if vp.u_coeff is None:
                vp.u_coeff = np.array([1,0,0], dtype=np.float64)

            MAX_TRIES = 10
            step = max(0.5 * self.occ_map.voxel_size, 0.5)
            u = vp.u_coeff if (vp.u_coeff is not None) else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            u = self._normalize(u)
            d = safe_r # plus dtfs
            p = np.array([vpos[0] + d*u[0], vpos[1] + d*u[1], z_ref], dtype=np.float64)
            tries = 0
            while tries < MAX_TRIES and not self._is_position_safe(p, 0.9 * safe_r):
                d += step
                p = np.array([vpos[0] + d*u[0], vpos[1] + d*u[1], z_ref], dtype=np.float64)
                tries += 1
            
            if not self._is_position_safe(p, 0.9 * safe_r):
                # delete viewpoint 
                self.viewpoints.pop(vptid, None)
                bucket = self._ids_by_vid.get(vid)
                if bucket:
                    try:
                        bucket.remove(vptid)
                    except ValueError:
                        pass
                continue
            
            vp.pos = p
            vp.yaw = float(self._yaw_to_face(vp.pos, vpos))
            vp.valid = True

    def _replace_for_vid(self, vid, vpts) -> None:
        old = self._ids_by_vid.pop(int(vid), [])
        for oid in old:
            self.viewpoints.pop(oid, None)
        self._insert_batch(vid, vpts)

    def _build_local_frame(self, vid: int):
        that = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n1hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n2hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        v = self.skeleton.skelver.get(vid)
        if v is None or v.type < 0:
            return that, n1hat, n2hat
        
        vids = getattr(self.skeleton, "_id_cache", None)
        P = getattr(self.skeleton, "_pos_cache", None)
        adj = getattr(self.skeleton, "adjacency", None)

        if vids is None or P is None or adj is None or adj.size == 0:
            print("error...")
            return that, n1hat, n2hat

        hits = np.where(vids == int(vid))[0]
        if hits.size == 0:
            print("Error 3")
            return that, n1hat, n2hat
        i = int(hits[0])

        # Get position of vertex
        p_i = v.pos
        nb_rows = np.where(adj[i] > 0)[0]
        k = nb_rows.size #number of nbs

        if k == 1:
            e = P[nb_rows[0]] - p_i 
            if float(np.linalg.norm(e)) > 1e-6:
                that = self._normalize(e)            
        elif k >= 2:
            E = (P[nb_rows] - p_i).T
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            that = self._normalize(U[:,0])
        
        if np.linalg.norm(that) < 1e-6:
            that = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        xax = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # prefer up x that unless nearly parallel -> else x x that
        if abs(float(np.dot(that, up))) < 0.95:
            n1hat = self._normalize(np.cross(up, that))
        else:
            n1hat = self._normalize(np.cross(xax, that))

        n2hat = self._normalize(np.cross(that, n1hat))

        return that, n1hat, n2hat

    def _insert_batch(self, vid: int, vpts: list[Viewpoint]) -> None:
        bucket = self._ids_by_vid.setdefault(int(vid), [])
        for vp in vpts:
            if vp.vptid is None or vp.vptid < 0:
                vp.vptid = self._next_vptid # give unique viewpoint id
                self._next_vptid += 1 #increment

            self.viewpoints[vp.vptid] = vp
            bucket.append(vp.vptid)

    def _is_active(self, vptid: int) -> bool:
        vp = self.viewpoints.get(vptid)
        return bool(vp and vp.valid and not vp.visited)

    def _is_position_safe(self, p: np.ndarray, margin: float) -> bool:
        if self.occ_map is None:
            return True
        return self.occ_map.is_safe(p, margin)

    def _has_clear_los(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        if self.occ_map is None:
            return True
        c = self._safe_dist
        return self.occ_map.line_of_sight(p0, p1, c)
    
    def _distance_to_structure(self, p: np.ndarray) -> float:
        if self.occ_map is None:
            return np.inf
        return self.occ_map.distance_to_structure(p)

    def _check_viewpoints(self) -> None:
        if self.occ_map is None:
            return
        
        items = [(vptid, vp) for vptid, vp in list(self.viewpoints.items()) if vp and vp.valid]
        if not items:
            return
        
        margin = 0.9*float(self._safe_dist)

        kdt = getattr(self.occ_map, "_kdtree", None)
        centers = getattr(self.occ_map, "_centers", None)

        if kdt is not None and centers is not None and centers.shape[0] > 0:
            P = np.vstack([vp.pos for _, vp in items]).astype(np.float32, copy=False)
            d, _ = kdt.query(P, k=1, workers=-1)

            to_del_idx = np.where(d < margin)[0]
            for i in to_del_idx[::-1]:
                vptid, vp = items[i]
                self.viewpoints.pop(vptid, None)
                bucket = self._ids_by_vid.get(vp.vid)
                if bucket:
                    try:
                        bucket.remove(vptid, None)
                    except ValueError:
                        pass
                items.pop(i)

        else:
            print("Error! Could not get kd tree")
            

    @staticmethod
    def _yaw_to_face(dir: np.ndarray, tgt: np.ndarray) -> float:
        # Yaw value to face target tgt from camera dir
        d = tgt[:2] - dir[:2]
        return np.atan2(d[1], d[0])
    
    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        return R.from_euler('z', yaw).as_quat()

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v if n < eps else v / n
    