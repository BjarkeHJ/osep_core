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


class ViewpointManager:
    def __init__(self, skel_state, safe_dist):
        self.skeleton = skel_state # reference to SkeletonState
        self.viewpoints: dict[int, Viewpoint] = {} # vptid -> Viewpoint
        
        self._ids_by_vid: dict[int, list[int]] = {} # vid -> [vptid, ...]
        self._next_vptid = 0

        self._last_skel_version = -1 # for synchronization
        self._safe_dist = safe_dist

    def get_viewpoints(self) -> list:
        return [vp for vp in self.viewpoints.values() if vp.valid]

    def mark_visited(self, vptid: int) -> None:
        vp = self.viewpoints.get(vptid)
        if vp and vp.valid:
            vp.visited = True

    def update_viewpoints(self) -> None:
        cur_ver = self.skeleton.current_version()
        if cur_ver == self._last_skel_version:
            return # no version change

        # Fetch updates in skeleton structure
        # changed_vers = self.skeleton.get_vertices_updated_since(self._last_skel_version)
        # print(f"Number of updated vertices: {len(changed_vers)}")

        changed_vids = self.skeleton.get_vertex_ids_updated_since(self._last_skel_version)
        print(f"Number of updated vertices: {len(changed_vids)}")
        
        for vid in changed_vids:
            new_vpts = self._make_viewpoint(vid)
            if not new_vpts:
                continue

            print(f"{len(new_vpts)} new viewpoints generated for Vertex {vid}")
            self._insert_batch(vid, new_vpts)
        
        self._last_skel_version = cur_ver # update stored version 

    def _make_viewpoint(self, vid: int):
        v = self.skeleton.skelver[vid]
        if v.type < 1:
            return []
        
        MAX_TRIES = 5

        vpos = v.pos
        z_ref = float(vpos[2])
        safe_r = float(self._safe_dist)

        dir, n1, n2 = self._build_local_frame(vid) # tanget, normal1, normal2
        print(f"[VPT] dir={dir.round(3)}, n1={n1.round(3)}, n2={n2.round(3)}")


        out_vpts = []

        def vpt_gen(u: np.ndarray, r: float, z: float):
            # Appends viewpoint to out_vpts based on direction vector u, safe distance r, and reference z value
            nonlocal out_vpts
            u = self._normalize(u)
            if u.shape != (3,) or float(np.linalg.norm(u) < 1e-8):
                return
            
            dtfs = 0.0 # implement distance to free space!
            d = safe_r + dtfs
            p = np.array([vpos[0] + d * u[0], vpos[1] + d * u[1], z], dtype=np.float64)
            tries = 0 # implement adjustments via octree or the likes

            yaw = self._yaw_to_face(p, vpos)

            vp = Viewpoint(
                # vptid = self._next_vptid,
                vptid = -1,
                vid = int(vid),
                pos = p,
                yaw = float(yaw),
                visited = False,
                valid = True
            )

            # self._next_vptid += 1 # update vpt
            out_vpts.append(vp)

        if abs(float(dir[2])) > max(abs(float(dir[0])), abs(float(dir[1]))):
            for u in (n1, n2, -n1, -n2):
                vpt_gen(u, safe_r, z_ref)
        else:
            vpt_gen(n1, safe_r, z_ref)
            vpt_gen(-n1, safe_r, z_ref)
            if v.type == 1:
                u1 = self._normalize(-dir + n1)
                u2 = self._normalize(-dir - n1)
                for u in (-dir, u1, u2):
                    vpt_gen(u, safe_r, z_ref)
        
        return out_vpts

    def _score_viewpoint(vptid: int):
        pass

    def _build_local_frame(self, vid: int):
        that = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n1hat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n2hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        v = self.skeleton.skelver.get(vid)
        if v.type < 0:
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
    