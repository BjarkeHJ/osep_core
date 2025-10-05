#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass

@dataclass
class CurrPath:
    path_len: int
    viewpoints: np.ndarray # array of vptids in current path order
    # points: np.ndarray

class PathPlanner:
    def __init__(self, skel_state, occ_map, vpman, drone_pos, max_horizon=10):
        self.skel = skel_state
        self.occ_map = occ_map
        self.vpman = vpman
        self.drone_pos = drone_pos
        self.path: CurrPath = None

        self._max_horizon = max_horizon
        self._start_vptid = None

    def generate_path(self) -> CurrPath | None:
        vpts = [(vptid, vp) for vptid, vp in self.vpman.viewpoints.items() if vp and vp.valid and not getattr(vp, "visited", False)]
        if not vpts:
            self.path = None
            return None
        
        vptid2idx = {vptid: i for i, (vptid, _) in enumerate(vpts)}

        pos = np.vstack([vp.pos for _, vp in vpts]) # all positions of viewpoints
        vids = np.array([vp.vid for _, vp in vpts], dtype=np.int32) # vid for each vpt

        vids_all = getattr(self.skel, "_id_cache", None)
        adj = getattr(self.skel, "adjacency", None)
        if vids_all is None or adj is None or adj.size == 0:
            # purely spatial greedy fallback
            self.path = None
            return None
        
        row_of_vid = {int(v): i for i, v in enumerate(vids_all)}
        deg = (adj > 0).sum(axis=1).astype(np.int32)
        vdeg = np.array([deg[row_of_vid.get(int(vid), -1)] if int(vid) in row_of_vid else 0 for vid in vids], dtype=np.int32)
        vbranch = np.array([self.skel.skelver[int(vid)].seg for vid in vids], dtype=np.int32)

        start_idx = self._choose_start_idx(pos, self.drone_pos)
        if start_idx < 0:
            return None
        
        chosen = [start_idx]
        used = np.zeros(pos.shape[0], dtype=bool)
        used[start_idx] = True

        while len(chosen) < self._max_horizon:
            i = chosen[-1]
            mask = ~used # inverse mask
            if not np.any(mask):
                break # none left

            vi = vids[i]
            ri = row_of_vid.get(int(vi), -1)

            same_vert = np.where(mask & (vids == vi))[0]

            if ri >= 0:
                nb_rows = np.where(adj[ri] > 0)[0]
                nb_vids = set(int(vids_all[r]) for r in nb_rows)
                skel_adjacent = np.where(mask & np.isin(vids, list(nb_vids)))[0]
            else:
                skel_adjacent = np.array([], dtype=np.int32)

            same_branch = np.where(mask & (vids != vi) & ~np.isin(vids, vids[skel_adjacent]) & (vbranch == vbranch[i]))[0]

            others = np.where(mask & (vids != vi) & ~np.isin(vids, vids[skel_adjacent]) & ~(vbranch == vbranch[i]))[0]
            
            j_best = self._pick_with_cost(i, same_vert, skel_adjacent, same_branch, others)

            if j_best < 0:
                break

            used[j_best] = True
            chosen.append(j_best)
        
        chosen_ids = np.array([vpts[k][0] for k in chosen], dtype=np.int64)
        self.path = CurrPath(path_len=chosen_ids.size,
                             viewpoints=chosen_ids)

        return self.path

    def _choose_start_idx(self, pos, drone_pos) -> int:
        if pos is None or pos.size == 0 or drone_pos is None:
            return -1 
        p0 = np.asarray(drone_pos, dtype=np.float32).reshape(3)
        clearance = max(2.0 * float(self.occ_map.voxel_size), 0.0)
        dists = np.linalg.norm(pos - p0[None, :], axis=1)
        for idx in np.argsort(dists):
            if self._has_clear_path(p0, pos[int(idx), clearance]):
                return int(idx)
        return -1


    def _spatial_greedy_order(self, pos, start_pos):
        N = pos.shape[0]
        if N == 0:
            return np.array([],dtype=np.int32)
        used = np.zeros(N, dtype=bool)
        if start_pos is None:
            i = 0
        else:
            d0 = np.linalg.norm(pos - np.asarray(start_pos)[None, :], axis=1)
            i = int(np.argmin(d0))
        
        order = [i]
        used[i] = True
        while len(order) < N:
            p = pos[order[-1]]
            d = np.linalg.norm(pos - p[None,:], axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]):
                break
            used[j] = True
            order.append(j)
        return np.array(order, dtype=np.int32)
        

    def _pick_with_cost(self, i, same_vert, skel_adjacent, same_branch, others, pos):
        p_i = pos[i]
        clearance = 2.0 * float(self.occ_map.voxel_size)

        buckets = [
            (same_vert, 0.0),
            (skel_adjacent, 0.25),
            (same_branch, 0.5),
            (others, 1.5)
        ]

        best_j, best_cost = -1, np.inf
        for cand, penalty in buckets:
            if cand.size == 0:
                continue
            
            d = np.linalg.norm(pos[cand] - p_i[None, :], axis=1)
            cost = d + penalty

            order = np.argsort(cost)
            for k in order:
                j = int(cand[k])
                if self._has_clear_path(p_i, pos[j], clearance):
                    c = float(cost[k])
                    if c < best_cost:
                        best_cost = c
                        best_j = j
                        if penalty <= 0.25:
                            return best_j
                    break
        return best_j
    
    def _has_clear_path(self, p0, p1, clearance: float) -> bool:
        if self.occ_map is None:
            return False
        try:
            return self.occ_map.line_of_sight(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), clearance=float(clearance))
        except Exception:
            return False