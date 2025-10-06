#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, field

@dataclass
class CurrPath:
    viewpoints: list[int] = field(default_factory=list) # list of vptids in current path order
    path_len: int = 0

class PathPlanner:
    def __init__(self, skel_state, occ_map, vpman, drone_pos, max_horizon=10):
        self.skel = skel_state
        self.occ_map = occ_map
        self.vpman = vpman
        self.drone_pos = drone_pos
        self.path: CurrPath = None

        self._max_horizon = max_horizon
        self._start = -1

        self.w_dist = 1.0
        self.w_pen = 1.0
        self.w_score = 1.0

    @staticmethod
    def _norm_scores(raw_scores: np.ndarray) -> np.ndarray:
        s = np.array(raw_scores, dtype=np.float32)
        s = np.log1p(np.maximum(s, 0.0))
        lo = float(s.min(initial=0.0))
        hi = float(s.max(initial=0.0))
        span = hi - lo
        return (s - lo) / span if span > 1e-6 else np.zeros_like(s, dtype=np.float32)
    
    def generate_path(self) -> None:
        if self.path is None:
            self.path = CurrPath([])

        if self.path.path_len >= self._max_horizon:
            return

        slots_left = self._max_horizon - self.path.path_len

        vpts = [(vptid, vp) for vptid, vp in self.vpman.viewpoints.items() if vp and vp.valid and not getattr(vp, "visited", False)]
        if not vpts:
            return
        
        vptid2idx = {vptid: i for i, (vptid, _) in enumerate(vpts)} # map vptid -> idx 
        pos = np.vstack([vp.pos for _, vp in vpts]) # all positions of viewpoints pos[x] -> pos_x
        vids = np.array([vp.vid for _, vp in vpts], dtype=np.int32) # vid for each valid and unvisited vpt
        vids_all = getattr(self.skel, "_id_cache", None) # all vids
        adj = getattr(self.skel, "adjacency", None)

        raw_scores = np.array([getattr(vp, "score", 0.0) for _, vp in vpts], dtype=np.float32)
        scores = self._norm_scores(raw_scores)

        if vids_all is None or adj is None or adj.size == 0:
            # purely spatial greedy fallback? ...
            return

        vid2idx = {int(v): i for i, v in enumerate(vids_all)} # map vid -> idx
        
        deg = (adj > 0).sum(axis=1).astype(np.int32) #adjacency degree
        vdeg = np.array([deg[vid2idx.get(int(vid), -1)] if int(vid) in vid2idx else 0 for vid in vids], dtype=np.int32) # degree for valid vids
        vbranch = np.array([self.skel.skelver[int(vid)].seg for vid in vids], dtype=np.int32) # branch id for valid vids

        if self.path.path_len == 0:
        # if self._start < 0:
            self._start = self._choose_start_idx(pos, self.drone_pos)
            if self._start < 0:
                return # could not find starting index
            init = True
        else:
            last_vptid = self.path.viewpoints[-1] # increment from last in path
            self._start = vptid2idx.get(int(last_vptid), -1)
            if self._start < 0:
                self._start = self._choose_start_idx(pos, self.drone_pos)
                if self._start < 0:
                    return
                init = True
            else:
                init = False

        used = np.zeros(pos.shape[0], dtype=bool) # used this search and current path
        for pid in self.path.viewpoints:
            j = vptid2idx.get(int(pid))
            if j is not None:
                used[j] = True
        used[self._start] = True
        chosen = [self._start] # what will be appended to the path

        while len(chosen) - (0 if init else 1) < slots_left:
            i = chosen[-1]
            mask = ~used # inverse used mask (available) -> mask should exclude viewpoints in current path
            if not np.any(mask):
                break # none left

            vi = vids[i] #vid
            ri = vid2idx.get(int(vi), -1) #index
            
            if ri >= 0:
                nb_rows = np.where(adj[ri] > 0)[0]
                nb_vids = [int(vids_all[r]) for r in nb_rows]
                skel_adjacent = np.where(mask & np.isin(vids, nb_vids))[0] # subset of viewpoints from adjacent vertices to vi
            else:
                skel_adjacent = np.array([], dtype=np.int32)

            same_vert = np.where(mask & (vids == vi))[0] # subset of viewpoints from same vertex vi
            same_branch = np.where(mask & (vids != vi) & ~np.isin(vids, vids[skel_adjacent]) & (vbranch == vbranch[i]))[0] # subset of viewpoints from same branch segment
            others = np.where(mask & (vids != vi) & ~np.isin(vids, vids[skel_adjacent]) & ~(vbranch == vbranch[i]))[0] # remaining unused subset
            
            j_best = self._pick_with_cost(i, same_vert, skel_adjacent, same_branch, others, pos, scores)

            if j_best < 0:
                break # could not find suitable viewpoint 

            used[j_best] = True
            chosen.append(j_best)
        
        chosen_ids = np.array([vpts[k][0] for k in chosen], dtype=np.int64) # index to viewpoint id
        if not init and chosen_ids.size > 0:
            chosen_ids = chosen_ids[1:] # remove duplicate if not chosen from empty path
        for id in chosen_ids:
            self.path.viewpoints.append(id)
        self.path.path_len = len(self.path.viewpoints)

    def _choose_start_idx(self, pos, drone_pos) -> int:
        if pos is None or pos.size == 0 or drone_pos is None:
            return -1 
        p0 = np.asarray(drone_pos, dtype=np.float32).reshape(3)
        clearance = max(5.0 * float(self.occ_map.voxel_size), 0.0)
        dists = np.linalg.norm(pos - p0[None, :], axis=1)
        for idx in np.argsort(dists):
            if self._has_clear_path(p0, pos[int(idx)], clearance):
                return int(idx)
        return -1

    def _pick_with_cost(self, i, same_vert, skel_adjacent, same_branch, others, pos, scores):
        p_i = pos[i]
        clearance = 2.0 * float(self.occ_map.voxel_size)

        # viewpoint class and corresponding penalty
        buckets = [
            (same_vert, 0.0),
            (skel_adjacent, 0.25),
            (same_branch, 0.5),
            (others, 1.5)
        ]

        best_j = -1
        best_cost = np.inf
        for cand, penalty in buckets:
            if cand.size == 0:
                continue
            
            d = np.linalg.norm(pos[cand] - p_i[None, :], axis=1) # distances from vertex i to candidate
            s = scores[cand]
            cost = self.w_dist*d + self.w_pen*penalty - self.w_score*s # cost based on distance and class

            order = np.argsort(cost) # indices sorted by smallest to largest cost
            for k in order:
                j = int(cand[k])
                if self._has_clear_path(p_i, pos[j], clearance):
                    c = float(cost[k])
                    if c < best_cost:
                        best_cost = c
                        best_j = j
                        # if penalty <= 0.25:
                            # return best_j
                    break
        return best_j
    
    def _refine_path(self):
        # Refinement: Make the current path order minimize the total distance of the path (linear segments)
        pass

    def _has_clear_path(self, p0, p1, clearance: float) -> bool:
        if self.occ_map is None:
            return False
        try:
            return self.occ_map.line_of_sight(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), clearance=float(clearance))
        except Exception:
            return False
        

