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

        self.plan_version = 0
        self._max_horizon = max_horizon
        self._start = -1

        self.w_score = 1.0

        self.w_dist = 1.5
        self.w_pen = 10.0
        self.w_turn = 1.0

        self.c_same_vert = 0.0
        self.c_adj_vert = 0.05
        self.c_same_branch = 0.5
        self.c_others = 2.0

        self._next_exec_idx = 0
        self._lock_ahead = 1
        self._replan_cooldown_ticks = 10
        self._replan_cooldown = 0
        self._prefer_old_margin = 0.5 # 50% increase needed

    @staticmethod
    def _norm_scores(raw_scores: np.ndarray) -> np.ndarray:
        s = np.array(raw_scores, dtype=np.float32)
        s = np.log1p(np.maximum(s, 0.0))
        lo = float(s.min(initial=0.0))
        hi = float(s.max(initial=0.0))
        span = hi - lo
        return (s - lo) / span if span > 1e-6 else np.zeros_like(s, dtype=np.float32)
    
    @staticmethod
    def _last_segment_dir(pos: np.ndarray, chosen: list[int]) -> np.ndarray | None:
        if len(chosen) < 2:
            return None
        p_prev = pos[chosen[-2]]
        p_curr = pos[chosen[-1]]
        v = p_curr - p_prev
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return None
        return v / n
    
    @staticmethod
    def _turn_penalty(v_last: np.ndarray | None, p_i: np.ndarray, p_j: np.ndarray) -> float:
        if v_last is None:
            return 0.0
        v = p_j - p_i
        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return 0.0
        v /= n
        p = (0.5 * (1.0 - float(np.dot(v_last, v))))**2
        return p

    def _decay_cooldown(self):
        if self._replan_cooldown > 0:
            self._replan_cooldown -= 1

    def _commit_if_changed(self, candidate_ids: list[int]) -> bool:
        cur = self.path.viewpoints
        if len(candidate_ids) == len(cur) and all(int(a) == int(b) for a, b in zip(candidate_ids, cur)):
            return False #nothing changed
        self.path.viewpoints = list(map(int, candidate_ids))
        self.path.path_len = len(self.path.viewpoints)
        self.plan_version += 1
        return True

    def generate_path(self) -> bool:
        self._decay_cooldown() 
        committed = False

        if self.path is None:
            self.path = CurrPath([])

        current_ids = list(self.path.viewpoints) # snapshot
         
        anchor_idx = min(self._next_exec_idx + self._lock_ahead, max(len(current_ids) - 1, 0)) if current_ids else -1
        old_suffix_first = current_ids[anchor_idx + 1] if (anchor_idx >= 0 and len(current_ids) > anchor_idx + 1) else None
        prefix_ids = current_ids[:anchor_idx + 1] if anchor_idx >= 0 else []

        if len(prefix_ids) >= self._max_horizon:
            return False
        
        # slots_left = self._max_horizon - self.path.path_len
        slots_left = self._max_horizon - len(prefix_ids)

        # Fetch viewpoint that are: VALID, NOT VISITED and HAS GAIN
        # vpts = [(vptid, vp) for vptid, vp in self.vpman.viewpoints.items() 
        #         if vp and vp.valid and not getattr(vp, "visited", False) and not getattr(vp, "no_gain", False)]
        
        vpts = [(vptid, vp) for vptid, vp in self.vpman.viewpoints.items() 
                if vp and vp.valid and not getattr(vp, "visited", False)]
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

        # Get starting index
        if len(prefix_ids) == 0:
            self._start = self._choose_start_idx(pos, self.drone_pos)
            if self._start < 0:
                return # could not find starting index
            init = True
        else:
            anchor_vptid = prefix_ids[-1] # start from anchor
            self._start = vptid2idx.get(int(anchor_vptid), -1)
            # last_vptid = self.path.viewpoints[-1] # increment from last in path
            # self._start = vptid2idx.get(int(last_vptid), -1)
            if self._start < 0:
                self._start = self._choose_start_idx(pos, self.drone_pos)
                if self._start < 0:
                    return
                init = True
            else:
                init = False

        used = np.zeros(pos.shape[0], dtype=bool) # used this search and current path
        for pid in prefix_ids:
            j = vptid2idx.get(int(pid))
            if j is not None:
                used[j] = True
        used[self._start] = True
        chosen = [self._start] # what will be appended to the path

        preferred_idx = None
        if old_suffix_first is not None and int(old_suffix_first) in vptid2idx:
            preferred_idx = int(vptid2idx[int(old_suffix_first)])

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
            
            v_last = self._last_segment_dir(pos, chosen)
            j_best, best_cost = self._pick_with_cost(i, same_vert, skel_adjacent, same_branch, others, v_last, pos, scores)

            # Prefer old path after anchor unless clearly worse (or not existing anymore)
            if preferred_idx is not None and used[preferred_idx] is False:
                if self._has_clear_path(pos[i], pos[preferred_idx], 2.0*float(self.occ_map.voxel_size)):
                    pref_cost = self._single_cost(i, preferred_idx, pos, scores, v_last)
                    if not (best_cost + 1e-9) < (1.0 - self._prefer_old_margin) * pref_cost:
                        j_best = preferred_idx
                        best_cost = pref_cost
            preferred_idx = None # only once...

            if j_best < 0:
                break # could not find suitable viewpoint 

            used[j_best] = True
            chosen.append(j_best)

        chosen_ids = np.array([vpts[k][0] for k in chosen], dtype=np.int64) # index to viewpoint id
        if not init and chosen_ids.size > 0:
            chosen_ids = chosen_ids[1:] # remove duplicate if not chosen from empty path

        candidate_ids = prefix_ids + [int(i) for i in chosen_ids]
        
        if self._replan_cooldown > 0 and old_suffix_first is not None and len(chosen_ids) > 0:
            return False
        
        if self._commit_if_changed(candidate_ids):
            self._replan_cooldown = self._replan_cooldown_ticks
            self._refine_path()
            committed = True

        return committed

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

    def _pick_with_cost(self, i, same_vert, skel_adjacent, same_branch, others, v_last, pos, scores):
        p_i = pos[i]
        clearance = 2.0 * float(self.occ_map.voxel_size)

        # viewpoint class and corresponding penalty
        buckets = [
            (same_vert, self.c_same_vert),
            (skel_adjacent, self.c_adj_vert),
            (same_branch, self.c_same_branch),
            (others, self.c_others)
        ]

        best_j = -1
        best_cost = np.inf
        for cand, penalty in buckets:
            if cand.size == 0:
                continue
            
            d = np.linalg.norm(pos[cand] - p_i[None, :], axis=1) # distances from vertex i to candidate
            s = scores[cand]

            if v_last is None:
                turn = np.zeros_like(d, dtype=np.float32)
            else:
                turn = np.array([self._turn_penalty(v_last, p_i, pos[int(j)]) for j in cand], dtype=np.float32)

            cost = self.w_dist*d + self.w_pen*penalty + self.w_turn*turn - self.w_score*s # cost 
            order = np.argsort(cost) # indices sorted by smallest to largest cost
            for k in order:
                j = int(cand[k])
                if self._has_clear_path(p_i, pos[j], clearance):
                    c = float(cost[k])
                    if c < best_cost:
                        best_cost = c
                        best_j = j
                    break
        return best_j, best_cost
    
    def _single_cost(self, i: int, j: int, pos: np.ndarray, scores: np.ndarray, v_last) -> float:
        p_i = pos[i]
        d = float(np.linalg.norm(pos[j] - p_i))
        s = float(scores[j])
        t = 0.0 if v_last is None else float(self._turn_penalty(v_last, p_i, pos[j]))
        penalty = self.c_others
        return self.w_dist*d + self.w_pen*penalty + self.w_turn*t - self.w_score*s

    def _refine_path(self, max_iters:int = 2) -> None:
        # Refinement: Make the current path order minimize the total distance of the path (linear segments)
        if self.path is None or self.path.path_len < 3:
            return
        
        max_tail = self._max_horizon
        m = min(int(max_tail), self.path.path_len)
        
        tail_start = self.path.path_len - m
        tail_end = self.path.path_len
        ids = self.path.viewpoints[tail_start:tail_end]

        try:
            pos = np.vstack([self.vpman.viewpoints.get(vptid).pos for vptid in ids]).astype(np.float32)
        except Exception:
            return
        
        clearance = 2.0 * float(self.occ_map.voxel_size)
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            n = len(ids)

            for i in range(n - 2):
                for j in range(i + 2, n):
                    pi = pos[i]
                    pi1 = pos[i + 1]
                    pjm1 = pos[j - 1]
                    pj = pos[j]

                    d_old = np.linalg.norm(pi - pi1) + np.linalg.norm(pjm1 - pj)
                    d_new = np.linalg.norm(pi - pjm1) + np.linalg.norm(pi1 - pj)

                    if d_new + 1e-6 < d_old:
                        if self._has_clear_path(pi, pjm1, clearance) and self._has_clear_path(pi1, pj, clearance):
                            ids[i + 1:j] = list(reversed(ids[i + 1:j]))
                            pos[i + 1:j] = pos[i + 1:j][::-1]
                            improved =  True

        self.path.viewpoints[tail_start:tail_end] = ids

    def _has_clear_path(self, p0, p1, clearance: float) -> bool:
        if self.occ_map is None:
            return False
        try:
            return self.occ_map.line_of_sight(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), clearance=float(clearance))
        except Exception:
            return False
        

