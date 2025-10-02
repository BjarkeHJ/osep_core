#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass

@dataclass
class Viewpoint:
    vptid: int #unique viewpoint id
    pos: np.ndarray
    yaw: float
    vid: int #parent vertex id
    visited: bool = False

class ViewpointManager:
    def __init__(self, skel_state):
        self.skeleton = skel_state # reference to SkeletonState
        self.viewpoints: dict[int, Viewpoint] = {}

        self._last_skel_version = -1 # for synchronization

    def get_viewpoints(self):
        return list(self.viewpoints.values())

    def mark_visited(self, vptid: id):
        self.viewpoints[vptid].visited = True

    def update_viewpoints(self):
        cur_ver = self.skeleton.current_version()
        if cur_ver == self._last_skel_version:
            return # nothing changed

        changed_vers = self.skeleton.get_vertices_updated_since(self._last_skel_version)
        print(f"Number of updated vertices: {len(changed_vers)}")

        for v in changed_vers:
            # generate viewpoints for updated vertices
            continue
        
        self._last_skel_version = cur_ver # update stored version 

    def _make_viewpoint(vertex):
        pass
    
    def _score_viewpoint(viewpoint):
        pass
