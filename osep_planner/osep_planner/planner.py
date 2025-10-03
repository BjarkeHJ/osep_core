#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass

@dataclass
class CurrPath:
    path_len: int
    viewpoints: np.ndarray # array of vptids in current path
    points: np.ndarray

class PathPlanner:
    def __init__(self, viewpoints, max_horizon=10):
        self.path: CurrPath = None

        self._max_horizon = max_horizon


    def generate_path(self):
        pass

    def path_search(self):
        pass