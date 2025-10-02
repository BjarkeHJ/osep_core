#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass

@dataclass
class Path:
    path_len: int
    points: np.ndarray

class PathPlanner:
    def __init__(self, max_horizon=10):
        self.path: Path = None

        self._max_horizon = max_horizon


    def generate_path(self):
        pass

    def path_search(self):
        pass