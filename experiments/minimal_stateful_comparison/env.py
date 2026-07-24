import random
import numpy as np

class MinimalState:
    def __init__(self):
        self.pos = 0
        self.target = 0

    def reset(self):
        self.pos = random.choice([-2, -1, 0, 1, 2])
        self.target = random.choice([-2, -1, 0, 1, 2])
        while self.pos == self.target:
            self.target = random.choice([-2, -1, 0, 1, 2])
        return np.array([[float(self.pos), float(self.target)]], dtype=np.float32)

    def get_obs(self):
        return np.array([[float(self.pos), float(self.target)]], dtype=np.float32)

    def is_success(self):
        return int(round(self.pos)) == int(round(self.target))

STATE = MinimalState()