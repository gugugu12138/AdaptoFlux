# experiments/embodied_bird/bird_state.py

class BirdActionState:
    def __init__(self):
        self.should_jump = False
        self.current_observation = None  # ← 新增

    def reset(self):
        self.should_jump = False

    def set_jump(self):
        self.should_jump = True

    def set_observation(self, obs):
        self.current_observation = obs

BIRD_STATE = BirdActionState()