# experiments/embodied_bird/bird_state.py
class BirdActionState:
    def __init__(self):
        self.should_jump = False

    def reset(self):
        self.should_jump = False

    def set_jump(self):
        self.should_jump = True

BIRD_STATE = BirdActionState()