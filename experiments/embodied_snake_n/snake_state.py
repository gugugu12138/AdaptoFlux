# experiments/embodied_snake/snake_state.py

class SnakeActionState:
    def __init__(self):
        self.decided_action = 0  # 0=forward, 1=left, 2=right
        self.current_observation = None

    def reset(self):
        self.decided_action = 0

    def set_action(self, action):
        if action in [0, 1, 2]:
            self.decided_action = action

    def set_observation(self, obs):
        self.current_observation = obs

    def set_snake_body(self, body):
        # body is list of (y, x) as in env.snake
        self.snake_body = set(body)  # 转为 set 加速查询

    def get_snake_body(self):
        return self.snake_body

SNAKE_STATE = SnakeActionState()