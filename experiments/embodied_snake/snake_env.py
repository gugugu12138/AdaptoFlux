# experiments/embodied_snake/snake_env.py
import numpy as np
from .snake_state import SNAKE_STATE

# experiments/embodied_snake/snake_env.py

class SimpleSnakeEnv:
    def __init__(self, grid_size=10, max_steps=10000):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]  # (y, x)
        self.direction = (0, 1)  # (dy, dx): (0,1)=right, (1,0)=down, (0,-1)=left, (-1,0)=up
        self.food = self._place_food()
        self.steps = 0
        self.score = 0
        self.done = False
        return self._get_observation()

    def _place_food(self):
        while True:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos

    def _get_observation(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dy, dx = self.direction

        return np.array([
            head_x / self.grid_size,   # normalized x
            head_y / self.grid_size,   # normalized y
            food_x / self.grid_size,   # food x
            food_y / self.grid_size,   # food y
            dx,                        # x-direction component
            dy                         # y-direction component
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True

        # Update direction based on action
        dy, dx = self.direction
        if action == 1:  # turn left: (dy, dx) → (-dx, dy)
            self.direction = (dx, -dy)
        elif action == 2:  # turn right: (dy, dx) → (dx, -dy)
            self.direction = (-dx, dy)
        # action == 0: go straight (no change)

        dy, dx = self.direction
        head_x, head_y = self.snake[0]

        # Compute new head — NO WRAP-AROUND (hard boundaries)
        new_y = head_y + dy
        new_x = head_x + dx
        new_head = (new_x, new_y)

        print("Step:", self.steps, "Action:", action, "New head:", new_head)

        reward = 0.0

        # === Collision checks ===
        # 1. Wall collision
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            self.done = True
            reward = -1.0
        # 2. Self-collision
        elif new_head in self.snake:
            self.done = True
            reward = -1.0
        # 3. Food
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            self.score += 1
            reward = 1.0
        else:
            # Move forward
            self.snake.insert(0, new_head)
            self.snake.pop()

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        return self._get_observation(), reward, self.done

def run_snake_episode(model, action_interval=1, max_steps=10000):
    """
    与 bird_env 接口对齐
    """
    env = SimpleSnakeEnv(grid_size=10, max_steps=max_steps)
    obs = env.reset()
    survival_time = 0
    total_reward = 0.0
    decision_steps = 0
    action_count = 0

    try:
        for step in range(max_steps):
            if step % action_interval == 0:
                SNAKE_STATE.reset()
                SNAKE_STATE.set_snake_body(env.snake)
                SNAKE_STATE.set_observation(obs)
                _ = model.infer_with_graph(obs.reshape(1, -1))  # 触发方法池执行
                action = SNAKE_STATE.decided_action  # 应为 0/1/2
                decision_steps += 1
                action_count += 1
            else:
                action = 0  # 默认直行（或保持上一动作？根据需求）

            obs, reward, done = env.step(action)
            survival_time += 1
            total_reward += reward

            if done:
                break
    finally:
        pass  # 无资源需释放

    avg_reward = total_reward / max(decision_steps, 1)
    action_freq = action_count / max(decision_steps, 1)  # ≈1.0
    return survival_time, env.score, avg_reward  # 可按需调整返回值