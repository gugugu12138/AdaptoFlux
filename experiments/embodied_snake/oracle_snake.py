# experiments/embodied_snake/oracle_snake.py
import numpy as np
from .snake_env import SimpleSnakeEnv

# experiments/embodied_snake/oracle_snake.py

def oracle_policy(obs):
    head_x, head_y = obs[0], obs[1]
    food_x, food_y = obs[2], obs[3]
    dx, dy = obs[4], obs[5]   # dx = x-direction, dy = y-direction

    delta_x = food_x - head_x
    delta_y = food_y - head_y

    # 如果几乎对齐，直行
    if abs(delta_x) < 0.05 and abs(delta_y) < 0.05:
        return 0

    # 主要沿 x 轴对齐
    if abs(delta_x) >= abs(delta_y):
        if delta_x > 0.05:  # food to the right
            if dx <= 0:  # currently not moving right
                return 2 if dy == 0 else (2 if dx == 0 else 1)  # simplify: just turn right
            else:
                return 0
        elif delta_x < -0.05:  # food to the left
            if dx >= 0:
                return 1
            else:
                return 0
    else:
        if delta_y > 0.05:  # food below
            if dy <= 0:
                # from right (dx=1) -> down: right turn
                if dx == 1:
                    return 2
                elif dx == -1:
                    return 1
                else:
                    return 0
            else:
                return 0
        elif delta_y < -0.05:  # food above
            if dy >= 0:
                if dx == 1:
                    return 1
                elif dx == -1:
                    return 2
                else:
                    return 0
            else:
                return 0

    return 0

def test_oracle(action_interval=1, max_steps=10000):
    env = SimpleSnakeEnv(grid_size=10, max_steps=max_steps)
    obs = env.reset()
    survival_time = 0
    score = 0
    decision_steps = 0

    for step in range(max_steps):
        if step % action_interval == 0:
            action = oracle_policy(obs)
            decision_steps += 1
        else:
            action = 0

        obs, reward, done = env.step(action)
        survival_time += 1
        score = env.score

        if done:
            break

    print(f"[ORACLE] Survival: {survival_time}, Score: {score}")
    return survival_time, score