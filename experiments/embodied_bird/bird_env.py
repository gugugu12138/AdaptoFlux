# experiments/embodied_bird/bird_env.py
import gymnasium as gym
import numpy as np
from .bird_state import BIRD_STATE
import flappy_bird_gymnasium 

# 在 run_bird_episode 中
# experiments/embodied_bird/bird_env.py

def run_bird_episode(model, action_interval=5, max_steps=5000):
    env = gym.make("FlappyBird-v0", render_mode=None)
    obs, _ = env.reset()
    survival_time = 0
    total_deviation = 0.0
    step_count = 0
    jump_count = 0      # ← 新增：记录跳了多少次
    decision_steps = 0  # ← 新增：记录做了多少次决策

    try:
        for frame in range(max_steps):
            if frame % action_interval == 0:
                BIRD_STATE.reset()
                BIRD_STATE.set_observation(obs)
                _ = model.infer_with_graph(obs.reshape(1, -1))
                action = 1 if BIRD_STATE.should_jump else 0
                decision_steps += 1
                if action == 1:
                    jump_count += 1
            else:
                action = 0

            obs, _, terminated, truncated, _ = env.step(action)
            survival_time += 1

            if len(obs) >= 5:
                bird_y = obs[0]
                pipe_top = obs[3]
                pipe_bottom = obs[4]
                gap_center = (pipe_top + pipe_bottom) / 2.0
                deviation = abs(bird_y - gap_center)
                total_deviation += deviation
                step_count += 1

            if terminated or truncated:
                break
    finally:
        env.close()

    avg_deviation = total_deviation / max(step_count, 1)
    jump_rate = jump_count / max(decision_steps, 1)  # 跳跃频率 ∈ [0, 1]
    return survival_time, avg_deviation, jump_rate  # ← 返回三个值！