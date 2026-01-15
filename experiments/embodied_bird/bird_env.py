# experiments/embodied_bird/bird_env.py
import gymnasium as gym
import numpy as np
from .bird_state import BIRD_STATE
import flappy_bird_gymnasium 

# 在 run_bird_episode 中
def run_bird_episode(model, action_interval=5, max_steps=5000):
    env = gym.make("FlappyBird-v0", render_mode=None)
    obs, _ = env.reset()
    survival_time = 0

    try:
        for frame in range(max_steps):
            if frame % action_interval == 0:
                BIRD_STATE.reset()
                BIRD_STATE.set_observation(obs)  # ← 关键！
                _ = model.infer_with_graph(obs.reshape(1, -1))
                action = 1 if BIRD_STATE.should_jump else 0
            else:
                action = 0

            obs, _, terminated, truncated, _ = env.step(action)
            survival_time += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return survival_time