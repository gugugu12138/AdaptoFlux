# -*- coding: utf-8 -*-
from ATF.methods.decorators import method_profile
import numpy as np
from experiments.embodied_bird.bird_state import BIRD_STATE

# === Input Methods (核心 + 干扰) ===
# === Input Methods (核心 + 干扰) ===
@method_profile(input_types=['raw_signal'], output_types=['state_value'], group="input", weight=1.0)
def get_player_y(x):
    obs = BIRD_STATE.current_observation
    return [float(obs[9])] if obs is not None and len(obs) > 9 else [None]

@method_profile(input_types=['raw_signal'], output_types=['state_value'], group="input", weight=1.0)
def get_player_vel(x):
    obs = BIRD_STATE.current_observation
    return [float(obs[10])] if obs is not None and len(obs) > 10 else [None]

@method_profile(input_types=['raw_signal'], output_types=['state_value'], group="input", weight=1.0)  # 核心：下一个下管道
def get_next_pipe_bottom_y(x):
    obs = BIRD_STATE.current_observation
    return [float(obs[5])] if obs is not None and len(obs) > 5 else [None]

@method_profile(input_types=['raw_signal'], output_types=['state_value'], group="input", weight=0.5)  # 干扰项：上管道
def get_next_pipe_top_y(x):
    obs = BIRD_STATE.current_observation
    return [float(obs[4])] if obs is not None and len(obs) > 4 else [None]

@method_profile(input_types=['raw_signal'], output_types=['state_value'], group="input", weight=0.5)  # 干扰项：水平距离到下一个管道
def get_next_pipe_dist(x):
    obs = BIRD_STATE.current_observation
    return [float(obs[3])] if obs is not None and len(obs) > 3 else [None]

# === Function Methods ===
@method_profile(input_types=['state_value','state_value'], output_types=['computed_score'], group="function", weight=2.0)
def predict_bird_y_next(bird_y, bird_vel):
    return [bird_y + bird_vel * 1.5] if None not in (bird_y, bird_vel) else [None]

@method_profile(input_types=['state_value'], output_types=['computed_score'], group="function", weight=2.0)
def compute_safe_lower_bound(pipe_y):
    return [pipe_y - 0.05] if pipe_y is not None else [None]

@method_profile(input_types=['computed_score','computed_score'], output_types=['computed_score'], group="function", weight=3.0)
def is_below_safe_zone(pred_y, safe_bound):
    return [pred_y - safe_bound] if None not in (pred_y, safe_bound) else [None]

# === Logic & Action ===
@method_profile(input_types=['computed_score'], output_types=['logic_signal','logic_signal'], group="logic", weight=3.0)
def gate_positive_negative(x):
    if x is None: return [None, None]
    return ([x, None] if x > 0 else [None, x])

@method_profile(input_types=['logic_signal'], output_types=['logic_signal'], group="action", weight=10.0)
def jump(x):
    if x is not None and np.isfinite(x) and x != 0.0:
        BIRD_STATE.set_jump()
    return [1.0 if x is not None else 0.0]

@method_profile(input_types=['state_value', 'state_value'], output_types=['computed_score'], group="function", weight=2.0)
def compute_gap_center(pipe_top, pipe_bottom):
    if None in (pipe_top, pipe_bottom):
        return [None]
    return [(pipe_top + pipe_bottom) / 2.0]

@method_profile(input_types=['state_value', 'computed_score'], output_types=['computed_score'], group="function", weight=2.0)
def bird_above_gap(bird_y, gap_center):
    if None in (bird_y, gap_center):
        return [None]
    return [bird_y - gap_center]  # >0 表示鸟在中心上方 → 需要跳