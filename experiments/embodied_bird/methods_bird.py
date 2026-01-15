# -*- coding: utf-8 -*-
# File: experiments/embodied_bird/methods_bird.py
import math
from ATF.methods.decorators import method_profile
import numpy as np
from experiments.embodied_bird.bird_state import BIRD_STATE


# --- Function Pool (Pure, No Side Effects) ---

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def identity(x):
    """Pass-through function (including None)."""
    return [x]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def add_values(x, y):
    if x is None or y is None:
        return [None]
    return [x + y]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def calculate_difference(a, b):
    if a is None or b is None:
        return [None]
    return [a - b]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def multiply_values(x, y):
    if x is None or y is None:
        return [None]
    return [x * y]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def divide_values(x, y):
    if x is None or y is None or y == 0:
        return [None]
    return [x / y]

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def fanout(x):
    """Duplicate signal; if x is None, return [None, None]."""
    if x is None:
        return [None, None]
    return [x, x]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def relu(x):
    if x is None:
        return [None]
    return [max(0.0, x)]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def sigmoid(x):
    if x is None:
        return [None]
    x = np.clip(x, -10.0, 10.0)
    return [1.0 / (1.0 + math.exp(-x))]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def absolute(x):
    if x is None:
        return [None]
    return [abs(x)]


# --- Action Pool (Side Effects via BIRD_STATE) ---

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="action",
    weight=1.0,
    vectorized=False
)
def jump(x):
    """
    Trigger jump if x is not None and valid.
    Always returns [x] to maintain data flow.
    """
    if x is not None and np.isfinite(x) and x != 0.0:
        BIRD_STATE.set_jump()
    return [x]


# --- Logic Routing Functions ---

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="logic",
    weight=1.0,
    vectorized=False
)
def gate_positive_negative(x):
    if x is None:
        return [None, None]
    if x > 0:
        return [x, None]
    else:
        return [None, x]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="logic",
    weight=1.0,
    vectorized=False
)
def threshold_gate(x):
    if x is None:
        return [None]
    threshold = 0.5
    if x > threshold:
        return [x]
    else:
        return [None]


# --- Input Extractors (Gated by input signal) ---

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="input",
    weight=1.0,
    vectorized=False
)
def get_player_y(x):
    if x is None:
        return [None]
    obs = BIRD_STATE.current_observation
    if obs is None or len(obs) < 1:
        return [None]
    return [float(obs[0])]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="input",
    weight=1.0,
    vectorized=False
)
def get_player_vel(x):
    if x is None:
        return [None]
    obs = BIRD_STATE.current_observation
    if obs is None or len(obs) < 2:
        return [None]
    return [float(obs[1])]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="input",
    weight=1.0,
    vectorized=False
)
def get_next_pipe_dist(x):
    if x is None:
        return [None]
    obs = BIRD_STATE.current_observation
    if obs is None or len(obs) < 3:
        return [None]
    return [float(obs[2])]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="input",
    weight=1.0,
    vectorized=False
)
def get_next_pipe_top_y(x):
    if x is None:
        return [None]
    obs = BIRD_STATE.current_observation
    if obs is None or len(obs) < 4:
        return [None]
    return [float(obs[3])]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="input",
    weight=1.0,
    vectorized=False
)
def get_next_pipe_bottom_y(x):
    if x is None:
        return [None]
    obs = BIRD_STATE.current_observation
    if obs is None or len(obs) < 5:
        return [None]
    return [float(obs[4])]