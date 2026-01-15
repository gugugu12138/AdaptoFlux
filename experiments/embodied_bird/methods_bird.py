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
    """Pass-through function."""
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
    return [x / y if y != 0 else 0.0]

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="function",
    weight=1.0,
    vectorized=False
)
def fanout(x):
    """Duplicate signal for graph connectivity."""
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
    return [abs(x)]


# --- Action Pool (Side Effects via BIRD_STATE) ---

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="action",          # ← 明确分组为 action
    weight=1.0,
    vectorized=False
)
def jump(x):
    """
    Action Pool member: triggers a jump in Flappy Bird.
    - If input x is not None and not zero-like, activate jump.
    - Uses global BIRD_STATE to communicate with environment.
    - Returns [x] to maintain data flow continuity.
    """
    # 触发条件：x 不为 None 且为有效数值（可扩展为 x > threshold）
    if x is not None and np.isfinite(x) and x != 0.0:
        BIRD_STATE.set_jump()
    # 也可改为：if x is not None: （更宽松）
    return [x]  # or [0.0] —— 保持图连通即可


# --- Optional: Logic Routing Functions (Inspired by your example) ---

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="logic",
    weight=1.0,
    vectorized=False
)
def gate_positive_negative(x):
    """
    Routes x to two outputs:
    - First: x if x > 0, else None
    - Second: x if x <= 0, else None
    """
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
    """
    Returns x if x > threshold, else None.
    Can be used upstream of jump to control activation.
    """
    threshold = 0.5
    if x is not None and x > threshold:
        return [x]
    else:
        return [None]