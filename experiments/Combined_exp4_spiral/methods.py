# -*- coding: utf-8 -*-
# File: experiments/Combined_exp4/methods.py
import math
from ATF.methods.decorators import method_profile
import numpy as np

# --- 原有基础方法 (保持不变) ---
@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def return_value(x):
    """
    Returns the input value.
    """
    return [x]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def add_values(x, y):
    """
    Adds two values.
    """
    return [x + y]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def calculate_difference(a, b):
    """
    Calculates the difference between two values.
    """
    return [a - b]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def multiply_values(x, y):
    """
    Multiplies two values.
    """
    return [x * y]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def divide_values(x, y):
    """
    Divides two values (with zero check).
    """
    return [x / y if y != 0 else 0]

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def return_two_values(x):
    """
    Returns the input value twice.
    """
    return [x, x]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def decrement(x):
    """
    Decrements the input value by 1.
    """
    return [x - 1]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def increment(x):
    """
    Increments the input value by 1.
    """
    return [x + 1]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def negate_value(x):
    """
    Negates the input value.
    """
    return [-x]

# --- 新增：非线性函数 ---
@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def square(x):
    """
    Squares the input value.
    Handles overflow, NaN, and infinity safely by returning 0.0 for non-finite inputs,
    and explicitly casts to float64 to avoid integer overflow.
    """
    # Explicitly convert to float64 to prevent integer overflow (e.g., from int64)
    x = float(x)  # This ensures Python float (IEEE 754 double), avoids numpy scalar overflow

    # Optional: add debug print if needed (you often request this)
    # print(f"[DEBUG] square input x = {x}")

    if not np.isfinite(x):
        # As per your original logic: return 0.0 for invalid inputs
        return [0.0]

    result = x * x

    # Additional safety: check if result is finite (covers overflow -> inf)
    if not np.isfinite(result):
        # You may choose to return a large finite number or 0.0
        # Returning 0.0 keeps consistency with non-finite input handling
        return [0.0]

    return [result]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def relu(x):
    """
    Applies the ReLU function (max(0, x)).
    """
    return [max(0, x)]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def absolute(x):
    """
    Returns the absolute value of the input.
    """
    return [abs(x)]

# --- 新增：聚合函数 (注意：需要固定数量的输入) ---
@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def mean_three(a, b, c):
    """
    Calculates the mean of three values.
    """
    return [(a + b + c) / 3.0]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def max_two(a, b):
    """
    Returns the maximum of two values.
    """
    return [max(a, b)]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",
    weight=1.0,
    vectorized=False
)
def min_two(a, b):
    """
    Returns the minimum of two values.
    """
    return [min(a, b)]

# --- 可选：更多三角函数 (如果复现双螺旋建议根据需要启用) ---
@method_profile(output_count=1, input_types=['scalar'], output_types=['scalar'], group="basic", weight=1.0)
def sine(x):
    return [math.sin(x) if np.isfinite(x) else 0.0]

@method_profile(output_count=1, input_types=['scalar'], output_types=['scalar'], group="basic", weight=1.0)
def cosine(x):
    return [math.cos(x) if np.isfinite(x) else 0.0]

@method_profile(output_count=1, input_types=['scalar', 'scalar'], output_types=['scalar'], group="basic", weight=1.0)
def atan2(y, x):
    return [math.atan2(y, x) if np.isfinite(x) and np.isfinite(y) else 0.0]

@method_profile(output_count=1, input_types=['scalar', 'scalar'], output_types=['scalar'], group="basic", weight=1.0)
def scale(x, factor):
    """Scales x by a factor (x * factor)."""
    return [x * factor if np.isfinite(x) and np.isfinite(factor) else 0.0]

@method_profile(output_count=1, input_types=['scalar'], output_types=['scalar'], group="basic", weight=1.0)
def sqrt_safe(x):
    return [math.sqrt(x) if x >= 0 and np.isfinite(x) else 0.0]
