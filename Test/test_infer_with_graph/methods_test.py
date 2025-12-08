# methods.py
import math
import numpy as np
from ATF.methods.decorators import method_profile


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=1.0,
    vectorized=True
)
def add(a):
    """a + 1 (vectorized)"""
    # a is (N,) or (N, 1)
    a = np.asarray(a)
    result = a + 1  # shape: (N,) or (N, 1)
    # 确保输出为 (N,) —— 更通用
    return np.squeeze(result)  # (N, 1) → (N,); (N,) unchanged


@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=1.0,
    vectorized=True
)
def multiply(a, b):
    """a * b (vectorized)"""
    a, b = np.asarray(a), np.asarray(b)
    result = a * b
    return np.squeeze(result)


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=0.8,
    vectorized=True
)
def square(a):
    """a ** 2 (vectorized)"""
    a = np.asarray(a)
    result = a ** 2
    return np.squeeze(result)


@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar', 'scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=0.7,
    vectorized=True
)
def sum3(a, b, c):
    """a + b + c (vectorized)"""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    result = a + b + c
    return np.squeeze(result)


@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group='identity',
    weight=0.5,
    vectorized=True  # ✅ 现在支持向量化
)
def identity_multi_output(a):
    """
    返回两个值：原值和其两倍 (vectorized)
    输出 shape: (N, 2)
    """
    a = np.asarray(a)  # shape (N,) or (N, 1)
    a = np.squeeze(a)  # ensure (N,)
    # 构造 (N, 2)
    result = np.stack([a, a * 2], axis=-1)  # shape (N, 2)
    return result