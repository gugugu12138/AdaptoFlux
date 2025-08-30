# methods.py
import math
import numpy as np
from ATF.methods.decorators import method_profile  # 假设新装饰器已定义在此处


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=1.0,
    vectorized=True  # 大多数简单数学函数可向量化
)
def add(a):
    """a + 1"""
    return [a + 1]


@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=1.0,
    vectorized=True
)
def multiply(a, b):
    """a * b"""
    return [a * b]


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=0.8,
    vectorized=True
)
def square(a):
    """a ** 2"""
    return [a ** 2]


@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar', 'scalar'],
    output_types=['scalar'],
    group='arithmetic',
    weight=0.7,
    vectorized=True
)
def sum3(a, b, c):
    """a + b + c"""
    return [a + b + c]


@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group='identity',
    weight=0.5,
    vectorized=False  # 可根据实际需求设为 True，如果支持批量输入
)
def identity_multi_output(a):
    """
    返回两个值：原值和其两倍
    示例输出: [a, a * 2]
    """
    return [a, a * 2]