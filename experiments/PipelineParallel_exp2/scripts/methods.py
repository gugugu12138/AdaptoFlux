import math
from ATF.methods.decorators import method_profile
import time


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",           # ✅ 统一分组
    weight=1.0,
    vectorized=False
)
def return_value(x):
    """
    返回原数值
    :param x: 输入数值
    :return: 原数值
    """
    time.sleep(0.001)
    return [x]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",           # ✅ 统一分组
    weight=1.0,
    vectorized=False
)
def add_values(x, y):
    """
    将两个数值相加
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之和
    """
    time.sleep(0.001)
    return [x + y]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",           # ✅ 统一分组
    weight=1.0,
    vectorized=False
)
def calculate_difference(a, b):
    """
    计算两个数的差
    :param a: 被减数
    :param b: 减数
    :return: 差值
    """
    time.sleep(0.001)
    return [a - b]


@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group="basic",           # ✅ 统一分组
    weight=1.0,
    vectorized=False
)
def multiply_values(x, y):
    """
    将两个数值相乘
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之积
    """
    time.sleep(0.001)
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
    将两个数值相除（带零检查）
    :param x: 被除数
    :param y: 除数
    :return: x / y 或 0（当y=0时）
    """
    time.sleep(0.001)
    return [x / y if y != 0 else 0]

@method_profile(
    output_count=2,
    input_types=['scalar'],
    output_types=['scalar', 'scalar'],
    group="basic",           # ✅ 统一分组
    weight=1.0,
    vectorized=False
)
def return_two_values(x):
    """
    返回两个原数值
    :param x: 输入数值
    :return: 原数值
    """
    time.sleep(0.001)
    return [x, x]

# 没提升没降低，只会拉低速度和计算量
# @method_profile(
#     output_count=3,
#     input_types=['scalar'],
#     output_types=['scalar', 'scalar', 'scalar'],
#     group="basic",           # ✅ 统一分组
#     weight=1.0,
#     vectorized=False
# )
# def return_three_values(x):
#     """
#     返回两个原数值
#     :param x: 输入数值
#     :return: 原数值
#     """
#     return [x, x, x]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",           # 统一归入 basic 组
    weight=1.0,
    vectorized=False
)
def decrement(x):
    """
    将输入值减 1
    :param x: 输入数值
    :return: x - 1
    """
    time.sleep(0.001)
    return [x - 1]


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",           # 统一归入 basic 组
    weight=1.0,
    vectorized=False
)
def increment(x):
    """
    将输入值加 1
    :param x: 输入数值
    :return: x + 1
    """
    time.sleep(0.001)
    return [x + 1]


@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group="basic",           # 统一归入 basic 组
    weight=1.0,
    vectorized=False
)
def negate_value(x):
    """
    将输入值取相反数
    :param x: 输入数值
    :return: -x
    """
    time.sleep(0.001)
    return [-x]