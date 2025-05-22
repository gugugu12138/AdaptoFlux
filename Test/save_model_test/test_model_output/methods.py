# operations.py

import math
from methods.decorators import output_count

@output_count(1)
def return_value(x):
    """
    返回原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x]

@output_count(1)
def add_values(x, y):
    """
    将两个数值相加
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之和
    """
    return [x + y]

@output_count(1)
def calculate_difference(a, b):
    return [a - b]

@output_count(1)
def multiply_values(x, y):
    """
    将两个数值相乘
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之积
    """
    return [x * y]

@output_count(2)
def return_two_values(x):
    """
    返回两个原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x,x]

@output_count(3)
def return_three_values(x):
    """
    返回三个原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x,x,x]

@output_count(4)
def return_four_values(x):
    """
    返回四个原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x,x,x,x]

@output_count(0)
def ignore(x):
    """
    忽略该数值
    """
    return []

@output_count(1)
def ceil_number(number):
    return [math.ceil(number)]  # 向上取整

@output_count(1)
def floor_number(number):
    return [math.floor(number)]  # 向下取整

@output_count(1)
def round_number(number):
    return [round(number)]  # 四舍五入