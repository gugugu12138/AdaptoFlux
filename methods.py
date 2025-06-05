import math
import random
#方法集的输入与输出之比直接决定数组收敛速度
def return_value(x):
    """
    返回原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x]

def add_values(x, y):
    """
    将两个数值相加
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之和
    """
    return [x + y]

def calculate_difference(a, b):
    return [a - b]

def multiply_values(x, y):
    """
    将两个数值相乘
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之积
    """
    return [x * y]


def return_values(x):
    """
    返回三个原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x,x,x]

def ignore(x):
    """
    忽略该数值
    """
    return []

def ceil_number(number):
    return [math.ceil(number)]  # 向上取整

def floor_number(number):
    return [math.floor(number)]  # 向下取整

def round_number(number):
    return [round(number)]  # 四舍五入