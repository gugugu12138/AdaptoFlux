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

#实验证明这个函数不适合用于这个操作建议不要使用#但其实即使这个函数导致路径出现指数爆炸也能训练回来一个合适的值#但还是不建议用
# def power(x, n):
#     """
#     返回第一个数的第二个数次方。如果指数为0，则返回 x 和 n。
#     :param x: 底数
#     :param n: 指数
#     :return: 底数的指数次方，如果 n 为 0，返回 x 和 n
#     """
#     try:
#         return [x ** n]  # 返回底数的指数次方
#     except:
#         return [random.choice([x, n])]


def return_values(x):
    """
    返回两个原数值
    :param x: 输入数值
    :return: 原数值
    """
    return [x,x]

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
