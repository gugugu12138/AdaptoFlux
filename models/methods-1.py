import math
import random
import numpy as np

# =====================================================================
# 🟢 基础数值操作类
# =====================================================================

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
    """
    计算两个数值的差
    :param a: 第一个数
    :param b: 第二个数
    :return: 差值
    """
    return [a - b]

def multiply_values(x, y):
    """
    将两个数值相乘
    :param x: 第一个数
    :param y: 第二个数
    :return: 两数之积
    """
    return [x * y]

# =====================================================================
# 🟡 多输出/复制类
# =====================================================================

def return_values(x):
    """
    返回三个原值
    :param x: 输入数值
    :return: 原数值三次
    """
    return [x, x, x]

def ignore(x):
    """
    忽略该数值
    """
    return []

# =====================================================================
# 🔵 取整与变换类
# =====================================================================

def ceil_number(number):
    """
    向上取整
    :param number: 输入数值
    :return: 向上取整后的值
    """
    return [math.ceil(number)]

def floor_number(number):
    """
    向下取整
    :param number: 输入数值
    :return: 向下取整后的值
    """
    return [math.floor(number)]

def round_number(number):
    """
    四舍五入
    :param number: 输入数值
    :return: 四舍五入后的值
    """
    return [round(number)]

# =====================================================================
# 🟣 非线性激活函数（模拟神经元）
# =====================================================================

def sigmoid(x):
    """
    Sigmoid 激活函数
    :param x: 输入数值
    :return: 经过 sigmoid 映射后的值
    """
    return [1 / (1 + math.exp(-x))]

def relu(x):
    """
    ReLU 激活函数
    :param x: 输入数值
    :return: max(0, x)
    """
    return [x if x > 0 else 0]

def tanh(x):
    """
    Tanh 激活函数
    :param x: 输入数值
    :return: Tanh(x)
    """
    return [math.tanh(x)]

# =====================================================================
# 🟠 统计类函数（增强特征抽象能力）
# =====================================================================

def mean_value(values):
    """
    返回输入数组的均值
    :param values: 输入数组
    :return: 均值
    """
    return [np.mean(values)]

def max_value(values):
    """
    返回输入数组的最大值
    :param values: 输入数组
    :return: 最大值
    """
    return [max(values)]

def min_value(values):
    """
    返回输入数组的最小值
    :param values: 输入数组
    :return: 最小值
    """
    return [min(values)]

def std_deviation(values):
    """
    返回输入数组的标准差
    :param values: 输入数组
    :return: 标准差
    """
    return [np.std(values)]

# =====================================================================
# 🟤 分类友好型函数（关键：输出 10 维向量）
# =====================================================================

def project_to_10_dim(x):
    """
    将输入映射到 10 维空间（模拟类别得分）
    :param x: 输入数值
    :return: 长度为 10 的数组
    """
    base = [0.1] * 10
    index = int(np.clip(round(x) % 10, 0, 9))
    base[index] += abs(x) % 1
    return base

def normalize_to_softmax(values):
    """
    归一化为 softmax-like 输出（概率分布）
    :param values: 输入数组（长度应为 10）
    :return: 概率分布
    """
    exp_values = np.exp(values - np.max(values))
    return (exp_values / exp_values.sum()).tolist() 