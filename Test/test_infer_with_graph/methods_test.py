# methods.py
import math
import numpy as np
from ATF.methods.decorators import output_count

@output_count(1)
def add(a):
    return [a + 1]

@output_count(1)
def multiply(a, b):
    return [a * b]

@output_count(1)
def square(a):
    return [a ** 2]

@output_count(1)
def sum3(a, b, c):
    return [a + b + c]

@output_count(2)
# 如果需要输出多个值，可以返回列表或 NumPy 数组
def identity_multi_output(a):
    return [a, a * 2]

