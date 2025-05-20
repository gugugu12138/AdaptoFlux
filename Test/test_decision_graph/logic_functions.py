# logic_functions.py
from methods.decorators import output_count

@output_count(3)
def classify_number(x):
    """
    判断 x 是正数、负数还是零，输出长度为 3 的列表。
    - 第一列为 x（如果 x < 0）
    - 第二列为 x（如果 x == 0）
    - 第三列为 x（如果 x > 0）
    否则对应项为 None
    """
    if x is None:
        return [None, None, None]
    if x < 0:
        return [x, None, None]
    elif x == 0:
        return [None, x, None]
    else:
        return [None, None, x]

@output_count(1)
def identity(x):
    """
    直接返回输入值（不进行判断）
    """
    return [x]