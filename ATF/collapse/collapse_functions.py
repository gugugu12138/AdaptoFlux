import numpy as np
from enum import Enum


class CollapseMethod(Enum):
    SUM = 1       # 求和
    AVERAGE = 2   # 平均
    VARIANCE = 3  # 方差
    PRODUCT = 4   # 相乘
    CUSTOM = 5    # 自定义方法

class CollapseFunctionManager:
    def __init__(self, method=CollapseMethod.SUM):
        self.collapse_method = method
        self.custom_function = None  # 自定义坍缩函数

    def set_custom_collapse(self, func):
        """
        设置自定义坍缩函数。

        :param func: 可调用函数，接受一维数组并返回标量值
        """
        if callable(func):
            self.custom_function = func
            self.collapse_method = CollapseMethod.CUSTOM
        else:
            raise ValueError("提供的坍缩方法必须是一个可调用函数")

    def collapse(self, values):
        """
        对单个样本数据进行聚合计算（此方法供 np.apply_along_axis 调用）。

        :param values: 输入数据，形状为 (特征维度,) 的一维数组（表示一个样本）
        :return: 聚合后的标量值
        :raises ValueError: 如果方法未设置或不可用
        """
        if self.collapse_method == CollapseMethod.SUM:
            return self._sum(values)
        elif self.collapse_method == CollapseMethod.AVERAGE:
            return self._average(values)
        elif self.collapse_method == CollapseMethod.VARIANCE:
            return self._variance(values)
        elif self.collapse_method == CollapseMethod.PRODUCT:
            return self._product(values)
        elif self.collapse_method == CollapseMethod.CUSTOM and self.custom_function:
            return self._custom(values)
        else:
            raise ValueError("未知或未设置的坍缩方法")

    def _sum(self, values):
        return np.sum(values)

    def _average(self, values):
        return np.mean(values)

    def _variance(self, values):
        return np.var(values)

    def _product(self, values):
        return np.prod(values)

    def _custom(self, values):
        if not callable(self.custom_function):
            raise ValueError("自定义坍缩函数未设置或不是一个可调用函数")
        return self.custom_function(values)