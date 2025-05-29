import numpy as np
from enum import Enum


class CollapseMethod(Enum):
    SUM = 1       # 求和
    AVERAGE = 2   # 平均
    VARIANCE = 3  # 方差
    PRODUCT = 4   # 相乘
    CUSTOM = 5    # 自定义方法
    AREA = 6

class CollapseFunctionManager:
    def __init__(self, method=CollapseMethod.SUM):
        
        self.collapse_method = method
        self.custom_function = None  # 自定义坍缩函数
        self.num_bins = 10  # 默认分为10段（可根据需要调整）

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
        
    def set_num_bins(self, num_bins):
        """设置波形面积分割的段数（默认为10）"""
        if num_bins > 0:
            self.num_bins = num_bins
        else:
            raise ValueError("段数必须为正整数")

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
        elif self.collapse_method == CollapseMethod.AREA:
            return self._area(values)
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
    
    def _area(self, values):
        """计算波形面积分割概率（返回概率列表或主峰索引）"""
        n = len(values)
        if n < 2 or self.num_bins == 0:
            print('使用area方法至少需要两个数据，且分段数大于0')
            return np.zeros(self.num_bins) if self.num_bins > 1 else 0.0

        # 计算绝对面积（梯形法）
        y = np.array(values)
        total_area = np.sum(np.abs(y[:-1] + y[1:]) * 0.5)

        # 分段计算面积
        segment_length = n / self.num_bins
        probabilities = []
        for k in range(self.num_bins):
            start = int(k * segment_length)
            end = min(int((k + 1) * segment_length), n - 1)
            if start >= end:
                probabilities.append(0.0)
                continue
            segment_y = y[start:end+1]
            segment_area = np.sum(np.abs(segment_y[:-1] + segment_y[1:]) * 0.5)
            probabilities.append(segment_area / total_area if total_area > 0 else 0.0)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum() if probabilities.sum() > 0 else 1.0  # 归一化
        return probabilities