# model_trainer.py
import random
import numpy as np
from collections import Counter
from ..GraphManager.graph_processor import GraphProcessor
import networkx as nx
from ..ModelGenerator.model_generator import ModelGenerator
from abc import ABC, abstractmethod

class ModelTrainer:
    """
    所有训练器的基类。
    负责绑定 AdaptoFlux 实例、管理模型生成器，并提供统一的损失计算接口。
    """

    # 将损失函数直接定义为类的静态方法
    @staticmethod
    def _mse_loss(pred, target):
        return np.mean((pred - target) ** 2)

    @staticmethod
    def _mae_loss(pred, target):
        return np.mean(np.abs(pred - target))

    @staticmethod
    def _rmse_loss(pred, target):
        return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def _huber_loss(pred, target, delta=1.0):
        diff = np.abs(target - pred)
        return np.mean(np.where(diff < delta,
                                0.5 * (diff ** 2),
                                delta * diff - 0.5 * (delta ** 2)))

    @staticmethod
    def _binary_crossentropy_loss(pred, target):
        # 加上极小值防止 log(0)
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    @staticmethod
    def _categorical_crossentropy_loss(pred, target):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return -np.mean(np.sum(target * np.log(pred), axis=1))

    # 定义内置损失函数的映射字典
    _LOSS_FUNCTIONS = {
        'mse': _mse_loss.__func__,
        'mae': _mae_loss.__func__,
        'rmse': _rmse_loss.__func__,
        'huber': _huber_loss.__func__,
        'binary_crossentropy': _binary_crossentropy_loss.__func__,
        'categorical_crossentropy': _categorical_crossentropy_loss.__func__
    }

    def __init__(self, adaptoflux_instance, loss_fn='mse', task_type='regression'):
        """
        初始化训练器，绑定 AdaptoFlux 实例。

        :param adaptoflux_instance: 已初始化的 AdaptoFlux 对象
        :param loss_fn: 损失函数名 (字符串) 或可调用对象
        :param task_type: 任务类型 ('regression', 'binary_classification', 'multiclass_classification')
        """
        self.adaptoflux = adaptoflux_instance
        self.model_generator = ModelGenerator(adaptoflux_instance)

        # 设置损失函数
        self._set_loss_fn(loss_fn, task_type)

    def _set_loss_fn(self, loss_fn, task_type):
        """
        根据参数设置 self.loss_fn 方法。
        """
        if callable(loss_fn):
            # 如果传入的是可调用对象，直接使用
            self.loss_fn = loss_fn
        else:
            # 否则，假设它是一个字符串
            loss_fn_name = loss_fn

            # 如果为 None，根据 task_type 选择默认值
            if loss_fn_name is None:
                default_map = {
                    'regression': 'mse',
                    'binary_classification': 'binary_crossentropy',
                    'multiclass_classification': 'categorical_crossentropy'
                }
                loss_fn_name = default_map.get(task_type, 'mse')

            # 从字典中查找
            if loss_fn_name not in self._LOSS_FUNCTIONS:
                raise ValueError(f"Unknown loss function: {loss_fn_name}. "
                               f"Available: {list(self._LOSS_FUNCTIONS.keys())}")

            # 直接从字典中获取函数
            self.loss_fn = self._LOSS_FUNCTIONS[loss_fn_name]

    @abstractmethod
    def train(self, **kwargs):
        """所有子类必须实现 train 方法"""
        pass