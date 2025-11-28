# model_trainer.py
import random
import numpy as np
from collections import Counter
from ..GraphProcessor.graph_processor import GraphProcessor
import networkx as nx
from ..ModelGenerator.model_generator import ModelGenerator
from abc import ABC, abstractmethod
import logging
import traceback
import sys


logger = logging.getLogger(__name__)


class ModelTrainer(ABC):
    """
    所有训练器的基类。
    负责绑定 AdaptoFlux 实例、管理模型生成器，并提供统一的损失与准确率评估接口。
    """

    # 静态损失函数定义
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
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    @staticmethod
    def _categorical_crossentropy_loss(pred, target):
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return -np.mean(np.sum(target * np.log(pred), axis=1))

    # 损失函数映射
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
        self.task_type = task_type  # ← 保存为实例属性，供 _evaluate_accuracy 使用
        self._set_loss_fn(loss_fn, task_type)

    def _set_loss_fn(self, loss_fn, task_type):
        """根据参数设置 self.loss_fn 方法。"""
        if callable(loss_fn):
            self.loss_fn = loss_fn
        else:
            loss_fn_name = loss_fn
            if loss_fn_name is None:
                default_map = {
                    'regression': 'mse',
                    'binary_classification': 'binary_crossentropy',
                    'multiclass_classification': 'categorical_crossentropy'
                }
                loss_fn_name = default_map.get(task_type, 'mse')

            if loss_fn_name not in self._LOSS_FUNCTIONS:
                raise ValueError(f"Unknown loss function: {loss_fn_name}. "
                               f"Available: {list(self._LOSS_FUNCTIONS.keys())}")

            self.loss_fn = self._LOSS_FUNCTIONS[loss_fn_name]

    def _evaluate_loss(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        use_pipeline: bool = False,
        num_workers: int = 4,
        adaptoflux_instance=None  # ← 新增参数
    ) -> float:
        """
        核心机制的 "Evaluate" 步骤。
        在指定图结构上执行前向传播并计算损失。

        :param input_ 用于评估的输入数据
        :param target: 对应的标签
        :param use_pipeline: 是否使用并行流水线推理
        :param num_workers: 并行线程数（仅在 use_pipeline=True 时有效）
        :param adaptoflux_instance: 可选，临时指定的 AdaptoFlux 实例；若为 None，则使用 self.adaptoflux
        :return: 损失值，失败时返回 float('inf')
        """
        try:
            af = adaptoflux_instance if adaptoflux_instance is not None else self.adaptoflux

            if use_pipeline:
                output = af.infer_with_task_parallel(values=input_data, num_workers=num_workers)
            else:
                output = af.infer_with_graph(values=input_data)

            if output.shape[0] != target.shape[0]:
                raise ValueError(f"Output batch size {output.shape[0]} != target batch size {target.shape[0]}")

            loss = self.loss_fn(output, target)
            return float(loss)

        except Exception as e:
            logger.exception("Loss evaluation failed – terminating program.")  # ← 打印完整堆栈
            sys.exit(1)  # ← 立即终止


    def _evaluate_accuracy(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        use_pipeline: bool = False,
        num_workers: int = 4,
        task_type: str = None,
        adaptoflux_instance=None  # ← 新增参数
    ) -> float:
        """
        计算指定图结构的准确率（或回归伪准确率）。

        :param input_ 输入数据
        :param target: 真实标签
        :param use_pipeline: 是否使用并行推理
        :param num_workers: 并行线程数
        :param task_type: 任务类型（None 表示使用 self.task_type）
        :param adaptoflux_instance: 可选，临时指定的 AdaptoFlux 实例；若为 None，则使用 self.adaptoflux
        :return: 准确率 (0~1)，失败时返回 0.0
        """
        # 确定任务类型
        if task_type is None:
            task_type = getattr(self, 'task_type', 'auto')

        internal_task = {
            'binary_classification': 'binary',
            'multiclass_classification': 'multiclass',
            'regression': 'regression'
        }.get(task_type, task_type)

        try:
            af = adaptoflux_instance if adaptoflux_instance is not None else self.adaptoflux

            if use_pipeline:
                output = af.infer_with_task_parallel(values=input_data, num_workers=num_workers)
            else:
                output = af.infer_with_graph(values=input_data)

            output = np.array(output)
            true_labels = np.array(target).flatten()

            if internal_task == 'regression':
                output = output.flatten()
                rel_error = np.abs(output - true_labels) / (np.abs(true_labels) + 1e-8)
                pseudo_acc = float(np.mean(rel_error < 0.01))
                return pseudo_acc

            if internal_task == "auto":
                if len(output.shape) == 1 or (output.ndim == 2 and output.shape[1] == 1):
                    internal_task = "binary"
                else:
                    internal_task = "multiclass"

            if internal_task == "binary":
                if output.ndim == 2 and output.shape[1] == 1:
                    output = output.flatten()
                elif output.ndim > 1 and output.shape[1] != 1:
                    raise ValueError("Binary task expects output shape [N,] or [N, 1].")
                pred_classes = (output >= 0.5).astype(int)
            elif internal_task == "multiclass":
                if output.ndim == 1:
                    pred_classes = output.astype(int)
                else:
                    pred_classes = np.argmax(output, axis=1)
            else:
                raise ValueError(f"Unsupported task_type: {internal_task}")

            pred_classes = pred_classes.flatten()
            accuracy = float(np.mean(pred_classes == true_labels))
            return accuracy

        except Exception as e:
            logger.exception("Accuracy evaluation failed – terminating program.")  # ← 完整堆栈
            sys.exit(1)  # ← 终止

    @abstractmethod
    def train(self, **kwargs):
        """所有子类必须实现 train 方法"""
        pass