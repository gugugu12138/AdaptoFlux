import pandas as pd
import numpy as np
from ATF.core.adaptoflux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod, CollapseFunctionManager
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.model_trainer import ModelTrainer

# 用于加载 MNIST
from sklearn.datasets import fetch_openml
import logging
import json

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)

def collapse_sum_positive(values):
    """
    自定义坍缩方法：
    - 将一维数组所有值相加
    - 如果和大于 0 返回 1，否则返回 0
    """
    total = np.sum(values)
    return 1 if total > 392 else 0


def load_mnist_for_adaptoflux(binary_class=0, methods_path=None, collapse_method=CollapseMethod.SUM):
    """
    加载 MNIST 数据并构建 AdaptoFlux 实例，执行二分类任务。

    :param binary_class: 要分类的目标类别（例如：0 表示识别是否为 0）
    :param methods_path: 方法路径（传给 AdaptoFlux）
    :param collapse_method: 坍缩方法
    :return: AdaptoFlux 实例
    """
    # 下载 MNIST 数据
    print("正在下载 MNIST 数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    # 归一化：像素值从 [0,255] 缩放到 [0,1]
    X = X / 255.0

    # 构造二分类标签：例如，是否为 `binary_class`
    labels = (y == binary_class).astype(int)  # 是目标类则为 1，否则为 0

    # 可选：为了测试快速性，可以只取部分数据
    # X, labels = X[:10000], labels[:10000]  # 减少数据量调试用

    print(f"数据形状: {X.shape}, 标签形状: {labels.shape}")
    print(f"正样本数量: {np.sum(labels)}, 负样本数量: {len(labels) - np.sum(labels)}")

    # 创建 AdaptoFlux 实例
    adaptoflux_instance = AdaptoFlux(
        values=X,
        labels=labels,
        methods_path=methods_path,
        collapse_method=collapse_method
    )

    return adaptoflux_instance


# =============================
# 开始训练
# =============================

if __name__ == "__main__":
    # 加载 MNIST 二分类数据（比如识别是否为数字 0）
    model = load_mnist_for_adaptoflux(
        binary_class=0,                    # 修改这里来切换目标类别
        methods_path='examples/mnist/methods.py',  # 假设你的方法文件仍可用
        collapse_method=CollapseMethod.SUM
    )

    # 添加自定义坍缩函数（如果需要）
    model.add_collapse_method(collapse_sum_positive)

    # 创建训练器
    trainer = LayerGrowTrainer(
        adaptoflux_instance=model,
        max_attempts=10,
        decision_threshold=0.0,
        verbose=True
    )

    # 开始训练
    result = trainer.train(
        input_data=model.values,
        target=model.labels,
        max_layers=20,
        save_model=True,
        on_retry_exhausted="rollback",
        rollback_layers=2,
        max_total_attempts=3000,
        model_save_path = 'examples/mnist/mnist_model'  # 保存模型路径
    )

    # 保存结果
    with open('mnist_training_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("✅ MNIST 二分类训练完成，结果已保存为 mnist_training_result.json")