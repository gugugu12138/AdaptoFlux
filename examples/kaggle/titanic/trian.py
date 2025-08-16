import pandas as pd
import numpy as np
from ATF.core.flux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod, CollapseFunctionManager

from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.model_trainer import ModelTrainer

import logging

logging.basicConfig(
    level=logging.INFO,  # 显示 INFO 及以上级别日志
    format='[%(levelname)s] %(name)s: %(message)s'
)


def collapse_sum_positive(values):
    """
    自定义坍缩方法：
    - 将一维数组所有值相加
    - 如果和大于 0 返回 1，否则返回 0
    """
    total = np.sum(values)
    return 1 if total > 0 else 0


def load_titanic_for_adaptoflux(train_processed_path, methods_path=None, collapse_method=CollapseMethod.SUM):
    """
    从预处理后的 Titanic 训练集 CSV 加载数据，并转换为 AdaptoFlux 可用的格式。

    :param train_processed_path: 预处理后的 train_processed.csv 文件路径
    :param methods_path: 方法路径（传给 AdaptoFlux）
    :param collapse_method: 坍缩方法（传给 AdaptoFlux）
    :return: AdaptoFlux 实例
    """
    # 读取 CSV
    df = pd.read_csv(train_processed_path)

    # 确保存在 Survived 列
    if 'Survived' not in df.columns:
        raise ValueError("train_processed.csv 必须包含 'Survived' 列作为标签")

    # 分离标签和特征
    labels = df['Survived'].values  # 一维标签
    values = df.drop(columns=['Survived']).values  # 二维特征矩阵

    # 转换为 numpy 浮点类型（防止 int64/float64 混合类型问题）
    values = np.array(values, dtype=np.float64)

    # 创建 AdaptoFlux 实例
    adaptoflux_instance = AdaptoFlux(
        values=values,
        labels=labels,
        methods_path=methods_path,
        collapse_method=collapse_method
    )

    return adaptoflux_instance


model = load_titanic_for_adaptoflux(train_processed_path='examples/kaggle/titanic/output/test_processed.csv',
                                    methods_path='examples/kaggle/titanic/methods.py',
                                    collapse_method=CollapseMethod.SUM)

model.add_collapse_method(collapse_sum_positive)

trainer = LayerGrowTrainer(
            adaptoflux_instance=model,
            max_attempts=10,
            decision_threshold=0.0,
            verbose=True
        )

trainer.train(
    input_data=model.values,
    target=model.labels,
    max_layers=20,  # 设置最大层数
    save_model=True,  # 保存模型
    on_retry_exhausted = "rollback",
    rollback_layers = 2,
    max_total_attempts = 3000
)







