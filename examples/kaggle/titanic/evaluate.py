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


def _evaluate_accuracy(output: np.ndarray, target: np.ndarray) -> float:
    """
    计算当前图结构的分类准确率。
    
    :param input_data: 输入数据
    :param target: 真实标签 (shape: [N,] 或 [N, 1])
    :return: 准确率 (0~1)
    """
    try:
        # 假设是分类任务
        if len(output.shape) == 1 or output.shape[1] == 1:
            # 二分类，输出是单值
            pred_classes = (output >= 0.5).astype(int).flatten()
        else:
            # 多分类，取最大值索引
            pred_classes = np.argmax(output, axis=1)

        true_labels = np.array(target).flatten()
        accuracy = np.mean(pred_classes == true_labels)
        return accuracy

    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}")
        traceback.print_exc()  # 👈 打印完整错误堆栈
        return 0.0  # 失败时返回 0

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
                                    methods_path='examples/kaggle/titanic/methods.py')

model.add_collapse_method(collapse_sum_positive)

model.load_model(folder='models/best')

pred = model.infer_with_graph(model.values)

# 生成对应的 PassengerId，从 892 开始
passenger_ids = range(892, 892 + len(pred))

# 构建 DataFrame
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': pred
})

# 保存为 submission.csv
submission.to_csv('examples/kaggle/titanic/submission.csv', index=False)

print("✅ 提交文件已生成：submission.csv")
