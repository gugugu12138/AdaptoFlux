import pandas as pd
import numpy as np
from ATF.core.flux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod, CollapseFunctionManager

from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.model_trainer import ModelTrainer

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

    # 直接使用所有列作为特征
    values = df.values  # 二维特征矩阵

    # 转换为 numpy 浮点类型
    values = np.array(values, dtype=np.float64)

    # 创建 AdaptoFlux 实例（不传 labels）
    adaptoflux_instance = AdaptoFlux(
        values=values,
        methods_path=methods_path,
        collapse_method=collapse_method
    )

    return adaptoflux_instance

model = load_titanic_for_adaptoflux(train_processed_path='examples/kaggle/titanic/output/test_processed.csv',
                                    methods_path='examples/kaggle/titanic/methods.py')

model.add_collapse_method(collapse_sum_positive)

model.load_model(folder='models/final')

import pandas as pd

# 假设 pred 是你的预测结果，shape 为 (n,)
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

import pandas as pd

# 1. 读取你的预测结果
pred_df = pd.read_csv('examples/kaggle/titanic/submission.csv')  # 或你保存的文件名

# 2. 读取真实标签（正确标注）
true_df = pd.read_csv('examples/kaggle/titanic/input/gender_submission.csv')  # 替换为你的真实文件名

# 3. 按 PassengerId 对齐数据（确保顺序一致）
pred_df = pred_df.sort_values('PassengerId').reset_index(drop=True)
true_df = true_df.sort_values('PassengerId').reset_index(drop=True)

# 4. 检查 PassengerId 是否完全一致
if not (pred_df['PassengerId'].equals(true_df['PassengerId'])):
    raise ValueError("PassengerId 不匹配，请确保两个文件的乘客 ID 一致")

# 5. 计算准确率
accuracy = (pred_df['Survived'] == true_df['Survived']).mean()

# 6. 输出结果
print(f"✅ 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")