import random
from copy import deepcopy
from ATF import AdaptoFlux, PathGenerator
import numpy as np

def test_replace_functions():
    # 创建模拟数据
    values = np.random.rand(100, 5)     # 100 个样本，每个样本有 5 个特征
    labels = np.random.randint(0, 2, 100)  # 二分类标签

    processor = AdaptoFlux(values,labels)

    processor.methods = {
            "method_A": {"input_count": 2, "output_count": 1},
            "method_B": {"input_count": 3, "output_count": 1}
        }

    # Step 1: 生成初始数据
    process_result = processor.process_random_method(shuffle_indices=True)

    print("原始 process_result:")
    print(process_result)

    # Step 2: 调用第一个函数（适配新结构）
    new_result_1 = processor.replace_random_elements(process_result, n=4, shuffle_indices=True)
    print("\n【第一个函数 replace_random_elements】输出:")
    print(new_result_1)

if __name__ == "__main__":
    test_replace_functions()
