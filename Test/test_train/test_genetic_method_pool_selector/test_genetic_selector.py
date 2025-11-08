# test_genetic_selector.py

import numpy as np
import sys
import os

# 模拟最小 AdaptoFlux 实例（仅用于测试）
class MockAdaptoFlux:
    def __init__(self):
        # 模拟方法池 methods = {name: metadata}
        self.methods = {
            "add": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "sub": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "mul": {"input_types": ["scalar", "scalar"], "output_types": ["scalar"], "output_count": 1},
            "div": {"input_types": ["scalar", "scalar"], "output_types": ["scalar"], "output_count": 1},
            "identity": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "square": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "sqrt": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "neg": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "inc": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
            "dec": {"input_types": ["scalar"], "output_types": ["scalar"], "output_count": 1},
        }

    def save_model(self, folder):
        pass  # 测试中不需要保存

# 模拟 LayerGrowTrainer 所需的 _evaluate_accuracy（简化版）
def mock_evaluate_accuracy(adaptoflux_instance, input_data, target):
    """
    模拟评估函数：随机返回一个与方法池大小相关的准确率。
    实际中应调用 adaptoflux_instance 的前向传播。
    """
    # 模拟：方法越多，可能准确率越高（带噪声）
    n_methods = len(adaptoflux_instance.methods)
    base_acc = min(0.5 + 0.03 * n_methods, 0.95)
    noise = np.random.normal(0, 0.05)
    acc = np.clip(base_acc + noise, 0.0, 1.0)
    return acc

# 替换 LayerGrowTrainer 的 _evaluate_accuracy（仅用于测试）
from unittest.mock import patch

# 临时替换 LayerGrowTrainer 的评估函数
def patched_train(self, input_data, target, **kwargs):
    """模拟轻量训练，返回 best_model_accuracy"""
    best_acc = mock_evaluate_accuracy(self.adaptoflux, input_data, target)
    return {
        "best_model_accuracy": best_acc,
        "final_model_accuracy": best_acc,
        "layers_added": kwargs.get("max_layers", 2)
    }

# -------------------------
# 主测试逻辑
# -------------------------

if __name__ == "__main__":
    # 添加当前路径以便导入（假设 GeneticMethodPoolSelector 在同一目录）
    sys.path.append(os.path.dirname(__file__))

    # 1. 创建模拟 AdaptoFlux 实例
    af = MockAdaptoFlux()
    print(f"原始方法池大小: {len(af.methods)} 方法")
    print("方法列表:", list(af.methods.keys()))

    # 2. 创建合成数据（回归任务：y = 2x + 1）
    np.random.seed(42)
    X = np.random.randn(1000, 1).astype(np.float32)
    y = (2 * X[:, 0] + 1).astype(np.float32)  # 目标

    # 3. 导入并 patch LayerGrowTrainer.train（跳过真实训练）
    try:
        from ATF.ModelTrainer.CombinedTrainer.GeneticMethodPoolSelector.genetic_method_pool_selector import GeneticMethodPoolSelector
    except ImportError:
        print("请确保 genetic_method_pool_selector.py 与本文件在同一目录！")
        exit(1)

    # 使用 mock 替换 train 方法（避免依赖真实图执行）
    with patch('ATF.ModelTrainer.CombinedTrainer.GeneticMethodPoolSelector.genetic_method_pool_selector.LayerGrowTrainer.train', patched_train):
        # 4. 创建选择器
        selector = GeneticMethodPoolSelector(
            base_adaptoflux=af,
            input_data=X,
            target=y,
            population_size=8,
            generations=5,
            subpool_size=4,
            layer_grow_layers=2,
            layer_grow_attempts=2,
            data_fraction=0.2,
            elite_ratio=0.25,
            mutation_rate=0.2,
            verbose=True
        )

        # 5. 执行选择
        print("\n=== 开始遗传筛选 ===")
        result = selector.select()

        # 6. 输出结果
        print("\n=== 遗传筛选结果 ===")
        print(f"最佳子方法池 (大小={len(result['best_subpool'])}): {result['best_subpool']}")
        print(f"最佳适应度（模拟准确率）: {result['best_fitness']:.4f}")
        print(f"适应度历史: {[f'{f:.3f}' for f in result['fitness_history']]}")

        # 7. 验证：子池是否为原池子集
        assert set(result["best_subpool"]) <= set(af.methods.keys()), "子池包含非法方法！"
        print("\n✅ 测试通过：子方法池有效！")