# test_combined_trainer.py
import sys
import os
import numpy as np

from ATF.core.adaptoflux import AdaptoFlux

def main():
    # 🔸 修改：指定 methods_path

    # 生成二分类数据
    np.random.seed(42)
    X = np.random.randn(500, 1).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)  # 0 或 1

    af = AdaptoFlux(values=X, labels=y, methods_path="Test/test_train/test_combined_trainer/method.py")  # 👈 关键！

    layer_grow_config = {
        "max_layers": 2,          # 降低层数，加速测试
        "max_attempts": 2,
        "decision_threshold": 0.0
    }

    graph_evo_config = {
        "num_initial_models": 2,
        "max_refinement_steps": 5,
        "enable_evolution": False,  # 🔸 先禁用进化，简化测试
        "enable_compression": False,
        "frozen_nodes": ["root", "collapse"],
        "refinement_strategy": "random_single"
    }

    from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer

    trainer = CombinedTrainer(
        adaptoflux_instance=af,
        layer_grow_config=layer_grow_config,
        graph_evo_config=graph_evo_config,
        num_evolution_cycles=1,   # 🔸 先跑1轮
        save_dir="Test/test_train/test_combined_trainer/log",
        verbose=True,
        genetic_mode="disabled",  # 🔸 先禁用遗传（可选）
        refine_only_new_layers=True
    )

    print("\n🚀 开始训练...")
    results = trainer.train(X, y)

    print("\n✅ 训练完成！最佳准确率:", results["best_overall_accuracy"])

if __name__ == "__main__":
    main()