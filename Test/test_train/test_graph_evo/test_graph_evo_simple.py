# test_graph_evo_simple.py
import os
import numpy as np
import logging
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 简单任务函数 ---
def simple_task(x):
    return (x + 1) * 2  # f1

def generate_simple_data(n=100):
    x = np.random.uniform(-5, 5, (n, 1)).astype(np.float32)
    y = simple_task(x).astype(np.float32)
    return x, y

# --- 构建最小 AdaptoFlux 实例 ---
def create_minimal_adaptoflux():
    dummy_x = np.array([[0.0]], dtype=np.float32)
    af = AdaptoFlux(values=dummy_x, methods_path="Test/test_train/test_graph_evo/dummy_methods.py") 

    # 添加基础方法
    base_methods = [
        ("add_1", lambda x: [x + 1], 1, 1),
        ("mul_2", lambda x: [x * 2], 1, 1),
        ("identity", lambda x: [x], 1, 1),
    ]

    for name, func, in_count, out_count in base_methods:
        af.add_method(
            method_name=name,
            method=func,
            input_count=in_count,
            output_count=out_count,
            input_types=['scalar'],
            output_types=['scalar'],
            group='math',
            weight=1.0,
            vectorized=False
        )
    return af

# --- 主测试函数 ---
def test_graph_evo_trainer_basic():
    logger.info("🚀 Starting Simple GraphEvoTrainer Test...")

    # 生成数据
    X, y = generate_simple_data(n=200)

    # 创建 AdaptoFlux 实例
    af = create_minimal_adaptoflux()

    # 创建 trainer
    trainer = GraphEvoTrainer(
        adaptoflux_instance=af,
        num_initial_models=3,          # 减少候选数，加快测试
        max_refinement_steps=10,       # 减少精炼步数
        max_init_layers=2,             # 最多初始化2层
        enable_evolution=False,        # 先关闭进化，专注测试初始化+精炼
        refinement_strategy="random_single",  # 使用轻量策略
        candidate_pool_mode="group",
        fallback_mode="group_first",
        verbose=True
    )

    # 执行训练（只跑1个 cycle）
    result = trainer.train(X, y, max_evo_cycles=1, model_save_path=None, save_model=False)

    logger.info(f"✅ Training completed.")
    logger.info(f"Final loss: {result['final_loss']:.6f}")
    logger.info(f"Best accuracy: {result['best_accuracy']:.6f}")
    logger.info(f"Total refinement attempts: {trainer._total_refinement_attempts}")

    # 验证结果合理性
    assert result['best_accuracy'] >= 0.0, "Accuracy should be non-negative"
    assert 'final_loss' in result, "Result should contain final_loss"

    logger.info("🎉 Simple test passed!")


if __name__ == "__main__":
    test_graph_evo_trainer_basic()