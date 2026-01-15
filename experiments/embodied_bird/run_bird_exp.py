# experiments/embodied_bird/run_bird_exp.py
import numpy as np
import logging
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from .bird_env import run_bird_episode

# experiments/embodied_bird/run_bird_exp.py （顶部附近）

import math
import random


# Set up logging
logging.basicConfig(level=logging.INFO)

# Dummy input shape (FlappyBird observation dim = 8)
dummy_input = np.zeros((1, 8), dtype=np.float32)

# Initialize AdaptoFlux (no labels needed!)
af = AdaptoFlux(
    values=dummy_input,
    labels=None,
    methods_path="experiments/embodied_bird/methods_bird.py"
)

# Optional: set collapse function (not used, but required by framework)
def collapse_first(values):
    return values[0] if len(values) > 0 else 0.0
af.set_custom_collapse(collapse_first)

# Define custom evaluators
def bird_loss(model, input_data, target):
    survival = run_bird_episode(model, action_interval=5, max_steps=5000)
    
    # 安全获取层数
    num_layers = 0
    if hasattr(model, 'graph_processor') and hasattr(model.graph_processor, 'layer'):
        num_layers = model.graph_processor.layer

    # 主目标：最大化存活时间 → -survival
    base_loss = -survival

    # 深度奖励/惩罚逻辑
    if num_layers <= 5:
        # 鼓励适度加深：每层带来微小收益（减少损失）
        depth_bonus = -0.01 * num_layers
    else:
        # 惩罚过深：超出5层的部分，每层增加损失
        excess = num_layers - 5
        depth_penalty = 0.1 * excess  # 惩罚力度应显著大于奖励
        depth_bonus = depth_penalty

    total_loss = base_loss + depth_bonus
    return float(total_loss)

def bird_acc(model, input_data, target):
    survival = run_bird_episode(model, action_interval=5, max_steps=5000)
    return float(survival / 5000.0)  # normalized [0, 1]

# === 关键：将 custom evaluators 放入 config 字典中 ===
lg_config = {
    "max_attempts": 5,
    "verbose": False,
    # ↓ 这些会被传给 LayerGrowTrainer.__init__
    # "custom_loss_evaluator": bird_loss,
    # "custom_accuracy_evaluator": bird_acc,
    # 需要分别传输不同的loss和acc时可以使用这种方法，目前会直接根据combined_trainer的传递
}

ge_config = {
    "verbose": False,
    "init_mode": "fixed",
    "max_init_layers": 5,
    # ↓ 这些会被传给 GraphEvoTrainer.__init__
    # "custom_loss_evaluator": bird_loss,
    # "custom_accuracy_evaluator": bird_acc,
}


# Use existing CombinedTrainer with custom evaluators!
trainer = CombinedTrainer(
    adaptoflux_instance=af,
    layer_grow_config=lg_config,      # ← 包含 custom evaluators
    graph_evo_config=ge_config,       # ← 包含 custom evaluators
    num_evolution_cycles=3,
    genetic_mode="disabled",
    save_dir="experiments/embodied_bird/results",
    verbose=True,
    # ↓ 统一传入评估器（会被自动传递给子训练器）
    custom_loss_evaluator=bird_loss,
    custom_accuracy_evaluator=bird_acc,
    task_type="regression",  # 可选，但建议明确
)



# Train! (input_data and target are dummy; not used)
results = trainer.train(
    input_data=dummy_input,
    target=np.array([0.0])  # or np.array([0]) to satisfy shape check
)

# Final evaluation
final_survival = run_bird_episode(trainer.adaptoflux, action_interval=5)
print(f"Final survival time: {final_survival} frames")