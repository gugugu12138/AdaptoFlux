# experiments/embodied_snake_n/snake_train.py
import numpy as np
import logging
import sys
import os
import json
from copy import deepcopy

from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from .snake_env import SimpleSnakeEnv, run_snake_episode
from .snake_state import SNAKE_STATE

# === 全局记录 ===
SURVIVAL_HISTORY = []
EVAL_COUNTER = 0
BEST_FITNESS = -np.inf

def collapse_snake(value):
    """
    优化：期望模型输出 3 个值，分别代表 [Left, Straight, Right] 的得分
    选择得分最高的动作
    """
    # 确保至少有 3 个输出，如果不足则补零
    while len(value) < 3:
        value = np.append(value, 0.0)
    
    # 取前 3 个值进行决策
    scores = np.array(value[:3])
    action = np.argmax(scores) 
    # 映射：0->Left, 1->Straight, 2->Right (需与 env 一致)
    # 注意：env 中 0=直，1=左，2=右。这里 argmax 0 对应 Left。
    # 为了对齐，我们调整映射：
    # value[0] -> Left (1)
    # value[1] -> Straight (0)
    # value[2] -> Right (2)
    
    if np.argmax(scores) == 0:
        SNAKE_STATE.decided_action = 1 # Left
    elif np.argmax(scores) == 1:
        SNAKE_STATE.decided_action = 0 # Straight
    else:
        SNAKE_STATE.decided_action = 2 # Right
        
    return value[:3] # 返回处理后的值

def evaluate_model_fitness(model, n_episodes=5, max_steps=2000):
    """
    多次评估取平均，减少随机性带来的噪声
    """
    total_survival = 0
    total_score = 0
    total_reward = 0
    
    for _ in range(n_episodes):
        survival, score, reward = run_snake_episode(model, action_interval=1, max_steps=max_steps)
        total_survival += survival
        total_score += score
        total_reward += reward
        
    avg_survival = total_survival / n_episodes
    avg_score = total_score / n_episodes
    avg_reward = total_reward / n_episodes
    
    # 综合适应度：生存是基础，食物是核心目标
    fitness = avg_survival + (avg_score * 500) 
    return fitness, avg_survival, avg_score, avg_reward

def snake_loss(model, input_data, target):
    global BEST_FITNESS
    fitness, survival, score, reward = evaluate_model_fitness(model, n_episodes=3)
    
    # 记录日志
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    
    # 只有当发现更好的模型时才详细打印，减少刷屏
    is_best = False
    if fitness > BEST_FITNESS:
        BEST_FITNESS = fitness
        is_best = True
        
    log_entry = {
        'eval_id': EVAL_COUNTER,
        'fitness': fitness,
        'survival': survival,
        'score': score,
        'is_best': is_best
    }
    SURVIVAL_HISTORY.append(log_entry)
    
    if EVAL_COUNTER % 5 == 0 or is_best:
        print(f"[Eval {EVAL_COUNTER}] Fit: {fitness:.1f}, Surv: {survival:.1f}, Score: {score:.1f} {'***NEW BEST***' if is_best else ''}")
    
    # 优化器通常最小化 Loss，所以返回负适应度
    return -fitness

def snake_acc(model, input_data, target):
    # 准确率定义为归一化的适应度
    fitness, _, _, _ = evaluate_model_fitness(model, n_episodes=1)
    # 假设理论最大适应度约为 5000 + 20*500 = 15000
    return max(0.0, min(1.0, fitness / 15000.0))

# === 初始化 AdaptoFlux ===
# 输入维度更新为 8 (相对坐标 + 方向 + 危险传感器 + Bias)
dummy_input = np.zeros((1, 8), dtype=np.float32)

af = AdaptoFlux(
    values=dummy_input,
    labels=None,
    methods_path="experiments/embodied_snake_n/methods_snake.py",
    input_types_list=['scalar'] * 8 
)

af.set_custom_collapse(collapse_snake)

lg_train_kwargs = {"max_layers": 12, "max_total_attempts": 3000, "rollback_layers": 2}
ge_train_kwargs = {"max_evo_cycles": 10} # 增加进化周期

combined_config = {
    "layer_grow_config": {"max_attempts": 15, "verbose": False},
    "graph_evo_config": {"refinement_strategy": "full_sweep", "verbose": False},
    "num_evolution_cycles": 5,
    "target_subpool_size": 12,
    "verbose": True,
    "save_dir": "experiments/embodied_snake_n/model",
    "lg_train_kwargs": lg_train_kwargs,
    "ge_train_kwargs": ge_train_kwargs,
    "custom_loss_evaluator": snake_loss,
    "custom_accuracy_evaluator": snake_acc
}

trainer = CombinedTrainer(adaptoflux_instance=af, **combined_config)

# 注意：input_data 在这里主要是为了初始化图结构，实际推理数据在 snake_loss 中通过 env 生成
trainer.train(input_data=dummy_input, target=None)

os.makedirs("experiments/embodied_snake_n/results_guided", exist_ok=True)

# === 最终评估 ===
print("\n=== Final Evaluation (10 Episodes) ===")
final_fit, final_surv, final_score, _ = evaluate_model_fitness(trainer.adaptoflux, n_episodes=10, max_steps=5000)
print(f"Final Avg Survival: {final_surv:.1f} frames")
print(f"Final Avg Score: {final_score:.1f} foods")

# === 保存结果 ===
history_path = "experiments/embodied_snake_n/results_guided/survival_history.json"
with open(history_path, "w") as f:
    json.dump(SURVIVAL_HISTORY, f, indent=2)
    
if SURVIVAL_HISTORY:
    best = max(SURVIVAL_HISTORY, key=lambda x: x['fitness'])
    print(f"Best Fitness: {best['fitness']:.1f} (Eval #{best['eval_id']})")