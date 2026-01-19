import numpy as np
import logging
import json
import os
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from .bird_env import run_bird_episode, gym  # 确保导入 gym

logging.basicConfig(level=logging.INFO)
            
import time  # 用于控制帧率

def test_oracle(action_interval=5, max_steps=5000, render=True):
    render_mode = "human" if render else None
    env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=render_mode)
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    survival_time = 0
    jump_count = 0
    decision_steps = 0

    try:
        for frame in range(max_steps):
            action = 0
            if frame % action_interval == 0:
                bird_y = obs[9]
                bird_vy = obs[10]
                next_pipe_dist = obs[3]
                next_pipe_top = obs[4]
                next_pipe_bottom = obs[5]

                # 检查是否为初始无效管道
                is_initial_state = (next_pipe_top >= 0.99 and next_pipe_bottom <= 0.01)
                is_pipe_far = next_pipe_dist > 0.95

                if is_initial_state or is_pipe_far:
                    # 初始阶段：维持安全高度（~0.4~0.6）
                    if bird_y < 0.5:
                        action = 1
                        jump_count += 1
                        print(">>> JUMP! (initial altitude control)")
                else:
                    # 真实管道阶段
                    gap_center = (next_pipe_top + next_pipe_bottom) / 2.0
                    SAFE_MARGIN = 0.07

                    print(f"[ORACLE] Frame {frame} | Y: {bird_y:.3f} | Vy: {bird_vy:.3f} | "
                          f"Gap: [{next_pipe_bottom:.3f}, {next_pipe_top:.3f}] | Dist: {next_pipe_dist:.3f}")

                    # 只在必要时跳：防止撞下管 或 快速下落
                    if bird_y < next_pipe_bottom + SAFE_MARGIN:
                        action = 1
                        jump_count += 1
                        print(">>> JUMP! (avoid lower pipe)")

                decision_steps += 1

            else:
                action = 0

            obs, _, terminated, truncated, _ = env.step(action)
            survival_time += 1

            if terminated or truncated:
                print(f"Game over at frame {survival_time}")
                break

            if render:
                time.sleep(0.05)

    finally:
        env.close()

    jump_rate = jump_count / max(decision_steps, 1)
    print(f"\n[ORACLE TEST] Survival: {survival_time}, JumpRate: {jump_rate:.2f}")
    return survival_time

# === 运行 Oracle 测试 ===
print("=== Testing Oracle Policy ===")
oracle_survival = test_oracle(action_interval=1, max_steps=5000)
print(f"Oracle survival time: {oracle_survival} frames\n")

# 如果 Oracle 表现差，说明环境或策略有误
if oracle_survival <= 50:
    print("⚠️ Warning: Oracle policy performs poorly. Check environment or logic!")
    exit(1)

# === 初始化 AdaptoFlux ===
dummy_input = np.zeros((1, 5), dtype=np.float32)  # ← 改回 5 维！
af = AdaptoFlux(
    values=dummy_input,
    labels=None,
    methods_path="experiments/embodied_bird/methods_bird.py",
    input_types_list=['raw_signal'] * 5  # ← 5 维输入
)

# 加载手工构建的初始图
af.load_model("experiments/embodied_bird/initial_scaffold.json")  # 确保路径正确

# # === 全局记录 ===
# SURVIVAL_HISTORY = []
# EVAL_COUNTER = 0

# def evaluate_with_logging(model, action_interval=1, max_steps=5000):
#     global EVAL_COUNTER
#     survival, avg_deviation, jump_rate = run_bird_episode(model, action_interval, max_steps)
#     EVAL_COUNTER += 1
#     SURVIVAL_HISTORY.append({
#         'eval_id': EVAL_COUNTER,
#         'survival': survival,
#         'avg_deviation': avg_deviation,
#         'jump_rate': jump_rate
#     })
#     if EVAL_COUNTER % 10 == 0:
#         print(f"[Eval {EVAL_COUNTER}] Survival: {survival}, Dev: {avg_deviation:.3f}")
#     return survival, avg_deviation, jump_rate

# # === 损失函数 ===
# def bird_loss(model, input_data, target):
#     survival, _, _ = evaluate_with_logging(model)
#     loss = -survival
#     return float(loss)  # 暂时移除 pipe_bottom 检查（因已修正初始图）

# def bird_acc(model, input_data, target):
#     survival, _, _ = evaluate_with_logging(model)
#     return min(survival, 5000) / 5000.0

# # === 配置纯 GraphEvo ===
# lg_config = {"max_attempts": 0}
# ge_config = {
#     "verbose": False,
#     "init_mode": "loaded",
#     "max_init_layers": 5
# }

# trainer = CombinedTrainer(
#     adaptoflux_instance=af,
#     layer_grow_config=lg_config,
#     graph_evo_config=ge_config,
#     num_evolution_cycles=5,
#     genetic_mode="disabled",
#     save_dir="experiments/embodied_bird/results_guided",
#     verbose=True,
#     custom_loss_evaluator=bird_loss,
#     custom_accuracy_evaluator=bird_acc,
#     task_type="regression",
# )

# os.makedirs("experiments/embodied_bird/results_guided", exist_ok=True)

# # === 执行训练 ===
# print("Starting Human-Guided Graph Evolution...")
# results = trainer.train(
#     input_data=dummy_input,
#     target=np.array([0.0]),
#     enable_early_stop=False
# )

# === 最终评估 ===
final_survival, _, _ = run_bird_episode(trainer.adaptoflux, action_interval=5)
print(f"\nFinal survival: {final_survival} frames")
print(f"Oracle survival: {oracle_survival} frames")

# # === 保存结果 ===
# history_path = "experiments/embodied_bird/results_guided/survival_history.json"
# with open(history_path, "w") as f:
#     json.dump(SURVIVAL_HISTORY, f, indent=2)
    
# if SURVIVAL_HISTORY:
#     best = max(SURVIVAL_HISTORY, key=lambda x: x['survival'])
#     print(f"Best survival: {best['survival']} (Eval #{best['eval_id']})")