import numpy as np
import random
import os
import sys

# 确保项目根目录在 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from experiments.minimal_stateful_comparison.env import STATE

# ==============================================================================
# 1. 评估函数
# ==============================================================================
def evaluate_stateful_model(model, num_episodes=100, max_steps=4):
    total_distance = 0.0
    success_count = 0
    for _ in range(num_episodes):
        obs = STATE.reset()
        for step in range(max_steps):
            _ = model.infer_with_graph(obs)
            if STATE.is_success():
                success_count += 1
                break
            obs = STATE.get_obs()
        dist = abs(int(round(STATE.pos)) - int(round(STATE.target)))
        total_distance += dist
    return float(total_distance / num_episodes), float(success_count / num_episodes)

# 训练时使用 20 episodes 评估以加速，最终验证使用 100 episodes
def stateful_success_loss(model, input_data, target):
    loss, _ = evaluate_stateful_model(model, num_episodes=20, max_steps=4)
    return loss

def stateful_accuracy(model, input_data, target):
    _, acc = evaluate_stateful_model(model, num_episodes=20, max_steps=4)
    return acc

# ==============================================================================
# 2. 单次训练与评估函数
# ==============================================================================
def run_single_experiment(config_name, methods_path, run_id, base_seed):
    # 🔥 核心：设置相同的随机种子，确保两种配置面对完全相同的随机初始化序列
    seed = base_seed + run_id
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n[{config_name.upper()}] Starting Run {run_id+1}/10 (Seed: {seed})...")
    
    dummy_input = np.array([[0.0, 0.0]], dtype=np.float32)
    
    # 根据配置决定 Root 节点的输入类型
    input_types = ['pos_type', 'target_type'] if config_name == "constrained" else ['scalar', 'scalar']
        
    af = AdaptoFlux(
        values=dummy_input,
        labels=None,
        methods_path=methods_path,
        input_types_list=input_types
    )
    
    ge_config = {
        "verbose": False, # 关闭详细日志，保持输出整洁
        "init_mode": "fixed",
        "max_init_layers": 3,
        "num_initial_models": 3,
        "max_refinement_steps": 30,
        "enable_evolution": False,
        "candidate_pool_mode": "all"
    }
    
    save_dir = f"experiments/minimal_stateful_comparison/results/{config_name}/run_{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = GraphEvoTrainer(
        adaptoflux_instance=af,
        custom_loss_evaluator=stateful_success_loss,
        custom_accuracy_evaluator=stateful_accuracy,
        task_type="regression",
        save_dir=save_dir,
        **ge_config
    )
    
    # 训练
    trainer.train(
        input_data=dummy_input,
        target=np.array([0.0]),
        max_evo_cycles=5,
        enable_early_stop=True,
        early_stop_eps=0.0,
        save_model=True,
        model_save_path=save_dir
    )
    
    # 最终严格验证 (100 episodes)
    final_loss, final_acc = evaluate_stateful_model(trainer.adaptoflux, num_episodes=100, max_steps=4)
    
    is_success = final_acc >= 0.95
    print(f"[{config_name.upper()}] Run {run_id+1} Finished. Final Acc: {final_acc*100:.1f}% | {'✅ SUCCESS' if is_success else '❌ FAILED'}")
    
    return is_success, final_acc

# ==============================================================================
# 3. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    NUM_RUNS = 10
    BASE_SEED = 42
    
    configs = [
        ("unconstrained", "experiments/minimal_stateful_comparison/methods_unconstrained.py"),
        ("constrained", "experiments/minimal_stateful_comparison/methods_constrained.py")
    ]
    
    results_summary = {}
    
    for config_name, methods_path in configs:
        print("\n" + "="*70)
        print(f"STARTING EXPERIMENT: {config_name.upper()} ({NUM_RUNS} Runs)")
        print("="*70)
        
        success_count = 0
        acc_list = []
        
        for run_id in range(NUM_RUNS):
            is_success, final_acc = run_single_experiment(config_name, methods_path, run_id, BASE_SEED)
            if is_success:
                success_count += 1
            acc_list.append(final_acc)
            
        results_summary[config_name] = {
            "success_rate": success_count / NUM_RUNS,
            "success_count": success_count,
            "avg_acc": np.mean(acc_list)
        }
        
    # 打印最终对比报告
    print("\n" + "="*70)
    print("FINAL COMPARISON REPORT (10 Independent Runs)")
    print("="*70)
    for config_name, stats in results_summary.items():
        print(f"[{config_name.upper()}] Success Rate: {stats['success_count']}/{NUM_RUNS} "
              f"({stats['success_rate']*100:.1f}%) | Avg Final Acc: {stats['avg_acc']*100:.2f}%")