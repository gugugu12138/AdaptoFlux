import numpy as np
import random
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt  # 🔥 新增：引入绘图库

# 确保项目根目录在 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from experiments.minimal_stateful_comparison.env import STATE

# ==============================================================================
# 0. 日志配置 (修复缓冲问题，确保日志文件不为空)
# ==============================================================================
log_dir = "experiments/minimal_stateful_comparison/logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(message)s')

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
    seed = base_seed + run_id
    random.seed(seed)
    np.random.seed(seed)
    
    logging.info(f"\n[{config_name.upper()}] Starting Run {run_id+1}/10 (Seed: {seed})...")
    
    dummy_input = np.array([[0.0, 0.0]], dtype=np.float32)
    input_types = ['pos_type', 'target_type'] if config_name == "constrained" else ['scalar', 'scalar']
        
    af = AdaptoFlux(
        values=dummy_input,
        labels=None,
        methods_path=methods_path,
        input_types_list=input_types
    )
    
    ge_config = {
        "verbose": False,
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
    
    trainer.train(
        input_data=dummy_input,
        target=np.array([0.0]),
        max_evo_cycles=5,
        enable_early_stop=True,
        early_stop_eps=0.0,
        save_model=True,
        model_save_path=save_dir
    )
    
    final_loss, final_acc = evaluate_stateful_model(trainer.adaptoflux, num_episodes=100, max_steps=4)
    
    is_success = final_acc >= 0.95
    logging.info(f"[{config_name.upper()}] Run {run_id+1} Finished. Final Acc: {final_acc*100:.1f}% | {'✅ SUCCESS' if is_success else '❌ FAILED'}")
    
    return is_success, final_acc

# ==============================================================================
# 3. 绘图函数 (🔥 新增)
# ==============================================================================
def plot_experiment_results(results_summary, save_dir="experiments/minimal_stateful_comparison/results"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建 1x2 的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    labels = ['Unconstrained\nBaseline', 'Type-Constrained\nAdaptoFlux']
    success_rates = [
        results_summary['unconstrained']['success_rate'] * 100, 
        results_summary['constrained']['success_rate'] * 100
    ]
    
    # --- 图 1: 成功率对比柱状图 ---
    colors = ['#d62728', '#2ca02c'] # 红色(失败), 绿色(成功)
    bars = ax1.bar(labels, success_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.set_title('Automatic Synthesis Success Rate\n(10 Independent Runs)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 在柱状图上方添加具体数值标签
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 3, f'{yval:.0f}%', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
    # --- 图 2: 每次运行的准确率分布 ---
    ax2.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.set_title('Final Accuracy per Independent Run', fontsize=12, fontweight='bold')
    
    # 绘制 95% 成功阈值线
    ax2.axhline(95, color='gray', linestyle='--', linewidth=1.5, label='Success Threshold (95%)')
    
    runs = np.arange(1, 11)
    acc_unconstrained = [acc * 100 for acc in results_summary['unconstrained']['acc_per_run']]
    acc_constrained = [acc * 100 for acc in results_summary['constrained']['acc_per_run']]
    
    # 绘制折线和散点
    ax2.plot(runs, acc_unconstrained, marker='o', color=colors[0], label='Unconstrained', linestyle='--', alpha=0.7)
    ax2.plot(runs, acc_constrained, marker='s', color=colors[1], label='Type-Constrained', linestyle='-', linewidth=2)
    
    ax2.scatter(runs, acc_unconstrained, color=colors[0], s=50, zorder=5)
    ax2.scatter(runs, acc_constrained, color=colors[1], s=50, zorder=5)
    
    ax2.set_xticks(runs)
    ax2.set_xlabel('Run Index (Random Seed)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(save_dir, "stateful_synthesis_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"📊 Experiment plot successfully saved to: {save_path}")
    plt.close()

# ==============================================================================
# 4. 主执行逻辑
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
        logging.info("\n" + "="*70)
        logging.info(f"STARTING EXPERIMENT: {config_name.upper()} ({NUM_RUNS} Runs)")
        logging.info("="*70)
        
        success_count = 0
        acc_list = []
        
        for run_id in range(NUM_RUNS):
            is_success, final_acc = run_single_experiment(config_name, methods_path, run_id, BASE_SEED)
            if is_success:
                success_count += 1
            acc_list.append(final_acc)
            
        # 🔥 修改：将每次运行的准确率列表也存入字典，供绘图使用
        results_summary[config_name] = {
            "success_rate": success_count / NUM_RUNS,
            "success_count": success_count,
            "avg_acc": np.mean(acc_list),
            "acc_per_run": acc_list  
        }
        
    # 打印最终对比报告
    logging.info("\n" + "="*70)
    logging.info("FINAL COMPARISON REPORT (10 Independent Runs)")
    logging.info("="*70)
    for config_name, stats in results_summary.items():
        logging.info(f"[{config_name.upper()}] Success Rate: {stats['success_count']}/{NUM_RUNS} "
              f"({stats['success_rate']*100:.1f}%) | Avg Final Acc: {stats['avg_acc']*100:.2f}%")
    
    logging.info("="*70)
    logging.info(f"✅ All experiments finished. Log saved to: {log_filename}")
    logging.info("="*70)

    # 🔥 新增：调用绘图函数生成论文用图
    plot_experiment_results(results_summary)

    # 强制刷新缓冲区并关闭所有文件句柄
    logging.shutdown()