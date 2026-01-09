# analyze_exp3_chain_from_logs.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_RESULT_DIR = "experiments/GraphEvo_exp3/exp3_results"
EXP_GROUPS = ["Exp_L2_Minimal", "Exp_L2_Extended", "Exp_L3_Minimal", "Oracle_L2"]
N_REPEATS = 100

# 输出目录
OUTPUT_DIR = "experiments/GraphEvo_exp3/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 精度阈值
ACC_THRESHOLD_TASK12 = 0.999
ACC_THRESHOLD_TASK3 = 0.95

def load_best_accuracy(log_path):
    if not os.path.exists(log_path):
        return 0.0
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
        return float(log.get("best_accuracy", 0.0))
    except Exception as e:
        print(f"⚠️ Failed to load {log_path}: {e}")
        return 0.0

# 收集数据
chain_stats = {}

for group in EXP_GROUPS:
    total = 0
    f1_learned = 0
    f1_f2_learned = 0
    full_chain = 0
    task3_success = 0

    group_dir = os.path.join(BASE_RESULT_DIR, group)
    for rep in range(N_REPEATS):
        rep_dir = os.path.join(group_dir, f"rep{rep}")
        if not os.path.isdir(rep_dir):
            continue

        acc1 = load_best_accuracy(os.path.join(rep_dir, "task1", "graph_evo_training_log.json"))
        acc2 = load_best_accuracy(os.path.join(rep_dir, "task2", "graph_evo_training_log.json"))
        acc3 = load_best_accuracy(os.path.join(rep_dir, "task3", "graph_evo_training_log.json"))

        task1_ok = acc1 >= ACC_THRESHOLD_TASK12
        task2_ok = acc2 >= ACC_THRESHOLD_TASK12
        task3_ok = acc3 > ACC_THRESHOLD_TASK3

        total += 1
        if task1_ok:
            f1_learned += 1
        if task1_ok and task2_ok:
            f1_f2_learned += 1
        if task1_ok and task2_ok and task3_ok:
            full_chain += 1
        if task3_ok:
            task3_success += 1

    chain_stats[group] = {
        "total": total,
        "task1_perfect_rate": f1_learned / total,
        "task1_task2_perfect_rate": f1_f2_learned / total,
        "full_chain_rate": full_chain / total,
        "task3_success_rate": task3_success / total,
    }

# === 打印并保存表格 ===
table_lines = []
table_lines.append(f"{'Group':<20} {'Task1=1.0':<12} {'Task1+2=1.0':<14} {'Full Chain':<12} {'Task3>0.95':<12}")
table_lines.append("-" * 70)
for group, stat in chain_stats.items():
    line = (
        f"{group:<20} "
        f"{stat['task1_perfect_rate']:<12.1%} "
        f"{stat['task1_task2_perfect_rate']:<14.1%} "
        f"{stat['full_chain_rate']:<12.1%} "
        f"{stat['task3_success_rate']:<12.1%}"
    )
    table_lines.append(line)
    print(line)

# 保存为 TXT
with open(os.path.join(OUTPUT_DIR, "exp3_chain_analysis.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines))

# 保存为 JSON
with open(os.path.join(OUTPUT_DIR, "exp3_chain_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(chain_stats, f, indent=4)

print(f"\n✅ Results saved to: {OUTPUT_DIR}")

# === 绘图 ===
groups = list(chain_stats.keys())
x = np.arange(len(groups))

task1_rate = [chain_stats[g]["task1_perfect_rate"] for g in groups]
task12_rate = [chain_stats[g]["task1_task2_perfect_rate"] for g in groups]
full_chain_rate = [chain_stats[g]["full_chain_rate"] for g in groups]

no_f1 = [1 - r for r in task1_rate]
f1_only = [t1 - t12 for t1, t12 in zip(task1_rate, task12_rate)]
f1_f2_no_task3 = [t12 - fc for t12, fc in zip(task12_rate, full_chain_rate)]
full_chain = full_chain_rate

plt.figure(figsize=(9, 5))
bottom = np.zeros(len(groups))

plt.bar(x, no_f1, bottom=bottom, label='Task1 not perfect', color='lightgray')
bottom += np.array(no_f1)

plt.bar(x, f1_only, bottom=bottom, label='Task1 perfect, Task2 not', color='orange')
bottom += np.array(f1_only)

plt.bar(x, f1_f2_no_task3, bottom=bottom, label='Task1+2 perfect, Task3 failed', color='skyblue')
bottom += np.array(f1_f2_no_task3)

plt.bar(x, full_chain, bottom=bottom, label='Full chain success', color='green')

plt.xticks(x, groups, rotation=15)
plt.ylabel('Proportion of Runs')
plt.title('Knowledge Transfer Chain: Task1 → Task2 → Task3')
plt.legend()
plt.tight_layout()

# 保存图像
img_path = os.path.join(OUTPUT_DIR, "exp3_knowledge_chain_from_logs.png")
plt.savefig(img_path, dpi=300)
print(f"✅ Plot saved to: {img_path}")
plt.show()