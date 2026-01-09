# plot_method_pool_evolution_from_meta.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "Times New Roman"

BASE_DIR = "experiments/GraphEvo_exp3/exp3_results"

# 初始方法数量（与之前一致）
INITIAL_COUNTS = {
    "minimal": 5,    # increment, ×2, +3, ×0.5, identity
    "extended": 6,   # + add_2
    "oracle": 6      # + f1_direct
}

# 要分析的组（只关心启用进化的）
GROUPS = [
    ("Exp_L2_Minimal", "Minimal"),
    ("Exp_L2_Extended", "Extended"),
    ("Oracle_L2", "Oracle"),
]

def count_evolved_in_task(task_dir: str) -> int:
    """统计 task_dir/evolved_methods/ 下 .meta.json 文件数量"""
    evolved_dir = os.path.join(task_dir, "evolved_methods")
    if not os.path.exists(evolved_dir):
        return 0
    return len([f for f in os.listdir(evolved_dir) if f.endswith(".meta.json")])

def collect_evolution_data():
    results = {}
    for group_name, display_name in GROUPS:
        group_path = os.path.join(BASE_DIR, group_name)
        if not os.path.exists(group_path):
            print(f"⚠️ 跳过 {group_name}: 路径不存在")
            continue

        rep_dirs = [d for d in os.listdir(group_path) if d.startswith("rep") and os.path.isdir(os.path.join(group_path, d))]
        if not rep_dirs:
            print(f"⚠️ {group_name}: 无重复实验")
            continue

        initial_counts = []
        after_task1_counts = []
        after_task2_counts = []

        for rep in rep_dirs:
            rep_path = os.path.join(group_path, rep)
            # 从 metadata.json 获取 variant（用于确定初始方法数）
            meta_path = os.path.join(rep_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                variant = meta["config"]["method_pool_variant"]
                N0 = INITIAL_COUNTS.get(variant, 0)
                initial_counts.append(N0)

                # 统计 task1 和 task2 新增的进化方法
                task1_dir = os.path.join(rep_path, "task1")
                task2_dir = os.path.join(rep_path, "task2")
                n1 = count_evolved_in_task(task1_dir)
                n2 = count_evolved_in_task(task2_dir)

                after_task1_counts.append(N0 + n1)
                after_task2_counts.append(N0 + n1 + n2)

            except Exception as e:
                print(f"❌ 错误处理 {rep_path}: {e}")

        if after_task1_counts:
            results[display_name] = {
                "initial": (np.mean(initial_counts), 0.0),  # 初始是确定的
                "after_task1": (np.mean(after_task1_counts), np.std(after_task1_counts)),
                "after_task2": (np.mean(after_task2_counts), np.std(after_task2_counts)),
            }
        else:
            print(f"⚠️ {group_name}: 无有效数据")

    return results

def plot_three_stage_evolution(data):
    stages = ["initial", "after_task1", "after_task2"]
    stage_labels = ["Initial", "After Task 1", "After Task 2"]
    groups = list(data.keys())

    x = np.arange(len(stages))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, group in enumerate(groups):
        means = [data[group][stage][0] for stage in stages]
        stds = [data[group][stage][1] for stage in stages]
        bars = ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=group,
            capsize=5,
            alpha=0.9
        )
        # 可选：在柱子上显示数值
        for j, (mean, std) in enumerate(zip(means, stds)):
            ax.text(x[j] + i * width, mean + std + 0.1, f"{mean:.1f}", ha='center', fontsize=9)

    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Number of Methods in Pool")
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(stage_labels)
    ax.legend(title="Method Pool")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_dir = "experiments/GraphEvo_exp3"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "method_pool_three_stage_evolution.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "method_pool_three_stage_evolution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 三阶段方法池演化图已保存！")

if __name__ == "__main__":
    data = collect_evolution_data()
    if data:
        plot_three_stage_evolution(data)
    else:
        print("❌ 未收集到有效数据，请检查目录结构。")