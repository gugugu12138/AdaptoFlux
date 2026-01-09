import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def collect_experiment_results(base_dir="experiments/LayerGrowTrainer_exp1"):
    """
    Traverse experiment directory and collect all training_log.json results.
    """
    results = []
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Experiment base directory not found: {base_path}")

    # Find all training_log.json files
    for log_file in base_path.rglob("training_log.json"):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse max_layers and run from path
            parts = log_file.parts
            max_layers = None
            run = None
            for part in parts:
                if part.startswith("exp_max_layers_"):
                    max_layers = int(part.split("_")[-1])
                elif part.startswith("run_"):
                    run = int(part.split("_")[1])

            if max_layers is None or run is None:
                print(f"⚠️ Could not parse max_layers or run from path: {log_file}")
                continue

            result = {
                "max_layers": max_layers,
                "run": run,
                "best_model_accuracy": data.get("best_model_accuracy", 0.0),
                "best_model_layers": data.get("best_model_layers", 0),
                "layers_added": data.get("layers_added", 0),
                "total_growth_attempts": data.get("total_growth_attempts", 0),
                "rollback_count": data.get("rollback_count", 0),
                "total_candidate_attempts": data.get("total_candidate_attempts", 0)
            }
            results.append(result)

        except Exception as e:
            print(f"❌ Failed to read {log_file}: {e}")

    return pd.DataFrame(results)

def load_duration_data(json_path):
    """
    Load duration data from experiment_results.json grouped by max_layers.
    Returns sorted layers and average durations, plus raw data for plotting.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载 {len(data)} 条训练记录用于耗时分析")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{json_path}'，请检查路径是否正确。")
        return None, None, None, None
    except json.JSONDecodeError as e:
        print(f"❌ 错误：JSON 文件格式有误，无法解析。错误信息：{e}")
        return None, None, None, None

    grouped = defaultdict(list)
    all_layers = []
    all_durations = []

    for item in data:
        if isinstance(item, dict) and 'max_layers' in item and 'duration' in item:
            max_layers = item['max_layers']
            duration = item['duration']
            grouped[max_layers].append(duration)
            all_layers.append(max_layers)
            all_durations.append(duration)
        else:
            print("⚠️ 警告：发现格式错误的数据条目，已跳过。")

    if not grouped:
        print("❌ 没有有效的耗时数据可供绘图。")
        return None, None, None, None

    layers = sorted(grouped.keys())
    avg_durations = [np.mean(grouped[l]) for l in layers]

    return layers, avg_durations, all_layers, all_durations

def plot_experiment_summary(df, duration_json_path, output_dir="experiments/LayerGrowTrainer_exp1"):
    """
    Plot summary charts using only matplotlib, with training duration replacing layers_added plot.
    """
    # Performance metrics grouped by max_layers
    grouped_metrics = df.groupby("max_layers").agg({
        "best_model_accuracy": "mean",
        "best_model_layers": "mean",
        "rollback_count": "mean",
        "total_growth_attempts": "mean"
    }).reset_index()

    # Load duration data
    layers, avg_durations, all_layers, all_durations = load_duration_data(duration_json_path)
    if layers is None:
        print("⚠️ 跳过耗时图表绘制。")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("LayerGrowTrainer Experiment Summary", fontsize=16, weight='bold')

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 1. Best Accuracy vs Max Layers
    axes[0, 0].plot(grouped_metrics["max_layers"], grouped_metrics["best_model_accuracy"],
                    marker='o', color=color_cycle[0], linewidth=2)
    axes[0, 0].set_title("Avg Best Accuracy vs Max Layers")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Best Model Layers vs Max Layers
    axes[0, 1].plot(grouped_metrics["max_layers"], grouped_metrics["best_model_layers"],
                    marker='s', color=color_cycle[1], linewidth=2)
    axes[0, 1].set_title("Avg Best Model Layers vs Max Layers")
    axes[0, 1].set_ylabel("Number of Layers")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Training Duration vs Max Layers (REPLACED)
    axes[1, 0].plot(layers, avg_durations, 'o-', color='#2E8B57', linewidth=2, markersize=6, label='Average Duration')
    axes[1, 0].scatter(all_layers, all_durations, alpha=0.4, color='lightcoral', s=20, label='Single Run')
    axes[1, 0].set_title("Training Duration vs Max Layers")
    axes[1, 0].set_ylabel("Duration (seconds)")
    axes[1, 0].set_xlabel("Max Layers")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Rollback Count vs Max Layers
    axes[1, 1].plot(grouped_metrics["max_layers"], grouped_metrics["rollback_count"],
                    marker='d', color=color_cycle[3], linewidth=2)
    axes[1, 1].set_title("Avg Rollback Count vs Max Layers")
    axes[1, 1].set_ylabel("Rollbacks")
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Total Growth Attempts vs Max Layers
    axes[2, 0].plot(grouped_metrics["max_layers"], grouped_metrics["total_growth_attempts"],
                    marker='x', color=color_cycle[4], linewidth=2)
    axes[2, 0].set_title("Avg Total Growth Attempts vs Max Layers")
    axes[2, 0].set_ylabel("Attempts")
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Accuracy vs Layers Added (colored by max_layers)
    scatter = axes[2, 1].scatter(df["layers_added"], df["best_model_accuracy"],
                                 c=df["max_layers"], cmap='viridis', alpha=0.7)
    axes[2, 1].set_title("Accuracy vs Layers Added (color: max_layers)")
    axes[2, 1].set_xlabel("Layers Added")
    axes[2, 1].set_ylabel("Best Accuracy")
    axes[2, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2, 1], label="Max Layers")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    output_path = os.path.join(output_dir, "experiment_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 Chart saved to: {output_path}")

def main():
    base_dir = "experiments/LayerGrowTrainer_exp1"
    duration_json_path = os.path.join(base_dir, "experiment_results.json")
    print("🔍 Collecting experiment results...")

    df = collect_experiment_results(base_dir)

    if df.empty:
        print("❌ No experiment data found!")
        return

    print(f"✅ Loaded {len(df)} experiment records.")
    print(df.head())

    # Save to CSV
    csv_path = os.path.join(base_dir, "experiment_results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Data saved to: {csv_path}")

    # Plot with duration data
    plot_experiment_summary(df, duration_json_path, output_dir=base_dir)

if __name__ == "__main__":
    main()