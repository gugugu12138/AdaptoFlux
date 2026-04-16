# process_full_statistics.py
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置路径
INPUT_DIR = "experiments/ExternalBaselines/results"
INPUT_FILE = "all_results_by_collapse_with_baselines.json"
OUTPUT_DIR = "experiments/ExternalBaselines/processed_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print(f"📥 加载数据: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_stats(values):
    """计算均值、标准差、中位数、最小值"""
    if not values: return {"mean": 0, "std": 0, "median": 0, "min": 0}
    return {
        "mean": np.mean(values), "std": np.std(values), 
        "median": np.median(values), "min": np.min(values)
    }

def process_and_visualize(data):
    modes = list(data.keys()) # e.g., 'first', 'sum', 'prod'
    tasks = list(data[modes[0]].keys()) if modes else []
    
    plot_data = []

    for mode in modes:
        print(f"🔍 处理模式: {mode}")
        
        # LaTeX 表格初始化
        latex_lines = [
            r"\begin{table}[h]", r"\centering",
            r"\caption{AdaptoFlux performance with collapse=" + mode + r" (10 independent runs)}",
            r"\label{tab:atf_full_stats_" + mode + r"}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r"Task & MSE (Mean $\pm$ Std) & Median & Min & Success Rate (\%) & Runtime (s) $\pm$ Std \\",
            r"\midrule"
        ]
        
        for task in tasks:
            runs = data[mode][task]
            mses, runtimes = [], []
            exact_matches = 0
            
            # 遍历 10 次运行
            for run in runs:
                af = run["adaptoflux"]
                mses.append(af["mse"])
                runtimes.append(af.get("runtime_sec", 0))
                if af.get("exact_match", False): exact_matches += 1
                
                # 收集绘图数据
                plot_data.append({"Task": task, "Mode": mode, "MSE": af["mse"]})
            
            mse_s = compute_stats(mses)
            rt_s = compute_stats(runtimes)
            success_rate = (exact_matches / len(runs)) * 100
            
            # 格式化 LaTeX 行
            row = (
                f"{task} & "
                f"${mse_s['mean']:.2e} \\pm {mse_s['std']:.2e}$ & "
                f"${mse_s['median']:.2e}$ & "
                f"${mse_s['min']:.2e}$ & "
                f"{success_rate:.1f} & "
                f"${rt_s['mean']:.2f} \\pm {rt_s['std']:.2f}$ \\\\"
            )
            latex_lines.append(row)
            
        latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        
        # 保存 LaTeX 表格
        table_path = os.path.join(OUTPUT_DIR, f"table_atf_stats_{mode}.tex")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))
        print(f"  ✅ 表格已保存: {table_path}")

    # 生成可视化图表
    df_plot = pd.DataFrame(plot_data)
    
    # 1. MSE 箱线图 (Log Scale)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Task", y="MSE", hue="Mode", data=df_plot, showfliers=True, linewidth=1.5)
    plt.yscale('log')
    plt.title("MSE Distribution of AdaptoFlux (10 Runs per Task)", fontsize=14)
    plt.ylabel("MSE (Log Scale)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "atf_mse_distribution_boxplot.png"), dpi=300)
    plt.close()
    print("  📊 箱线图已保存: atf_mse_distribution_boxplot.png")

    # 2. 成功率柱状图
    # 重新计算成功率用于绘图
    success_rows = []
    for mode in modes:
        for task in tasks:
            runs = data[mode][task]
            exact = sum(1 for r in runs if r["adaptoflux"].get("exact_match"))
            success_rows.append({"Task": task, "Mode": mode, "SuccessRate": exact/len(runs)*100})
            
    df_success = pd.DataFrame(success_rows)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Task", y="SuccessRate", hue="Mode", data=df_success, palette="Set2")
    plt.ylim(0, 110)
    plt.title("Exact Match Success Rate (%) of AdaptoFlux", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "atf_success_rate_bars.png"), dpi=300)
    plt.close()
    print("  📊 柱状图已保存: atf_success_rate_bars.png")
    
    print("\n🎉 处理完成！所有结果已保存至 processed_results/ 目录。")

if __name__ == "__main__":
    try:
        data = load_data(os.path.join(INPUT_DIR, INPUT_FILE))
        process_and_visualize(data)
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件。请确保数据文件位于 {os.path.join(INPUT_DIR, INPUT_FILE)}")
    except KeyError as e:
        print(f"❌ 错误：JSON 结构不匹配，缺失键 {e}。请检查 JSON 内容。")