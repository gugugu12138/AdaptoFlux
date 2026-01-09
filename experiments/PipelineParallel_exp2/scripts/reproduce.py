
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction script for Pipeline Parallelism Experiment 2 (Dual-Version, English-only plots)
用于复现实验2：流水线并行性能分析（双版本，图表仅英文）

This script processes TWO benchmark datasets (e.g., with/without GIL) and generates:
- Graph metrics (shared)
- Speedup data per version
- English-only plots for publication

本脚本处理两组基准数据（例如启用/禁用 GIL），并生成：
- 图结构指标（共用）
- 每个版本的加速比数据
- 仅含英文的出版级图表

To reproduce both versions:
    python reproduce.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import platform

# ========================
# Use English-only fonts (avoid CJK warnings)
# ========================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs display correctly

# ========================
# Configuration
# ========================
MODEL_BASE_DIR = 'experiments/PipelineParallel_exp2/models'
MODEL_COUNT = 30

EXPERIMENTS = {
    'GIL': {
        'benchmark_csv': 'experiments/PipelineParallel_exp2/results/gil/benchmark_results.csv',
        'output_dir': 'experiments/PipelineParallel_exp2/results/gil'
    },
    'noGIL': {
        'benchmark_csv': 'experiments/PipelineParallel_exp2/results/nogil/benchmark_results.csv',
        'output_dir': 'experiments/PipelineParallel_exp2/results/nogil'
    }
}

for config in EXPERIMENTS.values():
    os.makedirs(os.path.join(config['output_dir'], 'images'), exist_ok=True)

# Shared layer mapping
actual_layers = {
    'model_1': 5,   'model_2': 5,   'model_3': 5,   'model_4': 9,
    'model_5': 2,   'model_6': 8,   'model_7': 5,   'model_8': 6,
    'model_9': 5,   'model_10': 5,  'model_11': 12, 'model_12': 15,
    'model_13': 11, 'model_14': 15, 'model_15': 20, 'model_16': 9,
    'model_17': 20, 'model_18': 20, 'model_19': 17, 'model_20': 20,
    'model_21': 25, 'model_22': 25, 'model_23': 23, 'model_24': 22,
    'model_25': 30, 'model_26': 15, 'model_27': 30, 'model_28': 27,
    'model_29': 35, 'model_30': 11
}

def assign_group(layers):
    if 1 <= layers <= 10:
        return 'Shallow'
    elif 11 <= layers <= 20:
        return 'Medium'
    elif 21 <= layers <= 35:
        return 'Deep'
    else:
        return 'Unknown'

# ----------------------------
# Graph analysis function
# ----------------------------
def analyze_graph_parallelism(graph_json):
    nodes = graph_json.get("nodes", [])
    edges = graph_json.get("edges", [])
    valid_nodes = {node["id"] for node in nodes if node["id"] not in ["root", "collapse"]}
    node_count = len(valid_nodes)

    edge_count = 0
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if (src == "root" or src in valid_nodes) and (tgt == "collapse" or tgt in valid_nodes):
            edge_count += 1

    full_graph = defaultdict(list)
    full_in_degree = defaultdict(int)
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        full_graph[src].append(tgt)
        full_in_degree[tgt] += 1

    # Critical path (topological DP)
    dist = defaultdict(int)
    queue = deque(["root"])
    dist["root"] = 1
    in_deg_copy = full_in_degree.copy()
    while queue:
        node = queue.popleft()
        for succ in full_graph[node]:
            new_dist = dist[node] + 1
            if new_dist > dist[succ]:
                dist[succ] = new_dist
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)
    critical_path_length = dist.get("collapse", 0)

    # Build forward graph for BFS level traversal
    graph = defaultdict(list)
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        graph[src].append(tgt)

    # BFS by level to compute widths
    level_nodes = ["root"]
    visited = set(["root"])
    widths = []

    while level_nodes:
        valid_in_level = [n for n in level_nodes if n not in ["root", "collapse"]]
        level_width = len(valid_in_level)
        if level_width > 0:
            widths.append(level_width)

        next_level = []
        for node in level_nodes:
            for child in graph[node]:
                if child not in visited:
                    visited.add(child)
                    next_level.append(child)
        level_nodes = next_level

    max_parallelism = max(widths) if widths else 0
    avg_width = round(np.mean(widths), 3) if widths else 0.0
    width_std = round(np.std(widths, ddof=0), 3) if widths else 0.0

    # Out-degree stats
    out_degree = defaultdict(int)
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_nodes and tgt in valid_nodes:
            out_degree[src] += 1

    total_out = sum(out_degree[node] for node in valid_nodes)
    avg_out_degree = round(total_out / len(valid_nodes), 3) if valid_nodes else 0.0
    branching_nodes = sum(1 for node in valid_nodes if out_degree[node] > 1)
    branching_factor = round(branching_nodes / len(valid_nodes), 3) if valid_nodes else 0.0

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "max_parallelism": max_parallelism,
        "avg_out_degree": avg_out_degree,
        "critical_path_length": critical_path_length,
        "branching_factor": branching_factor,
        "avg_width": avg_width,
        "width_std": width_std
    }

# ----------------------------
# Precompute graph metrics
# ----------------------------
print("[0/6] Precomputing shared graph metrics... / 正在预计算共用的图结构指标...")
graph_records = []
for i in range(1, MODEL_COUNT + 1):
    model_id = f"model_{i}"
    graph_file = os.path.join(MODEL_BASE_DIR, model_id, "graph.json")
    if not os.path.exists(graph_file):
        print(f"⚠️ Warning: {graph_file} not found. Skipping. / 警告：{graph_file} 不存在，跳过。")
        continue
    try:
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        metrics = analyze_graph_parallelism(graph_data)
        metrics["model_id"] = model_id
        graph_records.append(metrics)
    except Exception as e:
        print(f"❌ Error processing {model_id}: {e} / 处理 {model_id} 时出错：{e}")

graph_df = pd.DataFrame(graph_records)
print(f"[✓] Graph metrics computed for {len(graph_df)} models. / 共为 {len(graph_df)} 个模型计算了图指标。")

# ----------------------------
# Process each version
# ----------------------------
all_grouped_data = {}

for exp_name, config in EXPERIMENTS.items():
    print(f"\n{'='*50}")
    print(f"Processing {exp_name} version... / 正在处理 {exp_name} 版本...")
    print(f"{'='*50}")

    BENCHMARK_CSV = config['benchmark_csv']
    OUTPUT_DIR = config['output_dir']
    IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')

    df = pd.read_csv(BENCHMARK_CSV)
    df['model_id'] = df['model_path'].str.extract(r'(model_\d+)')
    if df['model_id'].isnull().any():
        raise ValueError(f"Invalid model_path in {exp_name} data.")

    unknown_models = set(df['model_id'].unique()) - set(actual_layers.keys())
    if unknown_models:
        raise ValueError(f"Unknown models in {exp_name}: {unknown_models}")

    df['actual_layers'] = df['model_id'].map(actual_layers)
    df['group'] = df['actual_layers'].apply(assign_group)

    baseline = df[df['num_cores'] == 1][['model_id', 'avg_latency_ms']].set_index('model_id')
    baseline.rename(columns={'avg_latency_ms': 'latency_1'}, inplace=True)
    df = df.merge(baseline, on='model_id', how='left')
    df['speedup'] = df['latency_1'] / df['avg_latency_ms']

    df = df.merge(graph_df, on='model_id', how='left')

    graph_output_path = os.path.join(OUTPUT_DIR, 'graph_metrics.csv')
    graph_df.to_csv(graph_output_path, index=False)
    print(f"[✓] Graph metrics saved to {graph_output_path} / 图指标已保存至 {graph_output_path}")

    grouped = df.groupby(['group', 'num_cores'])['speedup'].mean().reset_index()
    all_grouped_data[exp_name] = grouped

    speedup_output_path = os.path.join(IMAGE_DIR, 'task_parallel_speedup_data.csv')
    grouped.to_csv(speedup_output_path, index=False)
    print(f"[✓] Speedup data saved to {speedup_output_path} / 加速比数据已保存至 {speedup_output_path}")

    # --- PLOT (ENGLISH ONLY) ---
    plt.figure(figsize=(10, 6))
    colors = {'Shallow': 'tab:blue', 'Medium': 'tab:orange', 'Deep': 'tab:green'}

    for group in ['Shallow', 'Medium', 'Deep']:
        subset = grouped[grouped['group'] == group]
        if not subset.empty:
            plt.plot(subset['num_cores'], subset['speedup'],
                     marker='o', label=f"{group} ({exp_name})", color=colors[group])

    ideal_cores = np.array([1, 2, 4, 8, 16])
    plt.plot(ideal_cores, ideal_cores, 'k--', label='Ideal Linear')

    plt.xscale('log', base=2)
    plt.xticks(ideal_cores, ideal_cores)
    plt.xlabel('Number of CPU Cores')
    plt.ylabel('Speedup (T₁ / Tₖ)')
    plt.title(f'Speedup vs. CPU Cores by Model Depth Group ({exp_name})')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    plot_path = os.path.join(IMAGE_DIR, f'task_parallel_speedup_cpu_{exp_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Plot saved to {plot_path} / 图表已保存至 {plot_path}")

# ----------------------------
# Comparison plot (ENGLISH ONLY)
# ----------------------------
print("\n[5/5] Generating COMPARISON plot... / 正在生成 GIL vs noGIL 对比图...")
plt.figure(figsize=(12, 7))
colors = {'Shallow': 'tab:blue', 'Medium': 'tab:orange', 'Deep': 'tab:green'}
linestyles = {'GIL': '-', 'noGIL': '--'}

for exp_name, grouped in all_grouped_data.items():
    for group in ['Shallow', 'Medium', 'Deep']:
        subset = grouped[grouped['group'] == group]
        if not subset.empty:
            plt.plot(subset['num_cores'], subset['speedup'],
                     marker='o', linestyle=linestyles[exp_name],
                     color=colors[group],
                     label=f"{group} ({exp_name})")

ideal_cores = np.array([1, 2, 4, 8, 16])
plt.plot(ideal_cores, ideal_cores, 'k--', label='Ideal Linear')

plt.xscale('log', base=2)
plt.xticks(ideal_cores, ideal_cores)
plt.xlabel('Number of CPU Cores')
plt.ylabel('Speedup (T₁ / Tₖ)')
plt.title('GIL vs noGIL: Speedup Comparison by Model Depth')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

comparison_plot_path = 'experiments/PipelineParallel_exp2/results/images/speedup_comparison_GIL_vs_noGIL.png'
os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
plt.savefig(comparison_plot_path, dpi=300)
plt.close()
print(f"[✓] Comparison plot saved to {comparison_plot_path} / 对比图已保存至 {comparison_plot_path}")

print("\n[6/6] Performing node count vs speedup analysis... / 正在进行节点数与加速比关联分析...")

# Combine all data into one DataFrame for cross-version analysis
all_dfs = []
for exp_name, config in EXPERIMENTS.items():
    df = pd.read_csv(config['benchmark_csv'])
    df['model_id'] = df['model_path'].str.extract(r'(model_\d+)')
    df['actual_layers'] = df['model_id'].map(actual_layers)
    df['group'] = df['actual_layers'].apply(assign_group)
    
    # Merge graph metrics (include new width metrics)
    graph_cols_to_merge = [
        'model_id',
        'node_count',
        'max_parallelism',
        'critical_path_length',
        'avg_width',
        'width_std',
        'branching_factor',      # <-- added
        'avg_out_degree'         # <-- added
    ]
    df = df.merge(graph_df[graph_cols_to_merge], on='model_id', how='left')

    # Compute speedup
    baseline = df[df['num_cores'] == 1][['model_id', 'avg_latency_ms']].set_index('model_id')
    baseline.rename(columns={'avg_latency_ms': 'latency_1'}, inplace=True)
    df = df.merge(baseline, on='model_id', how='left')
    df['speedup'] = df['latency_1'] / df['avg_latency_ms']
    df['version'] = exp_name
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Save combined analysis data
combined_output = 'experiments/PipelineParallel_exp2/results/images/combined_analysis.csv'
combined_df.to_csv(combined_output, index=False)
print(f"[✓] Combined analysis data saved to {combined_output} / 联合分析数据已保存至 {combined_output}")

# ----------------------------
# 1. Group by node count bins
# ----------------------------
def assign_node_group(n):
    if pd.isna(n):
        return 'Unknown'
    if n <= 93:
        return 'Small (≤93)'
    elif n <= 190:
        return 'Medium (94–190)'
    else:
        return 'Large (≥191)'

combined_df['node_group'] = combined_df['node_count'].apply(assign_node_group)

# Aggregate mean speedup by node_group, version, and num_cores
node_speedup = combined_df.groupby(['node_group', 'version', 'num_cores'])['speedup'].mean().reset_index()

# Save this aggregation
node_speedup_path = 'experiments/PipelineParallel_exp2/results/images/node_group_speedup.csv'
node_speedup.to_csv(node_speedup_path, index=False)
print(f"[✓] Node-group speedup data saved to {node_speedup_path}")

# ----------------------------
# 2. Plot: Node Count vs Speedup (scatter) — noGIL only
# ----------------------------
plt.figure(figsize=(10, 6))
# Only plot noGIL
versions = ['noGIL']  # <<== 修改：只保留 noGIL
colors = {'noGIL': 'blue'}  # <<== 可简化，但保留结构
markers = {'Shallow': 'o', 'Medium': '^', 'Deep': 's'}

for version in versions:
    subset = combined_df[combined_df['version'] == version]
    for depth in ['Shallow', 'Medium', 'Deep']:
        subsub = subset[subset['group'] == depth]
        if not subsub.empty:
            plt.scatter(
                subsub['node_count'],
                subsub['speedup'],
                color=colors[version],
                marker=markers[depth],
                alpha=0.7,
                label=depth  # <<== 修改：不再显示 "(noGIL)"，只保留深度标签
            )

plt.xlabel('Number of Nodes in Graph')
plt.ylabel('Speedup (T₁ / Tₖ)')
plt.title('Speedup vs. Graph Node Count (noGIL only, colored by model depth)')
plt.legend(title='Model Depth')
plt.grid(True, ls="--", linewidth=0.5)
plt.tight_layout()

scatter_plot_path = 'experiments/PipelineParallel_exp2/results/images/speedup_vs_node_count.png'
plt.savefig(scatter_plot_path, dpi=300)
plt.close()
print(f"[✓] Scatter plot saved to {scatter_plot_path} / 散点图已保存至 {scatter_plot_path}")

# ----------------------------
# 3. Plot: Node Group Speedup Curves
# ----------------------------
plt.figure(figsize=(10, 6))
linestyles = {'GIL': '-', 'noGIL': '--'}
node_groups = ['Small (≤93)', 'Medium (94–190)', 'Large (≥191)']

for version in versions:
    for group in node_groups:
        subset = node_speedup[(node_speedup['version'] == version) & (node_speedup['node_group'] == group)]
        if not subset.empty:
            plt.plot(
                subset['num_cores'],
                subset['speedup'],
                marker='o',
                linestyle=linestyles[version],
                label=f"{group} ({version})"
            )

plt.xscale('log', base=2)
plt.xticks([1,2,4,8,16], [1,2,4,8,16])
plt.xlabel('Number of CPU Cores')
plt.ylabel('Average Speedup (T₁ / Tₖ)')
plt.title('Speedup by Graph Node Size and GIL Setting')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

group_plot_path = 'experiments/PipelineParallel_exp2/results/images/speedup_by_node_group.png'
plt.savefig(group_plot_path, dpi=300)
plt.close()
print(f"[✓] Node-group speedup plot saved to {group_plot_path}")

# ----------------------------
# 4. Plot: Graph metrics vs. Parallel Efficiency (noGIL only)
# ----------------------------
print("\n[7/7] Plotting graph metrics vs. parallel efficiency (noGIL only)... / 正在绘制图指标与并行效率关系图（仅 noGIL）...")

# Compute efficiency: speedup / num_cores
combined_df['efficiency'] = combined_df['speedup'] / combined_df['num_cores']

# Filter to noGIL only
df_plot = combined_df[combined_df['version'] == 'noGIL'].copy()

# Ensure required columns exist
required_cols = ['max_parallelism', 'avg_width', 'width_std', 'efficiency', 'group']
for col in required_cols:
    if col not in df_plot.columns:
        raise KeyError(f"Column '{col}' not found in combined_df.")

# Prepare plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
metrics = [
    ('max_parallelism', 'Max Parallelism (Width)'),
    ('avg_width', 'Average Width'),
    ('width_std', 'Width Standard Deviation')
]
# Use only one color since only noGIL; differentiate by depth group with markers
colors = {'Shallow': 'tab:blue', 'Medium': 'tab:orange', 'Deep': 'tab:green'}
markers = {'Shallow': 'o', 'Medium': '^', 'Deep': 's'}

for ax, (metric, label) in zip(axes, metrics):
    for depth in ['Shallow', 'Medium', 'Deep']:
        subset = df_plot[df_plot['group'] == depth]
        if not subset.empty:
            ax.scatter(
                subset[metric],
                subset['efficiency'],
                color=colors[depth],
                marker=markers[depth],
                alpha=0.7,
                label=f"{depth}"
            )
    ax.set_ylabel('Parallel Efficiency (Speedup / Cores)')
    ax.set_xlabel(label)
    ax.grid(True, ls="--", linewidth=0.5)
    ax.set_ylim(0, 1.1)  # Efficiency cannot exceed 1
    ax.legend()

plt.tight_layout()

efficiency_plot_path = 'experiments/PipelineParallel_exp2/results/images/efficiency_vs_graph_metrics.png'
os.makedirs(os.path.dirname(efficiency_plot_path), exist_ok=True)
plt.savefig(efficiency_plot_path, dpi=300)
plt.close()
print(f"[✓] Efficiency vs. graph metrics plot (noGIL only) saved to {efficiency_plot_path}")

# ----------------------------
# 5. Generate LaTeX table for node-grouped performance (noGIL only)
# ----------------------------
def generate_latex_performance_table(df, output_path):
    """
    Generate a LaTeX table string for node-grouped performance stats (noGIL only).
    Saves it to a .txt file containing pure LaTeX code.
    """
    # Filter only noGIL data
    df_nogil = df[df['version'] == 'noGIL'].copy()
    
    # Apply node grouping
    df_nogil['node_group'] = df_nogil['node_count'].apply(assign_node_group)
    
    # Compute efficiency and std per group & core count
    stats = df_nogil.groupby(['node_group', 'num_cores']).agg(
        mean_speedup=('speedup', 'mean'),
        mean_efficiency=('efficiency', 'mean'),
        std_efficiency=('efficiency', 'std')
    ).reset_index()
    
    # Round values
    stats['mean_speedup'] = stats['mean_speedup'].round(2)
    stats['mean_efficiency'] = (stats['mean_efficiency'] * 100).round(1)  # to percentage
    stats['std_efficiency'] = (stats['std_efficiency'] * 100).round(1)   # to percentage

    # Define group order and labels
    group_order = {
        'Small (≤93)': '小规模 (≤93)',
        'Medium (94–190)': '中规模 (94–190)',
        'Large (≥191)': '大规模 (≥191)'
    }
    core_order = [1, 4, 8, 16]
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append(r"% >>> 实测表：按节点数量分组的性能统计 <<<")
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    latex_lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\textbf{节点规模区间} & \textbf{CPU 核心数} & \textbf{平均加速比} & \textbf{平均效率} & \textbf{效率标准差} \\")
    latex_lines.append(r"\hline")

    first_group = True
    for group_key in ['Small (≤93)', 'Medium (94–190)', 'Large (≥191)']:
        group_label = group_order[group_key]
        group_data = stats[stats['node_group'] == group_key]
        
        if group_data.empty:
            continue
            
        # Sort by core count
        group_data = group_data.set_index('num_cores').reindex(core_order).reset_index()
        
        for i, row in group_data.iterrows():
            if pd.isna(row['mean_speedup']):
                continue
                
            core = int(row['num_cores'])
            speedup = f"{row['mean_speedup']:.2f}×"
            eff = f"{row['mean_efficiency']}\\%"
            std = f"{row['std_efficiency']}\\%" if not pd.isna(row['std_efficiency']) and core != 1 else "—"
            
            if i == 0:
                # First row of group: use \multirow
                multirow = r"\multirow{4}{*}{" + group_label + r"}"
                line = f"{multirow} & {core}核 & {speedup} & {eff} & {std} \\\\"
            else:
                line = f" & {core}核 & {speedup} & {eff} & {std} \\\\"
                
            latex_lines.append(line)
        
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{不同节点规模模型在多核 CPU 下的平均性能表现（基于每组10个模型统计，仅 noGIL 有效数据）}")
    latex_lines.append(r"\label{tab:task_parallel_by_node_count}")
    latex_lines.append(r"\end{table}")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"[✓] LaTeX performance table saved to {output_path} / LaTeX 表格已保存至 {output_path}")

# Call the function
latex_table_path = 'experiments/PipelineParallel_exp2/results/images/node_performance_table.tex.txt'
generate_latex_performance_table(combined_df, latex_table_path)

# ----------------------------
# 5. NEW: Plot branching factor and avg out-degree vs. parallel efficiency (noGIL only)
# ----------------------------
print("\n[8/8] Plotting branching factor and average out-degree vs. efficiency (noGIL only)...")

df_plot = combined_df[combined_df['version'] == 'noGIL'].copy()

# Ensure metrics exist
assert 'branching_factor' in df_plot.columns, "branching_factor missing!"
assert 'avg_out_degree' in df_plot.columns, "avg_out_degree missing!"

fig, axes = plt.subplots(2, 1, figsize=(10, 9))
metrics_info = [
    ('branching_factor', 'Branching Factor (Fraction of Nodes with Out-Degree ≥ 2)'),
    ('avg_out_degree', 'Average Out-Degree')
]

colors = {'Shallow': 'tab:blue', 'Medium': 'tab:orange', 'Deep': 'tab:green'}
markers = {'Shallow': 'o', 'Medium': '^', 'Deep': 's'}

for ax, (col, xlabel) in zip(axes, metrics_info):
    for depth in ['Shallow', 'Medium', 'Deep']:
        subset = df_plot[df_plot['group'] == depth]
        if not subset.empty:
            ax.scatter(
                subset[col],
                subset['efficiency'],
                color=colors[depth],
                marker=markers[depth],
                alpha=0.7,
                label=f"{depth}"
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Parallel Efficiency (Speedup / Cores)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, ls="--", linewidth=0.5)
    ax.legend()

plt.tight_layout()
new_plot_path = 'experiments/PipelineParallel_exp2/results/images/efficiency_vs_branching_and_outdegree.png'
os.makedirs(os.path.dirname(new_plot_path), exist_ok=True)
plt.savefig(new_plot_path, dpi=300)
plt.close()
print(f"[✓] New plot saved: {new_plot_path}")

print("\n🎉 Dual-version reproduction complete! All results are ready.\n"
      "🎉 双版本复现完成！所有结果已生成，可用于论文分析与对比。")