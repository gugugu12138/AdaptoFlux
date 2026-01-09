import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
DATA_FILE = "experiments\\LayerGrowTrainer_exp1\\experiment_results_summary.csv"
OUTPUT_DIR = "experiments\\LayerGrowTrainer_exp1\\images"
TABLE_OUTPUT_FILE = "experiments\\LayerGrowTrainer_exp1\\convergence_stats_table.tex"
REPORT_FILE = "experiments\\LayerGrowTrainer_exp1\\summary_report.txt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"Data loaded successfully, total {len(df)} records.")

# --- 2. Compute Statistics ---
print("Computing statistics...")

stats = df.groupby('max_layers').agg(
    avg_best_acc=('best_model_accuracy', 'mean'),
    std_best_acc=('best_model_accuracy', 'std'),
    avg_layers_added=('layers_added', 'mean'),
    avg_total_attempts=('total_growth_attempts', 'mean'),
    failure_rate=('total_growth_attempts', lambda x: (x == 2500).sum() / len(x) * 100)
).reset_index()

# --- 3. Generate Figure: scaling_behavior_split.png (Three Subplots) ---
print("Generating figure: scaling_behavior_split.png...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Subplot 1: Average Best Accuracy ---
ax1.plot(stats['max_layers'], stats['avg_best_acc'], 'o-', color='blue', linewidth=2, markersize=8, label='Avg Best Accuracy')
ax1.set_ylabel('Average Best Accuracy', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
ax1.axvline(x=20, color='blue', linestyle='--', linewidth=2, label='Peak Performance (max_layers=20)')
ax1.axvline(x=35, color='red', linestyle='--', linewidth=2, label='Failure Boundary (max_layers=35)')
ax1.legend(loc='upper left', fontsize=10)

# --- Subplot 2: Average Total Attempts ---
ax2.plot(stats['max_layers'], stats['avg_total_attempts'], 's--', color='orange', linewidth=2, markersize=8, label='Avg Total Attempts')
ax2.set_ylabel('Average Total Attempts', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)

# --- Subplot 3: Failure Rate ---
ax3.plot(stats['max_layers'], stats['failure_rate'], '^-', color='red', linewidth=2, markersize=8, label='Failure Rate (%)')
ax3.set_ylabel('Failure Rate (%)', fontsize=12)
ax3.set_xlabel('max_layers', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(fontsize=10)

# Adjust layout and add title
plt.suptitle('Performance, Cost, and Failure Rate vs. max_layers', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scaling_behavior_split.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Figure 'scaling_behavior_split.png' generated.")
# --- 4. Generate Figure: run_comparison_20.png ---
print("Generating figure: run_comparison_20.png...")

# Filter data for max_layers=20
df_20 = df[df['max_layers'] == 20].copy()
df_20['run_label'] = 'Run ' + df_20['run'].astype(str)

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(df_20['total_growth_attempts'], df_20['best_model_accuracy'], 
            s=150, alpha=0.8, c='lightblue', edgecolors='black', linewidth=1, label='Runs')

# Highlight Run 10 (best performance)
run_10 = df_20[df_20['run'] == 10].iloc[0]
plt.scatter(run_10['total_growth_attempts'], run_10['best_model_accuracy'], 
            s=300, c='red', marker='*', edgecolors='black', linewidth=2, 
            label=f'Best Run 10 ({run_10["best_model_accuracy"]:.3f})')

# Highlight Run 2 (lowest cost, mediocre performance)
run_2 = df_20[df_20['run'] == 2].iloc[0]
plt.scatter(run_2['total_growth_attempts'], run_2['best_model_accuracy'], 
            s=300, c='green', marker='s', edgecolors='black', linewidth=2, 
            label=f'Low-Cost Run 2 ({run_2["best_model_accuracy"]:.3f})')

# Annotate each point
for i, row in df_20.iterrows():
    plt.text(row['total_growth_attempts'] + 50, row['best_model_accuracy'], 
             row['run_label'], fontsize=9)

# Labels and title
plt.xlabel('Total Growth Attempts', fontsize=14)
plt.ylabel('Best Model Accuracy', fontsize=14)
plt.title('Performance vs. Cost for 10 Runs (max_layers=20)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(OUTPUT_DIR, 'run_comparison_20.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Figure 'run_comparison_20.png' generated.")

# --- 5. Generate LaTeX Table ---
print("Generating LaTeX table...")

stats_formatted = stats.copy()
stats_formatted['avg_best_acc'] = stats_formatted['avg_best_acc'].round(3)
stats_formatted['std_best_acc'] = stats_formatted['std_best_acc'].round(3)
stats_formatted['avg_layers_added'] = stats_formatted['avg_layers_added'].round(1)
stats_formatted['avg_total_attempts'] = stats_formatted['avg_total_attempts'].round(1)
stats_formatted['failure_rate'] = stats_formatted['failure_rate'].round(0).astype(int)

latex_table = r"""\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\texttt{max\_layers} & Avg Best Acc & Std Dev & Avg Layers Added & Avg Total Attempts & Failure Rate \\
\hline
"""
for _, row in stats_formatted.iterrows():
    latex_table += f"{int(row['max_layers'])} & {row['avg_best_acc']} & {row['std_best_acc']} & {row['avg_layers_added']} & {row['avg_total_attempts']} & {row['failure_rate']}\% \\\\\n"
latex_table += r"""\hline
\end{tabular}
\caption{Statistical results under different max\_layers settings. Failure rate is the percentage of runs where \texttt{total\_growth\_attempts=2500}.}
\label{tab:convergence_stats}
\end{table}
"""

with open(TABLE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(latex_table)

print(f"LaTeX table saved to: {TABLE_OUTPUT_FILE}")

# --- 6. Generate Analysis Report ---
print("Generating analysis report...")

report = f"""AdaptoFlux Experiment 1 Analysis Report
============================

Data Source: {DATA_FILE}
Generated: {pd.Timestamp.now()}

Key Findings:
---------
1. Performance Saturation: Peak average best accuracy ({stats.loc[stats['max_layers']==20, 'avg_best_acc'].iloc[0]:.3f}) occurs at `max_layers=20`.
2. Strategy Failure Boundary: At `max_layers=35`, failure rate surges to {stats.loc[stats['max_layers']==35, 'failure_rate'].iloc[0]:.0f}%, indicating search strategy collapse.
3. Optimal Trade-off: `max_layers=20` offers the best balance. E.g., Run 10 achieved {run_10['best_model_accuracy']:.3f} accuracy with {run_10['total_growth_attempts']} attempts.
4. Path Dependency: Among 10 runs at `max_layers=20`, best accuracy ranged from {df_20['best_model_accuracy'].min():.3f} to {df_20['best_model_accuracy'].max():.3f}, showing high sensitivity to early decisions.

Detailed Statistics:
---------
{stats_formatted.to_string(index=False)}

"""

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Analysis report saved to: {REPORT_FILE}")
print("\n✅ All charts and files generated successfully!")