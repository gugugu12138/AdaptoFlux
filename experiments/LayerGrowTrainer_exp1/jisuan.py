import pandas as pd

# 读取你的数据（假设已保存为 'data.csv'）
df = pd.read_csv('experiments\LayerGrowTrainer_exp1\experiment_results_summary.csv')  # 或直接粘贴数据
# df = pd.read_csv(pd.compat.StringIO("""paste_your_csv_here"""))

# 聚合统计
summary = df.groupby('max_layers').agg(
    avg_accuracy=('best_model_accuracy', 'mean'),
    std_accuracy=('best_model_accuracy', 'std'),
    avg_best_layers=('best_model_layers', 'mean'),  # ✅ 改为“最好模型层数”
    avg_candidate_attempts=('total_candidate_attempts', 'mean')
).round(3)

# 填充缺失 std 为 0（单条数据）
summary['std_accuracy'] = summary['std_accuracy'].fillna(0)

# 生成 LaTeX 表格行
print("max\_layers & 平均准确率 & 标准差 & 最优模型平均实际层数 & 平均尝试次数 \\\\")
print("\\hline")
for ml in sorted(summary.index):
    row = summary.loc[ml]
    acc_mean = row['avg_accuracy']
    acc_std = row['std_accuracy']
    layers = row['avg_best_layers']
    attempts = row['avg_candidate_attempts']
    print(f"{ml} & {acc_mean:.3f} ± {acc_std:.3f} & {acc_std:.3f} & {layers:.1f} & {attempts:.1f} \\\\")