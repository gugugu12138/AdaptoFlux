import json

# 从外部 JSON 文件读取数据
with open("experiments/ExternalBaselines/results/baselines_best_of_10_with_time.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Helper: format MSE in scientific notation like 1.5e+02
def format_mse(mse):
    if mse == 0:
        return "0"
    # Use .1e to get one digit before decimal
    s = f"{mse:.1e}"
    base, exp_str = s.split('e')
    exp = int(exp_str)
    base_float = float(base)
    # Round base to avoid 1.0 -> keep as 1
    if abs(base_float - round(base_float)) < 1e-5:
        base_str = str(int(round(base_float)))
    else:
        base_str = base.rstrip('0').rstrip('.')
    if exp == 0:
        return base_str
    elif exp > 0:
        return f"{base_str}e+{exp:02d}" if exp < 10 else f"{base_str}e+{exp}"
    else:
        return f"{base_str}e-{abs(exp):02d}" if abs(exp) < 10 else f"{base_str}e-{abs(exp)}"

# Helper: format runtime to 1 decimal place
def format_time(t):
    return f"{t:.2f}"

# Order of tasks (preserve original order from your example)
tasks = [
    "E_mc2", "F_ma", "KE", "Coulomb",
    "IdealGas", "OhmsLaw", "Acceleration", "Power"
]

# Validate that all tasks exist in the loaded data
missing = [t for t in tasks if t not in data]
if missing:
    raise KeyError(f"Missing tasks in JSON: {missing}")

# ==============================
# 1. Generate performance table
# ==============================
performance_lines = []
for i, task in enumerate(tasks):
    if i > 0:
        performance_lines.append("\\midrule")
    performance_lines.append(f"\\multirow{{3}}{{*}}{{{task}}}")
    for method in ["gplearn", "pysr", "xgboost"]:
        info = data[task][method]
        exact = "\\textbf{Yes}" if info["exact_match"] else "No"
        mse_str = format_mse(info["mse"])
        time_str = format_time(info["runtime_sec"])
        method_name = "PySR" if method == "pysr" else method.capitalize()
        performance_lines.append(f"& {method_name} & {exact} & {mse_str} & {time_str} \\\\")

performance_table = r"""\begin{table}[H]
\centering
\caption{Baseline performance of gplearn, PySR, and XGBoost (best of 10 runs)}
\label{tab:baselines_gp_pysr}
\begin{tabular}{lcccc}
\toprule
Task & Method & Exact? & MSE & Runtime (s) \\
\midrule
""" + "\n".join(performance_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open("experiments/ExternalBaselines/results/baseline_performance.tex", "w", encoding="utf-8") as f:
    f.write(performance_table)

# ===================================
# 2. Generate symbolic expressions table (only gplearn and pysr)
# ===================================
expr_lines = []
for i, task in enumerate(tasks):
    if i > 0:
        expr_lines.append("\\midrule")
    expr_lines.append(f"\\multirow{{2}}{{*}}{{{task}}}")
    for method in ["gplearn", "pysr"]:
        info = data[task][method]
        # Escape underscores for LaTeX
        expr = info["expression"].replace("_", "\\_")
        method_name = "PySR" if method == "pysr" else "gplearn"
        expr_lines.append(f"& {method_name} & \\texttt{{{expr}}} \\\\")

expression_table = r"""\begin{table}[H]
\centering
\caption{Symbolic expressions recovered by gplearn and PySR}
\label{tab:symbolic_expressions}
\begin{tabular}{lcl}
\toprule
Task & Method & Expression \\
\midrule
""" + "\n".join(expr_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open("experiments/ExternalBaselines/results/symbolic_expressions.tex", "w", encoding="utf-8") as f:
    f.write(expression_table)

print("✅ Successfully generated:")
print("   - baseline_performance.tex")
print("   - symbolic_expressions.tex")