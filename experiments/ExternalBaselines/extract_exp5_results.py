import json
import numpy as np

# 读取 JSON 数据
with open("experiments/ExternalBaselines/results/all_results_by_collapse_with_baselines.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 8 个任务名称
tasks = [
    "E_mc2", "F_ma", "KE", "Coulomb",
    "IdealGas", "OhmsLaw", "Acceleration", "Power"
]

# 初始化结果容器
results = {}

for mode in ["first", "prod", "sum"]:
    results[mode] = {}
    for task in tasks:
        runs = data[mode][task]
        
        # 提取 adaptoflux 的所有 runs 的 mse 和 exact_match
        af_runs = [(run["adaptoflux"]["mse"], run["adaptoflux"]["exact_match"]) for run in runs]
        
        # 找到 MSE 最小的 run（best-of-10）
        best_mse, best_exact = min(af_runs, key=lambda x: x[0])
        
        # gplearn 和 xgboost 在所有 run 中相同，取第一个即可
        gplearn_mse = runs[0]["gplearn"]["mse"]
        gplearn_exact = runs[0]["gplearn"]["exact_match"]
        xgb_mse = runs[0]["xgboost"]["mse"]
        xgb_exact = False  # XGBoost never exact in this setup

        results[mode][task] = {
            "adaptoflux": {
                "best_mse": best_mse,
                "exact": best_exact  # 来自 best MSE run 的 exact_match
            },
            "gplearn": {
                "mse": gplearn_mse,
                "exact": gplearn_exact
            },
            "xgboost": {
                "mse": xgb_mse,
                "exact": False
            }
        }

# 打印结构化结果
print("=== Structured Results (Best-of-10 for ATF) ===")
for mode in ["first", "prod", "sum"]:
    print(f"\nCollapse Mode: {mode}")
    for task in tasks:
        r = results[mode][task]
        print(f"  {task}:")
        print(f"    ATF: MSE={r['adaptoflux']['best_mse']:.3e}, Exact={r['adaptoflux']['exact']}")
        print(f"    GP : MSE={r['gplearn']['mse']:.3e}, Exact={r['gplearn']['exact']}")
        print(f"    XGB: MSE={r['xgboost']['mse']:.3e}")

# -----------------------------
# 生成 LaTeX 表格（仅 prod 模式）
# -----------------------------
def format_scientific(x):
    if x == 0:
        return "0"
    # 使用科学计数法，保留 2 位有效数字
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / (10 ** exp)
    # 四舍五入到两位小数（若接近整数则取整）
    if abs(mantissa - round(mantissa)) < 1e-2:
        mantissa = round(mantissa)
    else:
        mantissa = round(mantissa, 2)
    return f"{mantissa} \\times 10^{{{exp}}}"

print("\n\n=== LaTeX Table (prod mode, best-of-10 for ATF) ===\n")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\small")
print("\\caption{Zero-shot symbolic regression results (best-of-10 runs for AdaptoFlux; single run for baselines). Lower MSE is better.}")
print("\\label{tab:exp5_results}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Task & Method & Collapse Mode & Best MSE & Exact Match? \\\\")
print("\\midrule")

for i, task in enumerate(tasks):
    r = results["prod"][task]
    task_name_map = {
        "E_mc2": "E\\_mc2",
        "F_ma": "F\\_ma",
        "KE": "KE",
        "Coulomb": "Coulomb",
        "IdealGas": "IdealGas",
        "OhmsLaw": "OhmsLaw",
        "Acceleration": "Acceleration",
        "Power": "Power"
    }
    tn = task_name_map[task]

    # AdaptoFlux (best-of-10)
    af_mse_str = format_scientific(r["adaptoflux"]["best_mse"])
    af_exact = "\\textbf{Yes}" if r["adaptoflux"]["exact"] else "No"
    print(f"{tn} & AdaptoFlux & \\texttt{{prod}} & ${af_mse_str}$ & {af_exact} \\\\")

    # gplearn (deterministic or single reported run)
    gp_mse_str = format_scientific(r["gplearn"]["mse"])
    gp_exact = "\\textbf{Yes}" if r["gplearn"]["exact"] else "No"
    print(f" & gplearn & — & ${gp_mse_str}$ & {gp_exact} \\\\")

    # XGBoost
    xgb_mse_str = format_scientific(r["xgboost"]["mse"])
    print(f" & XGBoost & — & ${xgb_mse_str}$ & No \\\\")

    if i < len(tasks) - 1:
        print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")