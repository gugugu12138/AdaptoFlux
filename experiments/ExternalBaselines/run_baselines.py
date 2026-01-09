# experiments/ExternalBaselines/run_baselines.py
import os
import json
import numpy as np
import time
from .tasks.feynman_tasks import get_task, TASK_REGISTRY
from .baselines.gplearn_runner import run_gplearn
from .baselines.xgboost_runner import run_xgboost
from .adaptoflux_runner import run_adaptoflux

NUM_REPEATS = 10
COLLAPSE_MODES = ["first", "mean", "sum", "prod"]  # 四种坍缩策略

def main():
    os.makedirs("experiments/ExternalBaselines/results", exist_ok=True)
    
    tasks = list(TASK_REGISTRY.keys())
    all_results = {mode: {task: [] for task in tasks} for mode in COLLAPSE_MODES}
    methods_path = "experiments/ExternalBaselines/methods/methods_feynman.py"
    
    # === 对每种坍缩策略运行实验 ===
    for collapse_mode in COLLAPSE_MODES:
        print(f"\n{'='*60}")
        print(f"🧪 Running collapse mode: {collapse_mode}")
        print(f"{'='*60}")
        
        for repeat in range(NUM_REPEATS):
            print(f"\n🔁 Repeat {repeat + 1}/{NUM_REPEATS}")
            
            for task_name in tasks:
                print(f"  → Task: {task_name}")
                X, y, expr_str, var_names = get_task(task_name, n_samples=50, seed=42 + repeat)
                save_dir = f"experiments/ExternalBaselines/saved_models/{task_name}/{collapse_mode}_run_{repeat}"
                
                # ⏱️ 记录运行时间
                start_time = time.time()
                af_res = run_adaptoflux(
                    X, y, methods_path,
                    random_state=42 + repeat,
                    collapse_mode=collapse_mode,
                    save_path=save_dir
                )
                af_res["runtime_sec"] = time.time() - start_time
                af_res["collapse_mode"] = collapse_mode
                
                print(f"    [AdaptoFlux] Exact: {af_res['exact_match']}, MSE: {af_res['mse']:.2e}, "
                      f"Time: {af_res['runtime_sec']:.2f}s")
                
                # 基线只在 mean 模式第 0 次运行（避免重复）
                gp_res = {"exact_match": False, "mse": 0.0}
                xb_res = {"exact_match": False, "mse": 0.0}
                if collapse_mode == "mean" and repeat == 0:
                    gp_res = run_gplearn(X, y, random_state=42 + repeat)
                    xb_res = run_xgboost(X, y, random_state=42 + repeat)
                
                all_results[collapse_mode][task_name].append({
                    "adaptoflux": af_res,
                    "gplearn": gp_res,
                    "xgboost": xb_res,
                    "ground_truth": expr_str
                })
    
    # === 汇总：对每种策略取 best-of-10 ===
    summaries = {}
    for mode in COLLAPSE_MODES:
        summaries[mode] = {}
        for task in tasks:
            repeats = all_results[mode][task]
            best_run = min(repeats, key=lambda r: (
                0 if r['adaptoflux']['exact_match'] else 1,
                r['adaptoflux']['mse'],
                r['adaptoflux']['runtime_sec']
            ))
            summaries[mode][task] = best_run

    # === 保存结果 ===
    with open("experiments/ExternalBaselines/results/all_results_by_collapse.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("experiments/ExternalBaselines/results/summaries_by_collapse.json", "w") as f:
        json.dump(summaries, f, indent=2)
    
    # === 生成 LaTeX 表格（每种策略一张表）===
    for mode in COLLAPSE_MODES:
        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Parameter-free symbolic regression with collapse=" + mode + r" (best of 10 runs)}",
            r"\label{tab:baselines_" + mode + r"}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Task & Exact? & MSE & Runtime (s) & Method Pool & Notes \\",
            r"\midrule"
        ]
        for task in tasks:
            af = summaries[mode][task]["adaptoflux"]
            row = (
                task + " & " +
                ("\\textbf{Yes}" if af['exact_match'] else "No") + " & " +
                f"{af['mse']:.0e}" + " & " +
                f"{af['runtime_sec']:.1f}" + " & " +
                str(af['method_pool_size']) + " & Supports side-effect actions \\\\"
            )
            latex_lines.append(row)
            latex_lines.append(r"\midrule")
        latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        
        with open(f"experiments/ExternalBaselines/results/comparison_table_{mode}.tex", "w") as f:
            f.write("\n".join(latex_lines))
    
    # === 打印最终结果 ===
    print("\n" + "="*70)
    print("🎯 FINAL BEST RESULTS BY COLLAPSE MODE:")
    print("="*70)
    for mode in COLLAPSE_MODES:
        runtimes = []
        print(f"\n--- {mode.upper()} ---")
        for task in tasks:
            af = summaries[mode][task]["adaptoflux"]
            print(f"  {task:15}: Exact={af['exact_match']}, MSE={af['mse']:.2e}, Time={af['runtime_sec']:.2f}s")
            runtimes.append(af['runtime_sec'])
        print(f"  → Avg Time: {np.mean(runtimes):.2f}s")
    
    # === 可视化 ===
    try:
        from .utils_viz import visualize_graph_hierarchy
        print("\n" + "="*60)
        print("🖼️  Saving best model visualizations...")
        print("="*60)
        for mode in COLLAPSE_MODES:
            for task in tasks:
                best_run = summaries[mode][task]
                save_path = best_run["adaptoflux"]["save_path"]
                if save_path and os.path.exists(save_path):
                    graph_path = os.path.join(save_path, "combined_trainer_temp/final/graph.json")
                    output_img = f"experiments/ExternalBaselines/results/best_{task}_collapse_{mode}.png"
                    try:
                        visualize_graph_hierarchy(graph_path, output_img, root="root")
                        print(f"  → Saved {task} ({mode}) to {output_img}")
                    except Exception as e:
                        print(f"  ⚠️  Failed to visualize {task} ({mode}): {e}")
    except Exception as e:
        print(f"\n⚠️  Visualization skipped: {e}")

    print("\n✅ All results saved to: experiments/ExternalBaselines/results/")

if __name__ == "__main__":
    main()