# experiments/ExternalBaselines/run_baselines_parallel.py
import os
import json
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .tasks.feynman_tasks import get_task, TASK_REGISTRY
from .baselines.gplearn_runner import run_gplearn
from .baselines.xgboost_runner import run_xgboost
from .baselines.pysr_runner import run_pysr  # ← 新增

NUM_REPEATS = 10
COLLAPSE_MODES = ["first", "sum", "prod"]


def setup_worker():
    """限制子进程中底层库的线程数，避免资源争抢"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_single_repeat(task_name: str, collapse_mode: str, repeat: int):
    """
    运行单个任务、单个 collapse_mode、单次 repeat
    """
    setup_worker()  # 确保子进程线程数限制
    
    from .adaptoflux_runner import run_adaptoflux
    from .tasks.feynman_tasks import get_task

    X, y, expr_str, var_names = get_task(task_name, n_samples=50, seed=42 + repeat)
    save_dir = f"experiments/ExternalBaselines/saved_models/{task_name}/{collapse_mode}_run_{repeat}"
    
    start_time = time.time()
    af_res = run_adaptoflux(
        X, y, "experiments/ExternalBaselines/methods/methods_feynman.py",
        random_state=42 + repeat,
        collapse_mode=collapse_mode,
        save_path=save_dir
    )
    af_res["runtime_sec"] = time.time() - start_time
    af_res["repeat"] = repeat
    af_res["collapse_mode"] = collapse_mode
    af_res["task_name"] = task_name
    
    return {
        "adaptoflux": af_res,
        "ground_truth": expr_str
    }


def run_baselines_best_of_10():
    """
    为每个任务运行 gplearn 和 xgboost 共 10 次，取 best-of-10。
    """
    print(f"\n[PID {os.getpid()}] 🧪 Running Baselines (gplearn & XGBoost) - Best of 10...")
    baseline_results = {}
    tasks = list(TASK_REGISTRY.keys())
    
    for task_name in tasks:
        print(f"  - Running {task_name} (10 repeats)...")
        best_gp = {"mse": float('inf')}
        best_xb = {"mse": float('inf')}
        best_ps = {"mse": float('inf')}  # ← 新增
        
        X, y, expr_str, _ = get_task(task_name, n_samples=50, seed=42)
        for repeat in range(NUM_REPEATS):
            gp_res = run_gplearn(X, y, random_state=42 + repeat)
            xb_res = run_xgboost(X, y, random_state=42 + repeat)
            ps_res = run_pysr(X, y, random_state=42 + repeat)  # ← 新增
            
            if gp_res["mse"] < best_gp["mse"]:
                best_gp = gp_res
            if xb_res["mse"] < best_xb["mse"]:
                best_xb = xb_res
            if ps_res["mse"] < best_ps["mse"]:
                best_ps = ps_res  # ← 新增
                
        baseline_results[task_name] = {
            "gplearn": best_gp,
            "xgboost": best_xb,
            "pysr": best_ps,
            "ground_truth": expr_str
        }
    
    return baseline_results


def run_single_collapse_mode(collapse_mode):
    """
    在独立进程中运行单个 collapse_mode 的 AdaptoFlux 实验（不依赖 baseline）
    """
    print(f"\n[PID {os.getpid()}] 🧪 Starting collapse mode: {collapse_mode}")
    
    from .adaptoflux_runner import run_adaptoflux
    
    tasks = list(TASK_REGISTRY.keys())
    methods_path = "experiments/ExternalBaselines/methods/methods_feynman.py"
    results = {task: [] for task in tasks}
    
    for repeat in range(NUM_REPEATS):
        print(f"[{collapse_mode}] 🔁 Repeat {repeat + 1}/{NUM_REPEATS}")
        for task_name in tasks:
            X, y, expr_str, var_names = get_task(task_name, n_samples=50, seed=42 + repeat)
            save_dir = f"experiments/ExternalBaselines/saved_models/{task_name}/{collapse_mode}_run_{repeat}"
            
            start_time = time.time()
            af_res = run_adaptoflux(
                X, y, methods_path,
                random_state=42 + repeat,
                collapse_mode=collapse_mode,
                save_path=save_dir
            )
            af_res["runtime_sec"] = time.time() - start_time
            
            # 只保存 ATF 结果和 ground truth
            results[task_name].append({
                "adaptoflux": af_res,
                "ground_truth": expr_str
            })
    
    return collapse_mode, results


def aggregate_and_save(all_atf_results, baseline_results):
    """汇总 ATF 和 Baseline 结果，生成 JSON、LaTeX 表格和可视化"""
    from .utils_viz import visualize_graph_hierarchy

    # Step 1: 合并 baseline 到每个 ATF run 中
    all_results = {}
    for mode, task_runs in all_atf_results.items():
        all_results[mode] = {}
        for task, runs in task_runs.items():
            enriched_runs = []
            for run in runs:
                enriched_run = {
                    "adaptoflux": run["adaptoflux"],
                    "gplearn": baseline_results[task]["gplearn"],
                    "xgboost": baseline_results[task]["xgboost"],
                    "ground_truth": run["ground_truth"]
                }
                enriched_runs.append(enriched_run)
            all_results[mode][task] = enriched_runs

    # Step 2: 保存完整结果
    os.makedirs("experiments/ExternalBaselines/results", exist_ok=True)
    with open("experiments/ExternalBaselines/results/all_results_by_collapse_with_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Step 3: 生成 summaries（best-of-10）
    summaries = {}
    for mode, task_results in all_results.items():
        summaries[mode] = {}
        for task, repeats in task_results.items():
            best_run = min(repeats, key=lambda r: (
                0 if r['adaptoflux']['exact_match'] else 1,
                r['adaptoflux']['mse'],
                r['adaptoflux']['runtime_sec']
            ))
            summaries[mode][task] = best_run

    with open("experiments/ExternalBaselines/results/summaries_by_collapse.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # Step 4: 生成 LaTeX 表格（仅 ATF，含 Runtime）
    for mode in COLLAPSE_MODES:
        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Parameter-free symbolic regression with collapse=" + mode + r" (best of 10 runs)}",
            r"\label{tab:baselines_" + mode + r"}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Task & Exact? & MSE & Runtime (s) & Method Pool Size & Notes \\",
            r"\midrule"
        ]
        for task in TASK_REGISTRY.keys():
            af = summaries[mode][task]["adaptoflux"]
            row = (
                task + " & " +
                ("\\textbf{Yes}" if af['exact_match'] else "No") + " & " +
                f"{af['mse']:.0e}" + " & " +
                f"{af['runtime_sec']:.1f}" + " & " +
                str(af.get('method_pool_size', '?')) + " & Supports actions \\\\"
            )
            latex_lines.append(row)
            if task != list(TASK_REGISTRY.keys())[-1]:
                latex_lines.append(r"\midrule")
        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        with open(f"experiments/ExternalBaselines/results/comparison_table_{mode}.tex", "w") as f:
            f.write("\n".join(latex_lines))

    # Step 5: 打印平均运行时间
    print("\n" + "="*60)
    print("⏱️  AVERAGE RUNTIME BY COLLAPSE MODE (best runs):")
    print("="*60)
    for mode in COLLAPSE_MODES:
        runtimes = [summaries[mode][task]["adaptoflux"]["runtime_sec"] for task in TASK_REGISTRY.keys()]
        avg_time = np.mean(runtimes)
        print(f"{mode:6} : {avg_time:.2f} seconds")

    # Step 6: 可视化（可选）
    print("\n🖼️  Generating visualizations...")
    for mode in COLLAPSE_MODES:
        for task in TASK_REGISTRY.keys():
            best_run = summaries[mode][task]
            save_path = best_run["adaptoflux"]["save_path"]
            if save_path and os.path.exists(save_path):
                graph_path = os.path.join(save_path, "combined_trainer_temp/final/graph.json")
                output_img = f"experiments/ExternalBaselines/results/best_{task}_collapse_{mode}.png"
                try:
                    visualize_graph_hierarchy(graph_path, output_img, root="root")
                except Exception as e:
                    print(f"⚠️  Failed to visualize {task} ({mode}): {e}")


def main():
    print("🚀 Starting fully parallel execution (all repeats)...")

    tasks = list(TASK_REGISTRY.keys())
    max_workers_total = min(28, os.cpu_count() or 28)

    with ProcessPoolExecutor(max_workers=max_workers_total, initializer=setup_worker) as executor:
        # 1. 提交 Baseline（不变）
        future_baseline = executor.submit(run_baselines_best_of_10)

        # 2. 【关键修改】提交所有 (mode, task, repeat) 组合
        all_atf_futures = {}
        for mode in COLLAPSE_MODES:
            for task in tasks:
                for repeat in range(NUM_REPEATS):
                    future = executor.submit(run_single_repeat, task, mode, repeat)
                    all_atf_futures[future] = (mode, task, repeat)

        # 3. 收集所有 ATF 结果
        all_atf_results = {mode: {task: [] for task in tasks} for mode in COLLAPSE_MODES}
        for future in as_completed(all_atf_futures):
            result = future.result()
            mode = result["adaptoflux"]["collapse_mode"]
            task = result["adaptoflux"]["task_name"]
            all_atf_results[mode][task].append(result)
            print(f"✅ Completed: {mode} | {task} | repeat {result['adaptoflux']['repeat']}")

        # 4. 等待 baseline
        baseline_results = future_baseline.result()
        print("✅ Baselines completed!")

    # 5. 聚合并保存（不变）
    aggregate_and_save(all_atf_results, baseline_results)
    print("\n🎉 All experiments completed!")


if __name__ == "__main__":
    main()