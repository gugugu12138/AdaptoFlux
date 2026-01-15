# experiments/ExternalBaselines/run_baselines_with_time.py
import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入模块（使用绝对导入风格，但通过 -m 运行）
from experiments.ExternalBaselines.tasks.feynman_tasks import get_task, TASK_REGISTRY
from experiments.ExternalBaselines.baselines.gplearn_runner import run_gplearn
from experiments.ExternalBaselines.baselines.xgboost_runner import run_xgboost
from experiments.ExternalBaselines.baselines.pysr_runner import run_pysr

NUM_REPEATS = 10


def setup_worker():
    """限制子进程线程数"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_gplearn_xgboost_for_task(task_name: str):
    """
    并行运行 gplearn 和 xgboost（不含 PySR）
    """
    setup_worker()
    best_gp = {"mse": float('inf')}
    best_xb = {"mse": float('inf')}

    for repeat in range(NUM_REPEATS):
        X, y, expr_str, _ = get_task(task_name, n_samples=50, seed=42 + repeat)

        # gplearn
        t0 = time.time()
        gp_res = run_gplearn(X, y, random_state=42 + repeat)
        gp_res["runtime_sec"] = time.time() - t0

        # xgboost
        t0 = time.time()
        xb_res = run_xgboost(X, y, random_state=42 + repeat)
        xb_res["runtime_sec"] = time.time() - t0

        if gp_res["mse"] < best_gp["mse"]:
            best_gp = gp_res
        if xb_res["mse"] < best_xb["mse"]:
            best_xb = xb_res

    return task_name, {
        "gplearn": best_gp,
        "xgboost": best_xb,
        "ground_truth": expr_str
    }


def run_pysr_serial_for_all_tasks(tasks):
    """
    串行运行所有任务的 PySR（避免多进程崩溃）
    """
    pysr_results = {}
    for task_name in tasks:
        print(f"  Running PySR for {task_name} (10 repeats)...")
        best_ps = {"mse": float('inf')}
        for repeat in range(NUM_REPEATS):
            X, y, expr_str, _ = get_task(task_name, n_samples=50, seed=42 + repeat)
            t0 = time.time()
            ps_res = run_pysr(X, y, random_state=42 + repeat)
            ps_res["runtime_sec"] = time.time() - t0
            if ps_res["mse"] < best_ps["mse"]:
                best_ps = ps_res
        pysr_results[task_name] = best_ps
    return pysr_results


def main():
    tasks = list(TASK_REGISTRY.keys())
    print(f"🧪 Running baselines with runtime (best of 10) on {len(tasks)} tasks...")

    # Step 1: 并行运行 gplearn + xgboost
    print("ParallelGroup: Running gplearn & xgboost in parallel...")
    gx_results = {}
    max_workers = 16

    with ProcessPoolExecutor(max_workers=max_workers, initializer=setup_worker) as executor:
        futures = {
            executor.submit(run_gplearn_xgboost_for_task, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            task_name, res = future.result()
            gx_results[task_name] = res
            print(f"✅ gplearn/xgboost done: {task_name}")

    # Step 2: 串行运行 PySR（安全！）
    print("\nParallelGroup: Running PySR serially (to avoid Julia crash)...")
    pysr_results = run_pysr_serial_for_all_tasks(tasks)

    # Step 3: 合并结果
    final_results = {}
    for task in tasks:
        final_results[task] = {
            "gplearn": gx_results[task]["gplearn"],
            "xgboost": gx_results[task]["xgboost"],
            "pysr": pysr_results[task],
            "ground_truth": gx_results[task]["ground_truth"]
        }

    # Step 4: 保存
    os.makedirs("experiments/ExternalBaselines/results", exist_ok=True)
    out_path = "experiments/ExternalBaselines/results/baselines_best_of_10_with_time.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ All baselines completed! Results saved to {out_path}")


if __name__ == "__main__":
    main()