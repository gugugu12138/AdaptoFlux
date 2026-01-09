# fill_baselines.py
import os
import json
import numpy as np
from experiments.ExternalBaselines.tasks.feynman_tasks import get_task, TASK_REGISTRY
from experiments.ExternalBaselines.baselines.gplearn_runner import run_gplearn
from experiments.ExternalBaselines.baselines.xgboost_runner import run_xgboost

NUM_REPEATS = 10  # 与 AdaptoFlux 的重复次数保持一致

def main():
    # 1. 加载已有的实验结果
    input_path = "experiments/ExternalBaselines/results/all_results_by_collapse.json"
    output_path = "experiments/ExternalBaselines/results/all_results_by_collapse_with_baselines.json"
    
    print(f"Loading existing results from {input_path}...")
    with open(input_path, 'r') as f:
        all_results = json.load(f)
    
    # 2. 为每个任务准备 baseline 的 best-of-10 结果
    print(f"Running gplearn and XGBoost for each task (best of {NUM_REPEATS} runs)...")
    task_baseline = {}
    for task_name in TASK_REGISTRY.keys():
        print(f"  - Running {NUM_REPEATS} repeats for {task_name}...")
        
        # 使用与 AdaptoFlux 相同的数据集 (注意：数据本身在不同 repeat 间是固定的，但模型的随机性不同)
        X, y, _, _ = get_task(task_name, n_samples=50, seed=42)  # 数据固定，符合常规做法
        
        best_gp = {"mse": float('inf'), "exact_match": False}
        best_xb = {"mse": float('inf'), "exact_match": False}
        
        # 为每个 baseline 运行 NUM_REPEATS 次
        for repeat in range(NUM_REPEATS):
            random_state = 42 + repeat
            gp_res = run_gplearn(X, y, random_state=random_state)
            xb_res = run_xgboost(X, y, random_state=random_state)
            
            # 更新 best gplearn
            if gp_res["mse"] < best_gp["mse"]:
                best_gp = gp_res
            # 更新 best XGBoost
            if xb_res["mse"] < best_xb["mse"]:
                best_xb = xb_res
        
        task_baseline[task_name] = {
            "gplearn": best_gp,
            "xgboost": best_xb
        }
    
    # 3. 将 baseline 的 best-of-10 结果注入到 JSON 的每个 repeat 条目中
    print("Injecting baseline results into all repeats...")
    for collapse_mode in all_results.keys():
        for task_name, repeats in all_results[collapse_mode].items():
            baseline = task_baseline[task_name]
            for repeat_entry in repeats:
                # 替换原来的占位符 { "mse": 0.0, "exact_match": false }
                repeat_entry["gplearn"] = baseline["gplearn"]
                repeat_entry["xgboost"] = baseline["xgboost"]
    
    # 4. 保存更新后的文件
    print(f"Saving updated results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✅ Done! Baseline results (best-of-10) have been filled.")

if __name__ == "__main__":
    main()