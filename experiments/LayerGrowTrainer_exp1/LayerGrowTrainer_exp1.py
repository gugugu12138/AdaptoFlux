import pandas as pd
import numpy as np
from ATF.core.flux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer

import logging
import os
import random
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


# =============================
# 自定义坍缩方法（需可序列化）
# =============================
def collapse_sum_positive(values):
    total = np.sum(values)
    return 1 if total > 0 else 0


# =============================
# 数据加载函数（必须在顶层定义，以便被子进程调用）
# =============================
def load_titanic_for_adaptoflux(train_processed_path, methods_path=None, collapse_method=CollapseMethod.SUM):
    df = pd.read_csv(train_processed_path)
    if 'Survived' not in df.columns:
        raise ValueError("train_processed.csv 必须包含 'Survived' 列作为标签")
    labels = df['Survived'].values
    values = df.drop(columns=['Survived']).values
    values = np.array(values, dtype=np.float64)
    af = AdaptoFlux(
        values=values,
        labels=labels,
        methods_path=methods_path,
        collapse_method=collapse_method
    )
    af.set_custom_collapse(collapse_sum_positive)
    return af


# =============================
# 单个实验任务函数（必须是顶层函数）
# =============================
def run_single_experiment(args):
    """
    单个实验任务：执行一次 LayerGrowTrainer 训练
    参数:
        args: dict 包含所有需要的参数
    """
    max_layers = args['max_layers']
    run = args['run']
    base_experiment_dir = args['base_experiment_dir']
    common_trainer_params = args['common_trainer_params']
    train_params_template = args['train_params_template']
    seed_offset = args['seed_offset']

    # 设置日志（ProcessPoolExecutor 不继承主进程日志配置，需重新设置）
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(levelname)s] PID:{os.getpid():6} | %(message)s'
    )

    # 设置随机种子
    seed = seed_offset * run + hash(f"max_layers_{max_layers}") % 1000
    random.seed(seed)
    np.random.seed(seed % (2**32))

    # 构造路径
    max_layers_dir = os.path.join(base_experiment_dir, f"exp_max_layers_{max_layers}")
    run_dir = os.path.join(max_layers_dir, f"run_{run}")
    os.makedirs(run_dir, exist_ok=True)

    log_prefix = f"[max_layers={max_layers}, run={run}]"

    try:
        # 每个进程独立加载数据
        train_processed_path = os.path.join(base_experiment_dir, "train_processed.csv")
        methods_py_path = os.path.join(base_experiment_dir, "methods.py")

        model_data = load_titanic_for_adaptoflux(
            train_processed_path=train_processed_path,
            methods_path=methods_py_path,
            collapse_method=CollapseMethod.SUM
        )

        # 构建训练参数
        train_params = train_params_template.copy()
        train_params["input_data"] = model_data.values
        train_params["target"] = model_data.labels
        train_params["model_save_path"] = run_dir
        train_params["max_layers"] = max_layers

        # 初始化训练器
        trainer = LayerGrowTrainer(
            adaptoflux_instance=model_data,
            **common_trainer_params
        )

        # 训练
        start_time = time.time()
        trainer.train(**train_params)
        elapsed = time.time() - start_time

        # 保存结果日志
        result = {
            "status": "success",
            "max_layers": max_layers,
            "run": run,
            "duration": round(elapsed, 2),
            "save_path": run_dir
        }
        logging.info(f"{log_prefix} ✅ 完成，耗时 {elapsed:.2f}s")
        return result

    except Exception as e:
        logging.error(f"{log_prefix} ❌ 失败: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "max_layers": max_layers,
            "run": run,
            "error": str(e)
        }


# =============================
# 主函数
# =============================
def main():
    base_experiment_dir = "experiments/LayerGrowTrainer_exp1"
    os.makedirs(base_experiment_dir, exist_ok=True)

    # 检查数据文件是否存在
    train_processed_path = os.path.join(base_experiment_dir, "train_processed.csv")
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"数据文件未找到: {train_processed_path}")

    # 实验参数
    max_layers_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_runs = 10

    # 训练器参数（模板）
    common_trainer_params = {
        "max_attempts": 10,
        "decision_threshold": 0.0,
        "verbose": True
    }

    train_params_template = {
        "save_model": True,
        "on_retry_exhausted": "rollback",
        "rollback_layers": 2,
        "max_total_attempts": 2500,
        "input_data": None,  # 会在 run_single_experiment 中填充
        "target": None,
        "model_save_path": None,
        "max_layers": None
    }

    # 准备所有任务参数
    tasks = []
    for max_layers in max_layers_list:
        for run in range(1, n_runs + 1):
            task_args = {
                'max_layers': max_layers,
                'run': run,
                'base_experiment_dir': base_experiment_dir,
                'common_trainer_params': common_trainer_params,
                'train_params_template': train_params_template,
                'seed_offset': 42
            }
            tasks.append(task_args)

    # 并行执行
    num_workers = 28 
    print(f"启动并行实验，共 {len(tasks)} 个任务，使用 {num_workers} 个进程。")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(run_single_experiment, task): task for task in tasks}

        # 实时获取完成的任务
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

    # 可选：保存汇总结果
    result_file = os.path.join(base_experiment_dir, "experiment_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 所有实验完成，结果保存至: {result_file}")


if __name__ == "__main__":
    main()