# -*- coding: utf-8 -*-
# File: experiments/Combined_exp4_spiral_no_Trigonometric/experiment_4_combined_trainer_v4.py
import logging
import numpy as np
import time
import copy
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
import shutil  # 导入 shutil 模块用于删除目录
import random
import traceback

# 假设你的 AdaptoFlux 框架已正确安装并可导入
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer

def make_spiral(n_samples=1000, noise=0.1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi  # angle
    r = theta + noise * np.random.randn(n)          # radius with noise

    x1 = -r * np.cos(theta)
    y1 = -r * np.sin(theta)
    x2 = r * np.cos(theta)
    y2 = r * np.sin(theta)

    X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X.astype(np.float32), y.astype(np.int32)

def collapse_sum_positive(values):
    total = np.sum(values)
    return 1 if total > 0 else 0

# --- 1. 设置日志 ---
class BilingualFormatter(logging.Formatter):
    def format(self, record):
        original_msg = record.msg
        if record.levelno == logging.INFO:
            record.msg = f"[INFO] {original_msg}"
        elif record.levelno == logging.WARNING:
            record.msg = f"[WARNING] {original_msg}"
        elif record.levelno == logging.ERROR:
            record.msg = f"[ERROR] {original_msg}"
        formatted = super().format(record)
        record.msg = original_msg
        return formatted

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
bilingual_formatter = BilingualFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(bilingual_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# --- 2. 加载/创建数据 ---
logger.info("开始加载/创建数据集 / Starting to load/create dataset")
RANDOM_SEED = 42
# X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=RANDOM_SEED)
X, y = make_spiral(n_samples=1000, noise=0.1, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_data = X_train_scaled.astype(np.float32)
target = y_train.astype(np.int32)
test_input = X_test_scaled.astype(np.float32)
test_target = y_test.astype(np.int32)
logger.info(f"数据集加载完成，训练集形状: {input_data.shape}, 测试集形状: {test_input.shape}")

# --- 3. 初始化 AdaptoFlux 实例 ---
base_adaptoflux = AdaptoFlux(values=input_data, labels=target, methods_path="experiments/Combined_exp4_spiral_no_Trigonometric/methods.py")
base_adaptoflux.set_custom_collapse(collapse_sum_positive)
logger.info("AdaptoFlux 基础实例已创建 / AdaptoFlux base instance created")
print(f"加载的方法数: {len(base_adaptoflux.methods)}")

# --- 4. 定义训练配置 ---
logger.info("定义训练配置 / Defining training configurations")

lg_config = {
    "max_attempts": 5,
    "decision_threshold": 0.0,
    "verbose": False,
}

ge_config = {
    "verbose": False,
    "init_mode": "fixed",
    "max_init_layers": 5,
}

lg_train_kwargs_base = {
    "on_retry_exhausted": "rollback",
    "rollback_layers": 2,
    "max_total_attempts": 250,
}

ge_train_kwargs_base = {
    'max_total_refinement_attempts': 250,
}

combined_config_base = {
    "layer_grow_config": lg_config,
    "graph_evo_config": ge_config,
    "num_evolution_cycles": 3,
    "target_subpool_size": 8,
    "genetic_config": {
        "population_size": 20,
        "generations": 4,
        "subpool_size": 10,
        "data_fraction": 0.3,
        "elite_ratio": 0.3,
        "mutation_rate": 0.15,
    },
    "verbose": False,
    "save_dir": "experiments/Combined_exp4_spiral_no_Trigonometric/temp_test_runs"
}

# --- 5. 决策边界可视化（支持 Train/Test 标注）<<< MODIFIED >>> ---
def plot_decision_boundary(adaptoflux_instance, X_orig, y, title, save_path, scaler, dataset_name="Test"):
    h = 0.02
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    preds = adaptoflux_instance.infer_with_graph(grid_scaled)
    Z = preds.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
    plt.title(f"{title} ({dataset_name})")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# --- 6. 评估函数（同时评估训练集和测试集）<<< MODIFIED >>> ---
def evaluate_model(adaptoflux_instance, train_input, train_target, test_input, test_target):
    def _eval_on(X, y, name):
        try:
            preds = adaptoflux_instance.infer_with_graph(X)
            accuracy = np.mean(preds == y)

            # 尝试用 raw scores 计算 BCE loss
            try:
                raw_scores = adaptoflux_instance.get_raw_outputs(X)
                probs = 1 / (1 + np.exp(-np.clip(raw_scores, -10, 10)))
                loss = -np.mean(
                    y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)
                )
            except Exception:
                loss = np.mean((preds - y) ** 2)  # fallback to MSE

            return {"accuracy": float(accuracy), "loss": float(loss)}
        except Exception as e:
            logger.error(f"评估 {name} 出错:\n{traceback.format_exc()}")
            return {"accuracy": 0.0, "loss": float('inf')}

    train_metrics = _eval_on(train_input, train_target, "train")
    test_metrics = _eval_on(test_input, test_target, "test")

    # 图结构指标（只需计算一次）
    num_layers = adaptoflux_instance.graph_processor.get_max_layer_from_graph()
    num_nodes = len(adaptoflux_instance.graph_processor.graph.nodes)
    method_pool_size = len(adaptoflux_instance.methods)

    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_loss": train_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "num_layers": int(num_layers),
        "num_nodes": int(num_nodes),
        "method_pool_size": int(method_pool_size),
    }

# --- 7. 各组运行函数（增强元信息 + 双集决策边界图）<<< MODIFIED >>> ---
def run_group_a(base_af, input_data, target, test_input, test_target, lg_config, lg_train_kwargs, run_seed):
    logger.info(f"运行组 A (种子: {run_seed}): 仅 LayerGrow")
    random.seed(run_seed); np.random.seed(run_seed); os.environ['PYTHONHASHSEED'] = str(run_seed)
    af_instance = copy.deepcopy(base_af)
    random.seed(run_seed); np.random.seed(run_seed)
    trainer = LayerGrowTrainer(adaptoflux_instance=af_instance, **lg_config)
    save_path = f"experiments/Combined_exp4_spiral_no_Trigonometric/models/group_a_run_{run_seed}"
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    train_kwargs = {**lg_train_kwargs, "input_data": input_data, "target": target, "model_save_path": save_path}
    results = trainer.train(**train_kwargs)
    training_time = time.time() - start_time
    # <<< MODIFIED >>>: 双集评估
    eval_result = evaluate_model(trainer.best_adaptoflux, input_data, target, test_input, test_target)
    # <<< MODIFIED >>>: 保存双集决策边界图
    plot_dir = "experiments/Combined_exp4_spiral_no_Trigonometric/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_decision_boundary(trainer.best_adaptoflux, X_train, y_train, f"Group A (seed={run_seed})", f"{plot_dir}/group_a_{run_seed}_train.png", scaler, "Train")
    plot_decision_boundary(trainer.best_adaptoflux, X_test, y_test, f"Group A (seed={run_seed})", f"{plot_dir}/group_a_{run_seed}_test.png", scaler, "Test")
    return {
        "strategy": "LayerGrow_Only",
        "train_accuracy": eval_result["train_accuracy"],
        "test_accuracy": eval_result["test_accuracy"],
        "train_loss": eval_result["train_loss"],
        "test_loss": eval_result["test_loss"],
        "training_time": training_time,
        "results": results,
        "model_save_path": save_path,
        "run_seed": run_seed,
        "max_layers_used": results.get("layers_added", 0),
        "total_attempts": results.get("total_candidate_attempts", 0),
        "final_method_pool_size": eval_result["method_pool_size"],
        "num_layers_in_graph": eval_result["num_layers"],
        "num_nodes_in_graph": eval_result["num_nodes"],
    }

def run_group_b(base_af, input_data, target, test_input, test_target, ge_config, ge_train_kwargs, run_seed):
    logger.info(f"运行组 B (种子: {run_seed}): 仅 GraphEvo")
    random.seed(run_seed); np.random.seed(run_seed); os.environ['PYTHONHASHSEED'] = str(run_seed)
    af_instance = copy.deepcopy(base_af)
    random.seed(run_seed); np.random.seed(run_seed)
    trainer = GraphEvoTrainer(adaptoflux_instance=af_instance, **ge_config)
    save_path = f"experiments/Combined_exp4_spiral_no_Trigonometric/models/group_b_run_{run_seed}"
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    train_kwargs = {**ge_train_kwargs, "input_data": input_data, "target": target, "model_save_path": save_path}
    results = trainer.train(**train_kwargs)
    training_time = time.time() - start_time
    eval_result = evaluate_model(trainer.adaptoflux, input_data, target, test_input, test_target)
    plot_dir = "experiments/Combined_exp4_spiral_no_Trigonometric/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_decision_boundary(trainer.adaptoflux, X_train, y_train, f"Group B (seed={run_seed})", f"{plot_dir}/group_b_{run_seed}_train.png", scaler, "Train")
    plot_decision_boundary(trainer.adaptoflux, X_test, y_test, f"Group B (seed={run_seed})", f"{plot_dir}/group_b_{run_seed}_test.png", scaler, "Test")
    return {
        "strategy": "GraphEvo_Only",
        "train_accuracy": eval_result["train_accuracy"],
        "test_accuracy": eval_result["test_accuracy"],
        "train_loss": eval_result["train_loss"],
        "test_loss": eval_result["test_loss"],
        "training_time": training_time,
        "results": results,
        "model_save_path": save_path,
        "run_seed": run_seed,
        "max_layers_used": results.get("layers_added", 0),
        "total_attempts": results.get("total_refinement_attempts", 0),
        "final_method_pool_size": eval_result["method_pool_size"],
        "num_layers_in_graph": eval_result["num_layers"],
        "num_nodes_in_graph": eval_result["num_nodes"],
    }

def run_group_c(base_af, input_data, target, test_input, test_target, lg_config, ge_config, lg_train_kwargs, ge_train_kwargs, run_seed):
    logger.info(f"运行组 C (种子: {run_seed}): LayerGrow -> GraphEvo")
    random.seed(run_seed); np.random.seed(run_seed); os.environ['PYTHONHASHSEED'] = str(run_seed)
    af_instance = copy.deepcopy(base_af)
    random.seed(run_seed); np.random.seed(run_seed)

    # LayerGrow
    lg_trainer = LayerGrowTrainer(adaptoflux_instance=af_instance, **lg_config)
    lg_save_path = f"experiments/Combined_exp4_spiral_no_Trigonometric/models/group_c_lg_run_{run_seed}"
    if os.path.exists(lg_save_path): shutil.rmtree(lg_save_path)
    os.makedirs(lg_save_path, exist_ok=True)
    start_time = time.time()
    lg_kwargs = {**lg_train_kwargs, "input_data": input_data, "target": target, "model_save_path": lg_save_path}
    lg_results = lg_trainer.train(**lg_kwargs)
    af_after_lg = lg_trainer.best_adaptoflux

    # GraphEvo
    random.seed(run_seed); np.random.seed(run_seed)
    ge_trainer = GraphEvoTrainer(adaptoflux_instance=af_after_lg, **ge_config)
    ge_save_path = f"experiments/Combined_exp4_spiral_no_Trigonometric/models/group_c_ge_run_{run_seed}"
    if os.path.exists(ge_save_path): shutil.rmtree(ge_save_path)
    os.makedirs(ge_save_path, exist_ok=True)
    ge_kwargs = {**ge_train_kwargs, "input_data": input_data, "target": target, "model_save_path": ge_save_path, "skip_initialization": True}
    ge_results = ge_trainer.train(**ge_kwargs)
    total_time = time.time() - start_time
    eval_result = evaluate_model(ge_trainer.adaptoflux, input_data, target, test_input, test_target)
    plot_dir = "experiments/Combined_exp4_spiral_no_Trigonometric/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_decision_boundary(ge_trainer.adaptoflux, X_train, y_train, f"Group C (seed={run_seed})", f"{plot_dir}/group_c_{run_seed}_train.png", scaler, "Train")
    plot_decision_boundary(ge_trainer.adaptoflux, X_test, y_test, f"Group C (seed={run_seed})", f"{plot_dir}/group_c_{run_seed}_test.png", scaler, "Test")
    return {
        "strategy": "LayerGrow_Then_GraphEvo",
        "train_accuracy": eval_result["train_accuracy"],
        "test_accuracy": eval_result["test_accuracy"],
        "train_loss": eval_result["train_loss"],
        "test_loss": eval_result["test_loss"],
        "training_time": total_time,
        "lg_results": lg_results,
        "ge_results": ge_results,
        "lg_model_save_path": lg_save_path,
        "ge_model_save_path": ge_save_path,
        "run_seed": run_seed,
        "max_layers_used": ge_results.get("layers_added", lg_results.get("layers_added", 0)),
        "total_attempts": lg_results.get("total_candidate_attempts", 0) + ge_results.get("total_refinement_attempts", 0),
        "final_method_pool_size": eval_result["method_pool_size"],
        "num_layers_in_graph": eval_result["num_layers"],
        "num_nodes_in_graph": eval_result["num_nodes"],
    }

def run_group_d(base_af, input_data, target, test_input, test_target, config, genetic_mode, lg_train_kwargs, ge_train_kwargs, run_seed):
    logger.info(f"运行组 D (种子: {run_seed}): CombinedTrainer ({genetic_mode})")
    random.seed(run_seed); np.random.seed(run_seed); os.environ['PYTHONHASHSEED'] = str(run_seed)
    af_instance = copy.deepcopy(base_af)
    random.seed(run_seed); np.random.seed(run_seed)
    run_config = copy.deepcopy(config)
    run_config["genetic_mode"] = genetic_mode
    run_config["lg_train_kwargs"] = lg_train_kwargs
    run_config["ge_train_kwargs"] = ge_train_kwargs
    save_path = f"experiments/Combined_exp4_spiral_no_Trigonometric/models/group_d_{genetic_mode}_run_{run_seed}"
    if os.path.exists(save_path): shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    run_config["save_dir"] = save_path
    trainer = CombinedTrainer(adaptoflux_instance=af_instance, enable_early_stop=True, early_stop_eps=1e-6, **run_config)
    start_time = time.time()
    results = trainer.train(input_data=input_data, target=target)
    training_time = time.time() - start_time
    eval_result = evaluate_model(trainer.adaptoflux, input_data, target, test_input, test_target)
    plot_dir = "experiments/Combined_exp4_spiral_no_Trigonometric/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_decision_boundary(trainer.adaptoflux, X_train, y_train, f"Group D ({genetic_mode}, seed={run_seed})", f"{plot_dir}/group_d_{genetic_mode}_{run_seed}_train.png", scaler, "Train")
    plot_decision_boundary(trainer.adaptoflux, X_test, y_test, f"Group D ({genetic_mode}, seed={run_seed})", f"{plot_dir}/group_d_{genetic_mode}_{run_seed}_test.png", scaler, "Test")
    return {
        "strategy": f"CombinedTrainer_{genetic_mode}",
        "train_accuracy": eval_result["train_accuracy"],
        "test_accuracy": eval_result["test_accuracy"],
        "train_loss": eval_result["train_loss"],
        "test_loss": eval_result["test_loss"],
        "training_time": training_time,
        "results": results,
        "model_save_path": save_path,
        "run_seed": run_seed,
        "max_layers_used": results.get("layers_added", 0),
        "total_attempts": results.get("total_candidate_and_refinement_attempts", 0),
        "final_method_pool_size": eval_result["method_pool_size"],
        "num_layers_in_graph": eval_result["num_layers"],
        "num_nodes_in_graph": eval_result["num_nodes"],
        "genetic_mode": genetic_mode,
    }

# --- 8. 执行实验 ---
NUM_REPEATS = 100  # 注意：当前仍为1，但日志注释已修正
logger.info(f"开始实验 4: 组合训练器消融测试 (重复 {NUM_REPEATS} 次)")

run_seeds = [RANDOM_SEED + i for i in range(NUM_REPEATS)]
all_results = []

for repeat_idx in range(NUM_REPEATS):
    logger.info(f"开始第 {repeat_idx+1} 次重复实验")
    current_seed = run_seeds[repeat_idx]
    random.seed(current_seed); np.random.seed(current_seed); os.environ['PYTHONHASHSEED'] = str(current_seed)
    
    lg_train_kwargs_a = {**lg_train_kwargs_base, "max_layers": 15}
    ge_train_kwargs_b = {**ge_train_kwargs_base, "max_evo_cycles": 10}
    lg_train_kwargs_c = {**lg_train_kwargs_base, "max_layers": 10}
    ge_train_kwargs_c = {**ge_train_kwargs_base, "max_evo_cycles": 8}
    lg_train_kwargs_d = {**lg_train_kwargs_base, "max_layers": 10}
    ge_train_kwargs_d = {**ge_train_kwargs_base, "max_evo_cycles": 8}

    result_a = run_group_a(base_adaptoflux, input_data, target, test_input, test_target, lg_config, lg_train_kwargs_a, current_seed)
    result_b = run_group_b(base_adaptoflux, input_data, target, test_input, test_target, ge_config, ge_train_kwargs_b, current_seed)
    result_c = run_group_c(base_adaptoflux, input_data, target, test_input, test_target, lg_config, ge_config, lg_train_kwargs_c, ge_train_kwargs_c, current_seed)
    result_d_dis = run_group_d(base_adaptoflux, input_data, target, test_input, test_target, combined_config_base, "disabled", lg_train_kwargs_d, ge_train_kwargs_d, current_seed)
    result_d_once = run_group_d(base_adaptoflux, input_data, target, test_input, test_target, combined_config_base, "once", lg_train_kwargs_d, ge_train_kwargs_d, current_seed)
    # 为 periodic 模式创建独立配置副本，并修改 generations
    combined_config_per = copy.deepcopy(combined_config_base)
    combined_config_per["genetic_config"]["generations"] = 2

    result_d_per = run_group_d(
        base_adaptoflux, input_data, target, test_input, test_target,
        combined_config_per, "periodic",
        lg_train_kwargs_d, ge_train_kwargs_d,
        current_seed
    )

    current_repetition_results = {
        "repetition": repeat_idx + 1,
        "run_seed": current_seed,
        "results": [result_a, result_b, result_c, result_d_dis, result_d_once, result_d_per]
    }
    all_results.append(current_repetition_results)
    logger.info(f"第 {repeat_idx+1} 次重复实验完成")

# --- 9. 统合结果（含训练/测试指标）<<< MODIFIED >>> ---
logger.info("开始统合所有重复实验的结果")

strategy_results = {}
for repetition_data in all_results:
    for result in repetition_data["results"]:
        strategy = result["strategy"]
        if strategy not in strategy_results:
            strategy_results[strategy] = {
                "train_accuracies": [], "test_accuracies": [],
                "train_losses": [], "test_losses": [],
                "times": [], "seeds": [],
                "max_layers_used": [], "total_attempts": [],
                "final_method_pool_size": [],
                "num_layers_in_graph": [], "num_nodes_in_graph": [],
                "genetic_modes": []
            }
        strategy_results[strategy]["train_accuracies"].append(result["train_accuracy"])
        strategy_results[strategy]["test_accuracies"].append(result["test_accuracy"])
        strategy_results[strategy]["train_losses"].append(result["train_loss"])
        strategy_results[strategy]["test_losses"].append(result["test_loss"])
        strategy_results[strategy]["times"].append(result["training_time"])
        strategy_results[strategy]["seeds"].append(result["run_seed"])
        strategy_results[strategy]["max_layers_used"].append(result.get("max_layers_used", 0))
        strategy_results[strategy]["total_attempts"].append(result.get("total_attempts", 0))
        strategy_results[strategy]["final_method_pool_size"].append(result.get("final_method_pool_size", 0))
        strategy_results[strategy]["num_layers_in_graph"].append(result.get("num_layers_in_graph", 0))
        strategy_results[strategy]["num_nodes_in_graph"].append(result.get("num_nodes_in_graph", 0))
        strategy_results[strategy]["genetic_modes"].append(result.get("genetic_mode", None))

aggregated_results = []
for strategy, data in strategy_results.items():
    train_acc = np.array(data["train_accuracies"])
    test_acc = np.array(data["test_accuracies"])
    train_loss = np.array(data["train_losses"])
    test_loss = np.array(data["test_losses"])
    times = np.array(data["times"])
    layers = np.array(data["max_layers_used"])
    attempts = np.array(data["total_attempts"])
    pool_size = np.array(data["final_method_pool_size"])
    num_layers = np.array(data["num_layers_in_graph"])
    num_nodes = np.array(data["num_nodes_in_graph"])
    
    agg = {
        "strategy": strategy,
        "mean_train_accuracy": float(np.mean(train_acc)),
        "std_train_accuracy": float(np.std(train_acc)),
        "mean_test_accuracy": float(np.mean(test_acc)),
        "std_test_accuracy": float(np.std(test_acc)),
        "mean_train_loss": float(np.mean(train_loss)),
        "std_train_loss": float(np.std(train_loss)),
        "mean_test_loss": float(np.mean(test_loss)),
        "std_test_loss": float(np.std(test_loss)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "mean_max_layers_used": float(np.mean(layers)),
        "mean_total_attempts": float(np.mean(attempts)),
        "mean_final_method_pool_size": float(np.mean(pool_size)),
        "mean_num_layers_in_graph": float(np.mean(num_layers)),
        "mean_num_nodes_in_graph": float(np.mean(num_nodes)),
        "genetic_modes_used": list(set(gm for gm in data["genetic_modes"] if gm is not None)),
        "all_train_accuracies": data["train_accuracies"],
        "all_test_accuracies": data["test_accuracies"],
        "all_train_losses": data["train_losses"],
        "all_test_losses": data["test_losses"],
        "all_times": data["times"],
        "seeds_used": data["seeds"]
    }
    aggregated_results.append(agg)

# 打印摘要（含 Train/Test）<<< MODIFIED >>> #
logger.info("实验 4 统合结果摘要 (Train / Test):")
logger.info("="*120)
for res in aggregated_results:
    logger.info(
        f"策略: {res['strategy']:<30} | "
        f"Train Acc: {res['mean_train_accuracy']:.4f}±{res['std_train_accuracy']:.4f} | "
        f"Test Acc: {res['mean_test_accuracy']:.4f}±{res['std_test_accuracy']:.4f} | "
        f"Train Loss: {res['mean_train_loss']:.4f}±{res['std_train_loss']:.4f} | "
        f"Test Loss: {res['mean_test_loss']:.4f}±{res['std_test_loss']:.4f} | "
        f"Time: {res['mean_time']:.2f}s"
    )

# 保存结果
os.makedirs("experiments/Combined_exp4_spiral_no_Trigonometric/experiment_results", exist_ok=True)
with open("experiments/Combined_exp4_spiral_no_Trigonometric/experiment_results/exp4_combined_trainer_all_repeats_v4.json", "w", encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, default=str, ensure_ascii=False)
with open("experiments/Combined_exp4_spiral_no_Trigonometric/experiment_results/exp4_combined_trainer_aggregated_v4.json", "w", encoding='utf-8') as f:
    json.dump(aggregated_results, f, indent=4, default=str, ensure_ascii=False)

# 绘图（Accuracy & Time）
strategies = [r["strategy"] for r in aggregated_results]
mean_test_acc = [r["mean_test_accuracy"] for r in aggregated_results]
std_test_acc = [r["std_test_accuracy"] for r in aggregated_results]
mean_time = [r["mean_time"] for r in aggregated_results]
std_time = [r["std_time"] for r in aggregated_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.bar(strategies, mean_test_acc, yerr=std_test_acc, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Test Accuracy Comparison (v4)')
ax1.set_ylabel('Mean Accuracy ± Std')
ax1.tick_params(axis='x', rotation=45)
for i, (a, s) in enumerate(zip(mean_test_acc, std_test_acc)):
    ax1.text(i, a + s + 0.005, f'{a:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=9)

ax2.bar(strategies, mean_time, yerr=std_time, capsize=5, alpha=0.7, color='lightcoral', edgecolor='black')
ax2.set_title('Training Time Comparison (v4)')
ax2.set_ylabel('Mean Time (s) ± Std')
ax2.tick_params(axis='x', rotation=45)
for i, (t, s) in enumerate(zip(mean_time, std_time)):
    ax2.text(i, t + s + 0.1, f'{t:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("experiments/Combined_exp4_spiral_no_Trigonometric/experiment_results/exp4_combined_trainer_comparison_v4.png", dpi=300, bbox_inches='tight')
plt.show()

# 生成报告（含 Train/Test）<<< MODIFIED >>> #
report_path = "experiments/Combined_exp4_spiral_no_Trigonometric/experiment_results/exp4_combined_trainer_report_v4.txt"
with open(report_path, "w", encoding='utf-8') as f:
    f.write("实验 4: 组合训练器消融测试报告 (v4)\n")
    f.write("="*80 + "\n\n")
    f.write(f"重复次数: {NUM_REPEATS}\n")
    f.write("指标: Train/Test Accuracy, Train/Test Loss, 结构大小, 耗时\n\n")
    for res in aggregated_results:
        f.write(f"策略: {res['strategy']}\n")
        f.write(f"  Train Acc: {res['mean_train_accuracy']:.4f} ± {res['std_train_accuracy']:.4f}\n")
        f.write(f"  Test Acc:  {res['mean_test_accuracy']:.4f} ± {res['std_test_accuracy']:.4f}\n")
        f.write(f"  Train Loss: {res['mean_train_loss']:.4f} ± {res['std_train_loss']:.4f}\n")
        f.write(f"  Test Loss:  {res['mean_test_loss']:.4f} ± {res['std_test_loss']:.4f}\n")
        f.write(f"  平均层数: {res['mean_num_layers_in_graph']:.1f}\n")
        f.write(f"  方法池大小: {res['mean_final_method_pool_size']:.1f}\n")
        f.write(f"  总尝试次数: {res['mean_total_attempts']:.0f}\n\n")

logger.info("实验 4 完成。所有结果、图表、决策边界图（Train/Test）和报告已保存。")