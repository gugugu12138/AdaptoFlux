# -*- coding: utf-8 -*-
# File: experiments/Combined_exp4_text/experiment_text_classification.py
import logging
import numpy as np
import time
import copy
import os
import shutil
import json
import random
import traceback
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# 假设 ATF 已安装
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from ATF.CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod

# --- 1. 设置日志 ---
class BilingualFormatter(logging.Formatter):
    def format(self, record):
        original_msg = record.msg
        prefixes = {logging.INFO: "[INFO]", logging.WARNING: "[WARNING]", logging.ERROR: "[ERROR]"}
        if record.levelno in prefixes:
            record.msg = f"{prefixes[record.levelno]} {original_msg}"
        formatted = super().format(record)
        record.msg = original_msg
        return formatted

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(BilingualFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# --- 2. 加载文本数据集（20 Newsgroups，多分类）---
logger.info("加载 20 Newsgroups 文本数据集（多分类）")
RANDOM_SEED = 42
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=RANDOM_SEED)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=RANDOM_SEED)

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', dtype=np.float32)
X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test = vectorizer.transform(newsgroups_test.data).toarray()
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# 确认类别数
num_classes = len(np.unique(y_train))
task_type = 'binary_classification' if num_classes == 2 else 'multiclass_classification'
logger.info(f"任务类型: {task_type}, 类别数: {num_classes}, 训练样本: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")

# --- 3. 初始化 AdaptoFlux ---
# 注意：AdaptoFlux 默认假设方法输出标量，但多分类需输出向量
# 若你框架不支持多输出方法，可改用 One-vs-Rest（此处假设支持多分类输出）
base_adaptoflux = AdaptoFlux(
    values=X_train.astype(np.float32),
    labels=y_train.astype(np.int32),
    methods_path="experiments/Combined_exp4_text/methods.py"
)
logger.info(f"AdaptoFlux 实例创建完成，加载方法数: {len(base_adaptoflux.methods)}")

# 在主脚本中
num_classes = len(np.unique(y_train))  # 例如 4
collapse_manager = CollapseFunctionManager(method=CollapseMethod.Energy)
collapse_manager.set_num_bins(num_classes)  # ← 关键：对齐类别数

# 然后，自定义一个 collapse 函数，调用 _energy 并 argmax
def collapse_energy_to_label(values):
    probabilities = collapse_manager._energy(values)  # shape: (num_classes,)
    return int(np.argmax(probabilities))  # 返回 0 ~ num_classes-1

if task_type == 'multiclass_classification':
    base_adaptoflux.set_custom_collapse(collapse_energy_to_label)
    logger.info("已设置多分类 collapse 函数（argmax of averaged logits）")
else:
    # 二分类保留默认（或用 collapse_sum_positive）
    pass

# --- 5. 训练配置（适配文本）---
lg_config = {
    "max_attempts": 5,
    "decision_threshold": 0.0,
    "loss_fn": "auto",
    "task_type": task_type,     # ← 关键：指定任务类型
    "verbose": True,
}

ge_config = {
    "verbose": True,
    "init_mode": "fixed",
    "loss_fn": "auto",
    "task_type": task_type,
    "max_init_layers": 6,  # 文本维度高，可稍大
}

lg_train_kwargs_base = {
    "on_retry_exhausted": "rollback",
    "rollback_layers": 2,
    "max_total_attempts": 250,
    "max_layers": 12,
}

ge_train_kwargs_base = {
    'max_total_refinement_attempts': 250,
    'max_evo_cycles': 8,
}

combined_config_base = {
    "layer_grow_config": lg_config,
    "graph_evo_config": ge_config,
    "num_evolution_cycles": 3,
    "target_subpool_size": 8,
    "genetic_config": {
        "population_size": 16,      # 降低以加速
        "generations": 3,
        "subpool_size": 8,
        "data_fraction": 0.2,       # 小批量加速
        "elite_ratio": 0.3,
        "mutation_rate": 0.15,
    },
    "verbose": False,
    "save_dir": "experiments/Combined_exp4_text/temp_runs"
}

# --- 6. 评估函数（使用 ModelTrainer 的评估逻辑）---
def evaluate_model(adaptoflux_instance, X_train, y_train, X_test, y_test, task_type):
    from ATF.ModelTrainer.model_trainer import ModelTrainer  # 假设路径正确
    dummy_trainer = ModelTrainer(adaptoflux_instance, loss_fn='auto', task_type=task_type)
    
    train_acc = dummy_trainer._evaluate_accuracy(X_train, y_train, task_type=task_type)
    test_acc = dummy_trainer._evaluate_accuracy(X_test, y_test, task_type=task_type)
    
    train_loss = dummy_trainer._evaluate_loss(X_train, y_train)
    test_loss = dummy_trainer._evaluate_loss(X_test, y_test)
    
    num_layers = adaptoflux_instance.graph_processor.get_max_layer_from_graph()
    num_nodes = len(adaptoflux_instance.graph_processor.graph.nodes)
    method_pool_size = len(adaptoflux_instance.methods)
    
    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "num_layers": int(num_layers),
        "num_nodes": int(num_nodes),
        "method_pool_size": int(method_pool_size),
    }

# --- 7. 运行组函数（简化版，仅保留 Group D 为例）---
def run_combined_trainer(base_af, X_train, y_train, X_test, y_test, config, mode, seed):
    logger.info(f"运行 CombinedTrainer ({mode}), seed={seed}")
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    af = copy.deepcopy(base_af)
    run_config = copy.deepcopy(config)
    run_config["genetic_mode"] = mode
    run_config["lg_train_kwargs"] = lg_train_kwargs_base
    run_config["ge_train_kwargs"] = ge_train_kwargs_base
    
    save_path = f"experiments/Combined_exp4_text/models/group_d_{mode}_seed_{seed}"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    run_config["save_dir"] = save_path
    
    trainer = CombinedTrainer(
        adaptoflux_instance=af,
        enable_early_stop=True,
        early_stop_eps=1e-5,
        **run_config
    )
    
    start_time = time.time()
    results = trainer.train(input_data=X_train, target=y_train)
    train_time = time.time() - start_time
    
    eval_res = evaluate_model(trainer.adaptoflux, X_train, y_train, X_test, y_test, task_type)
    
    return {
        "strategy": f"Combined_{mode}",
        "train_accuracy": eval_res["train_accuracy"],
        "test_accuracy": eval_res["test_accuracy"],
        "train_loss": eval_res["train_loss"],
        "test_loss": eval_res["test_loss"],
        "training_time": train_time,
        "run_seed": seed,
        "num_layers_in_graph": eval_res["num_layers"],
        "final_method_pool_size": eval_res["method_pool_size"],
    }

# --- 8. 执行实验 ---
os.makedirs("experiments/Combined_exp4_text/models", exist_ok=True)
os.makedirs("experiments/Combined_exp4_text/experiment_results", exist_ok=True)

NUM_REPEATS = 1
seeds = [RANDOM_SEED + i for i in range(NUM_REPEATS)]
all_results = []

for i, seed in enumerate(seeds):
    logger.info(f"=== 开始第 {i+1}/{NUM_REPEATS} 次重复，seed={seed} ===")
    res_dis = run_combined_trainer(base_adaptoflux, X_train, y_train, X_test, y_test, combined_config_base, "disabled", seed)
    res_once = run_combined_trainer(base_adaptoflux, X_train, y_train, X_test, y_test, combined_config_base, "once", seed)
    config_per = copy.deepcopy(combined_config_base)
    config_per["genetic_config"]["generations"] = 2
    res_per = run_combined_trainer(base_adaptoflux, X_train, y_train, X_test, y_test, config_per, "periodic", seed)
    
    all_results.append({
        "repetition": i + 1,
        "seed": seed,
        "results": [res_dis, res_once, res_per]
    })

# --- 9. 聚合与保存结果 ---
strategy_results = {}
for rep in all_results:
    for res in rep["results"]:
        s = res["strategy"]
        if s not in strategy_results:
            strategy_results[s] = {k: [] for k in res if k not in ["strategy"]}
        for k, v in res.items():
            if k != "strategy":
                strategy_results[s][k].append(v)

aggregated = []
for strat, data in strategy_results.items():
    agg = {"strategy": strat}
    for k, vals in data.items():
        if isinstance(vals[0], (int, float)):
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals))
        else:
            agg[k] = vals
    aggregated.append(agg)

# 保存
with open("experiments/Combined_exp4_text/experiment_results/results.json", "w", encoding='utf-8') as f:
    json.dump(aggregated, f, indent=2, ensure_ascii=False)

# 打印摘要
logger.info("\n=== 文本分类实验结果摘要 ===")
for r in aggregated:
    logger.info(
        f"{r['strategy']:<25} | "
        f"Test Acc: {r['mean_test_accuracy']:.4f}±{r['std_test_accuracy']:.4f} | "
        f"Time: {r['mean_training_time']:.1f}s"
    )

logger.info("文本分类实验完成！结果已保存至 experiments/Combined_exp4_text/")