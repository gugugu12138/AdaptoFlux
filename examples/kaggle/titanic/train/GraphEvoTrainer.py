import pandas as pd
import numpy as np
from ATF.core.adaptoflux import AdaptoFlux
from ATF.CollapseManager.collapse_functions import CollapseMethod

# 导入你的 GraphEvoTrainer（注意路径是否正确）
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer

import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)

def collapse_sum_positive(values):
    total = np.sum(values)
    return 1 if total > 0 else 0

def load_titanic_for_adaptoflux(train_processed_path, methods_path=None, collapse_method=CollapseMethod.SUM):
    df = pd.read_csv(train_processed_path)
    if 'Survived' not in df.columns:
        raise ValueError("train_processed.csv 必须包含 'Survived' 列作为标签")

    labels = df['Survived'].values
    values = df.drop(columns=['Survived']).values
    values = np.array(values, dtype=np.float64)

    adaptoflux_instance = AdaptoFlux(
        values=values,
        labels=labels,
        methods_path=methods_path,
        collapse_method=collapse_method
    )
    return adaptoflux_instance

# === 加载模型 ===
model = load_titanic_for_adaptoflux(
    train_processed_path='examples/kaggle/titanic/output/train_processed.csv',
    methods_path='examples/kaggle/titanic/methods.py',
    collapse_method=CollapseMethod.SUM
)
model.add_collapse_method(collapse_sum_positive)

# === 使用 GraphEvoTrainer ===
trainer = GraphEvoTrainer(
    adaptoflux_instance=model,
    enable_compression=False,
    num_initial_models=10,
    max_refinement_steps=50,
    compression_threshold=0.95,
    max_init_layers=5,
    evolution_sampling_frequency=1,
    evolution_trigger_count=5,
    init_mode="fixed",
    frozen_nodes=["root", "collapse"],
    frozen_methods=["return_value"],
    refinement_strategy="random_single",
    # === 关键配置 ===
    candidate_pool_mode="group",   # 第三步：在同组中找候选
    fallback_mode="self",          # 第五步：兜底只返回自己
    verbose=True
)

# === 训练 ===
result = trainer.train(
    input_data=model.values,
    target=model.labels,
    max_evo_cycles=5,                           # 总共跑 5 个进化周期
    save_model=True,
    model_save_path="examples/kaggle/titanic/models/graph_evo",         # 修改保存路径
    save_best_model=True,
    best_model_subfolder="best",
    final_model_subfolder="final",
    log_filename="graph_evo_training_log.json"  # 日志文件名
)

print("Training complete. ")