# experiments/ExternalBaselines/adaptoflux_runner.py
import os
import json
import numpy as np
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from .utils import is_numerical_exact_match
from sklearn.model_selection import train_test_split

# 定义四种坍缩函数
def collapse_first(values):
    return float(values[0])

# 这个完全不行，没有实验价值
# def collapse_mean(values):
#     return float(np.mean(values))

def collapse_sum(values):
    return float(np.sum(values))

def collapse_prod(values):
    return float(np.prod(values))

COLLAPSE_FUNCTIONS = {
    "first": collapse_first,
    "sum": collapse_sum,
    "prod": collapse_prod
}

def run_adaptoflux(X, y, methods_path, test_size=0.5, random_state=42, 
                   collapse_mode="mean", save_path=None):
    """
    新增参数:
        collapse_mode (str): one of ["first", "mean", "sum", "prod"]
        save_path (str or None): 保存路径
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    af = AdaptoFlux(values=X_train, labels=y_train, methods_path=methods_path)
    collapse_fn = COLLAPSE_FUNCTIONS[collapse_mode]
    af.set_custom_collapse(collapse_fn)
    
    lg_train_kwargs = {"max_layers": 10, "max_total_attempts": 100}
    ge_train_kwargs = {"max_evo_cycles": 5}
    
    combined_config = {
        "layer_grow_config": {"max_attempts": 10, "rollback_layers": 2, "verbose": False},
        "graph_evo_config": {"refinement_strategy": "full_sweep", "verbose": False},
        "num_evolution_cycles": 3,
        "target_subpool_size": 8,
        "genetic_mode": "disabled",
        "verbose": False,
        "save_dir": os.path.join(save_path, "combined_trainer_temp") if save_path else "temp_adaptoflux",
        "lg_train_kwargs": lg_train_kwargs,
        "ge_train_kwargs": ge_train_kwargs,
        "enable_early_stop": True,
        "early_stop_eps": 1e-6,
    }
    
    trainer = CombinedTrainer(adaptoflux_instance=af, **combined_config)
    trainer.train(input_data=X_train, target=y_train)
    
    # 评估
    y_pred_test = trainer.adaptoflux.infer_with_graph(X_test)
    exact = is_numerical_exact_match(y_pred_test, y_test)
    mse = float(np.mean((y_pred_test - y_test) ** 2))
    final_pool_size = len(trainer.adaptoflux.methods)
    
    return {
        "exact_match": bool(exact),
        "method_pool_size": int(final_pool_size),
        "supports_side_effects": True,
        "mse": mse,
        "save_path": save_path,
        "collapse_mode": collapse_mode
    }