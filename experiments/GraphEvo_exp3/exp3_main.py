# exp3_main.py
import os
import sys
import numpy as np
import copy
import logging
from typing import Tuple, Callable, Dict, Any
import json

# ✅ 标准导入（依赖 pip install -e .）
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer

# 本地辅助模块（与主框架解耦）
from math_tasks import f1, f2, f3, generate_task_data, SCALAR_TYPE


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiments/GraphEvo_exp3/exp3_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_adaptoflux_variant(variant: str) -> AdaptoFlux:
    dummy_values = np.array([[0.0]], dtype=np.float32)
    af = AdaptoFlux(values=dummy_values, methods_path="experiments/GraphEvo_exp3/dummy_methods.py")

    base_methods = [
        ("multiply_by_2", lambda x: [x * 2], 1, 1),
        ("add_3", lambda x: [x + 3], 1, 1),
        ("multiply_by_05", lambda x: [x * 0.5], 1, 1),
        ("identity", lambda x: [x], 1, 1),
    ]

    if variant == "minimal":
        methods = [("increment", lambda x: [x + 1], 1, 1)] + base_methods
    elif variant == "extended":
        methods = [
            ("increment", lambda x: [x + 1], 1, 1),
            ("add_2", lambda x: [x + 2], 1, 1),
        ] + base_methods
    elif variant == "oracle":
        methods = [
            ("increment", lambda x: [x + 1], 1, 1),
            ("f1_direct", lambda x: [2 * (x + 1)], 1, 1),
        ] + base_methods
    else:
        raise ValueError(f"Unknown variant: {variant}")

    for name, func, in_count, out_count in methods:
        af.add_method(
            method_name=name,
            method=func,
            input_count=in_count,
            output_count=out_count,
            input_types=['scalar'] * in_count,
            output_types=['scalar'] * out_count,
            group='math',
            weight=1.0,
            vectorized=True
        )
    return af


def save_detailed_results(
    save_dir: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    adaptoflux: AdaptoFlux
):
    """保存本次重复实验的详细元数据和快照"""
    os.makedirs(save_dir, exist_ok=True)

    # 方法池快照
    method_pool_snapshot = {}
    for name, method_dict in adaptoflux.methods.items():
        method_pool_snapshot[name] = {
            "input_count": method_dict.get("input_count"),
            "output_count": method_dict.get("output_count"),
            "input_types": method_dict.get("input_types"),
            "output_types": method_dict.get("output_types"),
            "group": method_dict.get("group"),
            "weight": method_dict.get("weight"),
            "is_evolved": name.startswith("evolved_method")
        }

    metadata = {
        "config": config,
        "results": results,
        "method_pool_snapshot": method_pool_snapshot
    }

    with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, default=str)


def run_experiment_group(
    group_name: str,
    enable_evolution: bool,
    data1: Tuple[np.ndarray, np.ndarray],
    data2: Tuple[np.ndarray, np.ndarray],
    data3: Tuple[np.ndarray, np.ndarray],
    save_dir_base: str,
    max_init_layers_task12: int,
    method_pool_variant: str,
    rep_index: int,
    max_evo_cycles: int = 3
):
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {group_name} | layers={max_init_layers_task12} | pool={method_pool_variant} | evolution={enable_evolution}")
    logger.info(f"{'='*60}")

    # 阶段1: 任务1
    af = create_adaptoflux_variant(method_pool_variant)
    trainer1 = GraphEvoTrainer(
        adaptoflux_instance=af,
        num_initial_models=5,
        max_refinement_steps=50,
        max_init_layers=max_init_layers_task12,
        enable_evolution=enable_evolution,
        evolution_sampling_frequency=1,
        evolution_trigger_count=2,
        methods_per_evolution=2,
        refinement_strategy="full_sweep",
        candidate_pool_mode="group",
        fallback_mode="group_first",
        verbose=True
    )
    logger.info(">>> Training on Task 1 (f1 = (x+1)*2)")
    trainer1.train(*data1, max_evo_cycles=max_evo_cycles, model_save_path=os.path.join(save_dir_base, "task1"))

    # 阶段2: 任务2
    trainer2 = GraphEvoTrainer(
        adaptoflux_instance=trainer1.adaptoflux,
        num_initial_models=5,
        max_refinement_steps=50,
        max_init_layers=max_init_layers_task12,
        enable_evolution=enable_evolution,
        evolution_sampling_frequency=1,
        evolution_trigger_count=2,
        methods_per_evolution=2,
        refinement_strategy="full_sweep",
        candidate_pool_mode="group",
        fallback_mode="group_first",
        verbose=True
    )
    logger.info(">>> Training on Task 2 (f2 = f1(x) + 3)")
    trainer2.train(*data2, max_evo_cycles=max_evo_cycles, model_save_path=os.path.join(save_dir_base, "task2"))

    evolved_methods = [name for name in trainer2.adaptoflux.methods.keys() if name.startswith("evolved_method")]
    logger.info(f"Evolved methods after Task 2: {evolved_methods}")

    # 阶段3: 任务3（冻结进化）
    trainer3 = GraphEvoTrainer(
        adaptoflux_instance=trainer2.adaptoflux,
        num_initial_models=5,
        max_refinement_steps=100,
        max_init_layers=3,
        enable_evolution=False,
        refinement_strategy="full_sweep",
        candidate_pool_mode="group",
        fallback_mode="group_first",
        verbose=True
    )
    logger.info(">>> Training on Task 3 (f3 = f2(x) * 0.5) —— CRITICAL TASK")
    result3 = trainer3.train(*data3, max_evo_cycles=5, model_save_path=os.path.join(save_dir_base, "task3"))

    final_acc = result3['best_accuracy']
    success = final_acc > 0.95
    logger.info(f"Task 3 Final Accuracy: {final_acc:.4f} → {'SUCCESS' if success else 'FAILURE'}")

    # 构建结果字典
    results_dict = {
        'group': group_name,
        'enable_evolution': enable_evolution,
        'max_init_layers': max_init_layers_task12,
        'method_pool_variant': method_pool_variant,
        'task3_accuracy': float(final_acc),
        'task3_success': bool(success),
        'evolved_methods': evolved_methods,
        'total_methods': len(trainer3.adaptoflux.methods)
    }

    # 保存详细结果
    config_dict = {
        'group_name': group_name,
        'enable_evolution': enable_evolution,
        'max_init_layers_task12': max_init_layers_task12,
        'method_pool_variant': method_pool_variant,
        'rep_index': rep_index,
        'random_seed': 42 + rep_index
    }

    save_detailed_results(save_dir_base, config_dict, results_dict, trainer3.adaptoflux)

    return results_dict


def main():
    base_save_dir = "experiments/GraphEvo_exp3/exp3_results"
    os.makedirs(base_save_dir, exist_ok=True)

    configs = [
        ("Exp_L2_Minimal", True, 2, "minimal"),
        ("Exp_L3_Minimal", True, 3, "minimal"),
        ("Ctrl_L2_Minimal", False, 2, "minimal"),
        ("Ctrl_L3_Minimal", False, 3, "minimal"),
        ("Exp_L2_Extended", True, 2, "extended"),
        ("Ctrl_L2_Extended", False, 2, "extended"),
        ("Oracle_L2", False, 2, "oracle"),
    ]

    n_repeats = 100

    for group_name, enable_evo, layers, pool in configs:
        logger.info(f"\n🔁 Starting {n_repeats} repeats for {group_name}")
        group_dir = os.path.join(base_save_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        repeat_results = []
        for rep in range(n_repeats):
            np.random.seed(42 + rep)
            data1 = generate_task_data(f1, n=500)
            data2 = generate_task_data(f2, n=500)
            data3 = generate_task_data(f3, n=500)

            rep_save_dir = os.path.join(group_dir, f"rep{rep}")
            result = run_experiment_group(
                group_name=f"{group_name}_rep{rep}",
                enable_evolution=enable_evo,
                data1=data1,
                data2=data2,
                data3=data3,
                save_dir_base=rep_save_dir,
                max_init_layers_task12=layers,
                method_pool_variant=pool,
                rep_index=rep,
                max_evo_cycles=3
            )
            repeat_results.append(result)

        # 聚合统计
        accs = [r['task3_accuracy'] for r in repeat_results]
        successes = [r['task3_success'] for r in repeat_results]
        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        success_rate = np.mean(successes)

        group_summary = {
            'avg_task3_accuracy': float(avg_acc),
            'std_task3_accuracy': float(std_acc),
            'success_rate': float(success_rate),
            'n_repeats': n_repeats,
            'raw_results': repeat_results
        }

        with open(os.path.join(group_dir, "aggregated.json"), 'w', encoding='utf-8') as f:
            json.dump(group_summary, f, indent=4, ensure_ascii=False)

        logger.info(f"✅ {group_name} | Avg Acc: {avg_acc:.4f} ± {std_acc:.4f} | Success Rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()