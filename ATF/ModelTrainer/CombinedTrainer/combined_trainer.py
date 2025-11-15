# combined_trainer.py
import logging
import copy
import os
from typing import Optional, Dict, Any
from ..LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ..GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer

# 导入遗传选择器（可插拔）
try:
    from .GeneticMethodPoolSelector.genetic_method_pool_selector import GeneticMethodPoolSelector
    GENETIC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GeneticMethodPoolSelector not available: {e}")
    GENETIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class CombinedTrainer:
    """
    组合训练器：实现 AdaptoFlux 的完整自进化闭环。
    支持可插拔的遗传筛选模块，用于控制方法池规模与质量。

    遗传筛选模式（genetic_mode）：
    - "disabled": 不使用遗传（默认）
    - "once": 仅在训练开始前执行一次（对应论文 §3.3.5 阶段1）
    - "periodic": 每 genetic_interval 轮执行一次，用于周期性压缩/重选方法池
    """

    def __init__(
        self,
        adaptoflux_instance,
        layer_grow_config: dict,
        graph_evo_config: dict,
        num_evolution_cycles: int = 1,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        # === 遗传筛选可插拔配置 ===
        genetic_mode: str = "disabled",           # "disabled", "once", "periodic"
        genetic_interval: int = 1,                # 仅在 periodic 模式下生效
        target_subpool_size: Optional[int] = None,  # 控制筛选后方法池大小
        genetic_config: Optional[dict] = None,
        refine_only_new_layers: bool = False,
        # === 新增：LayerGrowTrainer.train() 参数 ===
        lg_train_kwargs: Optional[Dict[str, Any]] = None,
        # === 新增：GraphEvoTrainer.train() 参数 ===
        ge_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if genetic_mode not in {"disabled", "once", "periodic"}:
            raise ValueError("genetic_mode must be one of: 'disabled', 'once', 'periodic'")

        if genetic_mode != "disabled" and not GENETIC_AVAILABLE:
            raise ImportError("GeneticMethodPoolSelector is required but not available.")

        self.base_adaptoflux = adaptoflux_instance
        self.layer_grow_config = layer_grow_config
        self.graph_evo_config = graph_evo_config
        self.num_evolution_cycles = num_evolution_cycles
        self.save_dir = save_dir or "combined_training"
        self.verbose = verbose

        # 遗传配置
        self.genetic_mode = genetic_mode
        self.genetic_interval = genetic_interval
        self.target_subpool_size = target_subpool_size
        self.genetic_config = genetic_config or {}
        self.refine_only_new_layers = refine_only_new_layers

        # === 存储新增参数 ===
        self.lg_train_kwargs = lg_train_kwargs or {}
        self.ge_train_kwargs = ge_train_kwargs or {}

        self._final_adaptoflux_instance = None

    def _perform_genetic_selection(self, adaptoflux_instance, input_data, target) -> tuple:
        """
        执行一次遗传筛选，返回 (筛选后的 adaptoflux 实例, 遗传结果字典)
        """
        if self.verbose:
            logger.info("=== 开始遗传筛选方法池 ===")

        # 设置默认参数
        default_params = {
            "population_size": 12,
            "generations": 6,
            "subpool_size": self.target_subpool_size or 8,
            "layer_grow_layers": 2,
            "layer_grow_attempts": 2,
            "data_fraction": 0.2,
            "elite_ratio": 0.25,
            "mutation_rate": 0.1,
            "verbose": self.verbose,
            "random_seed": 42
        }
        genetic_params = {**default_params, **self.genetic_config}

        selector = GeneticMethodPoolSelector(
            base_adaptoflux=adaptoflux_instance,
            input_data=input_data,
            target=target,
            **genetic_params
        )

        result = selector.select()
        best_subpool = result["best_subpool"]

        # 构建新实例
        selected_af = copy.deepcopy(adaptoflux_instance)
        selected_af.methods = {
            k: v for k, v in selected_af.methods.items() if k in best_subpool
        }

        if self.verbose:
            logger.info(f"遗传筛选完成。选出 {len(best_subpool)} 个方法")
            logger.info(f"适应度: {result['best_fitness']:.4f}")

        genetic_log = {
            "used": True,
            "mode": self.genetic_mode,
            "best_subpool": best_subpool,
            "fitness": result["best_fitness"],
            "subpool_size": len(best_subpool),
        }

        return selected_af, genetic_log

    def train(
        self,
        input_data,
        target,
        **kwargs
    ) -> Dict[str, Any]:
        os.makedirs(self.save_dir, exist_ok=True)

        results = {
            "genetic_logs": [],  # 记录每次遗传操作
            "cycles": [],
            "final_model_path": None,
            "best_overall_accuracy": -1.0,
            "best_cycle": -1
        }

        current_af = copy.deepcopy(self.base_adaptoflux)

        # === 初始遗传筛选（once 或 periodic 的第0轮）===
        if self.genetic_mode == "once":
            current_af, log = self._perform_genetic_selection(current_af, input_data, target)
            results["genetic_logs"].append({"cycle": 0, **log})
        elif self.genetic_mode == "periodic":
            # 在 Cycle 0 前执行一次
            current_af, log = self._perform_genetic_selection(current_af, input_data, target)
            results["genetic_logs"].append({"cycle": 0, **log})

        # === 自进化循环 ===
        for cycle in range(self.num_evolution_cycles):
            if self.verbose:
                logger.info(f"=== Combined Training Cycle {cycle + 1}/{self.num_evolution_cycles} ===")

            cycle_result = {}

            old_nodes = set(current_af.graph.nodes)  # 记录 LayerGrow 前的节点（即“旧节点”）

            init_params = {
                k: v for k, v in self.layer_grow_config.items()
                if k in {'max_attempts', 'decision_threshold', 'verbose'}
            }

            # LayerGrow
            lg_trainer = LayerGrowTrainer(
                adaptoflux_instance=current_af,
                **init_params
            )

            # 把 max_layers 传给 train()
            lg_train_kwargs_to_pass = self.lg_train_kwargs.copy()
            lg_train_kwargs_to_pass.update({
                "input_data": input_data,
                "target": target,
                "model_save_path": os.path.join(self.save_dir, f"cycle_{cycle+1}", "layer_grow"),
                "save_best_model": True,
                "max_layers": self.layer_grow_config.get("max_layers", 10),
                **kwargs # 合并传入的通用 kwargs
            })

            lg_result = lg_trainer.train(**lg_train_kwargs_to_pass)

            current_af = lg_trainer.adaptoflux
            cycle_result["layer_grow"] = lg_result

            # === 2. GraphEvo（可选：仅精炼新节点）===
            ge_config = self.graph_evo_config.copy()

            if self.refine_only_new_layers:
                # 计算新节点：LayerGrow 后有、但之前没有的节点
                new_nodes = set(current_af.graph.nodes) - old_nodes
                # 要冻结的节点 = 所有旧节点（包括 root/collapse）
                frozen_nodes = list(old_nodes)

                # 合并用户可能已设置的 frozen_nodes
                user_frozen = ge_config.get("frozen_nodes", [])
                ge_config["frozen_nodes"] = list(set(frozen_nodes) | set(user_frozen))

                if self.verbose:
                    logger.info(f"  [Refine-Only-New] Freezing {len(frozen_nodes)} old nodes, refining {len(new_nodes)} new nodes.")

            # 5. 调用 GraphEvo 时传入 frozen_nodes
            ge_trainer = GraphEvoTrainer(
                adaptoflux_instance=current_af,
                **ge_config
            )

            ge_train_kwargs_to_pass = self.ge_train_kwargs.copy()
            ge_train_kwargs_to_pass.update({
                "input_data": input_data,
                "target": target,
                "model_save_path": os.path.join(self.save_dir, f"cycle_{cycle+1}", "graph_evo"),
                "save_best_model": True,
                "skip_initialization": True,
                **kwargs # 合并传入的通用 kwargs
            })

            ge_result = ge_trainer.train(**ge_train_kwargs_to_pass)

            current_af = ge_trainer.adaptoflux
            cycle_result["graph_evo"] = ge_result

            # 全局最优更新
            final_acc = ge_result.get("best_accuracy", -1.0)
            if final_acc > results["best_overall_accuracy"]:
                results["best_overall_accuracy"] = final_acc
                results["best_cycle"] = cycle + 1
                best_path = os.path.join(self.save_dir, "best_overall")
                current_af.save_model(folder=best_path)
                results["best_model_path"] = best_path

            results["cycles"].append(cycle_result)

            # === 周期性遗传筛选（仅 periodic 模式）===
            if (
                self.genetic_mode == "periodic"
                and (cycle + 1) % self.genetic_interval == 0
                and (cycle + 1) < self.num_evolution_cycles  # 不在最后一轮后执行
            ):
                current_af, log = self._perform_genetic_selection(current_af, input_data, target)
                results["genetic_logs"].append({"cycle": cycle + 1, **log})

        # 保存最终模型
        final_path = os.path.join(self.save_dir, "final")
        current_af.save_model(folder=final_path)
        results["final_model_path"] = final_path
        self._final_adaptoflux_instance = current_af

        # 保存日志
        import json
        log_path = os.path.join(self.save_dir, "combined_training_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=str)
        results["training_log_saved"] = log_path

        return results

    @property
    def adaptoflux(self):
        """
        获取训练完成后的 AdaptoFlux 实例。

        Returns:
            AdaptoFlux: 训练完成后的 AdaptoFlux 实例，如果尚未训练则为 None。
        """
        return self._final_adaptoflux_instance
