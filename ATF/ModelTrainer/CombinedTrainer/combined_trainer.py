# combined_trainer.py
import logging
import copy
import os
from typing import Optional, Dict, Any
from .LayerGrowTrainer import LayerGrowTrainer
from .GraphEvoTrainer import GraphEvoTrainer

logger = logging.getLogger(__name__)

class CombinedTrainer:
    """
    组合训练器：实现 AdaptoFlux 的完整自进化闭环。
    流程：LayerGrow（主干构建） → GraphEvo（精炼+进化） → （可选）多轮迭代
    """

    def __init__(
        self,
        adaptoflux_instance,
        layer_grow_config: dict,
        graph_evo_config: dict,
        num_evolution_cycles: int = 1,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.base_adaptoflux = adaptoflux_instance
        self.layer_grow_config = layer_grow_config
        self.graph_evo_config = graph_evo_config
        self.num_evolution_cycles = num_evolution_cycles
        self.save_dir = save_dir or "combined_training"
        self.verbose = verbose

    def train(
        self,
        input_data,
        target,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行完整的自进化训练流程。
        """
        os.makedirs(self.save_dir, exist_ok=True)
        results = {
            "cycles": [],
            "final_model_path": None,
            "best_overall_accuracy": -1.0,
            "best_cycle": -1
        }

        current_af = copy.deepcopy(self.base_adaptoflux)

        for cycle in range(self.num_evolution_cycles):
            if self.verbose:
                logger.info(f"=== Combined Training Cycle {cycle + 1}/{self.num_evolution_cycles} ===")

            cycle_result = {}

            # === 阶段 1: LayerGrow 主干构建 ===
            lg_trainer = LayerGrowTrainer(
                adaptoflux_instance=current_af,
                **self.layer_grow_config
            )
            lg_result = lg_trainer.train(
                input_data=input_data,
                target=target,
                model_save_path=os.path.join(self.save_dir, f"cycle_{cycle+1}", "layer_grow"),
                save_best_model=True,
                **kwargs
            )
            current_af = lg_trainer.adaptoflux  # 更新状态
            cycle_result["layer_grow"] = lg_result

            # === 阶段 2: GraphEvo 精炼与进化 ===
            ge_trainer = GraphEvoTrainer(
                adaptoflux_instance=current_af,
                **self.graph_evo_config
            )
            ge_result = ge_trainer.train(
                input_data=input_data,
                target=target,
                model_save_path=os.path.join(self.save_dir, f"cycle_{cycle+1}", "graph_evo"),
                save_best_model=True,
                **kwargs
            )
            current_af = ge_trainer.adaptoflux  # 更新状态（含新方法）
            cycle_result["graph_evo"] = ge_result

            # === 记录最佳模型 ===
            final_acc = ge_result.get("best_accuracy", -1.0)
            if final_acc > results["best_overall_accuracy"]:
                results["best_overall_accuracy"] = final_acc
                results["best_cycle"] = cycle + 1
                # 保存当前最佳模型快照
                best_snapshot = copy.deepcopy(current_af)
                best_path = os.path.join(self.save_dir, "best_overall")
                current_af.save_model(folder=best_path)
                results["best_model_path"] = best_path

            results["cycles"].append(cycle_result)

            # === 为下一轮准备：使用进化后的方法池 ===
            # 注意：GraphEvoTrainer 已将新方法注入 current_af.methods
            # 所以下一轮 LayerGrow 会自动使用增强后的方法池

        # 保存最终模型
        final_path = os.path.join(self.save_dir, "final")
        current_af.save_model(folder=final_path)
        results["final_model_path"] = final_path

        # 保存总日志
        import json
        log_path = os.path.join(self.save_dir, "combined_training_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, default=str)
        results["training_log_saved"] = log_path

        return results