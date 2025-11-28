# genetic_method_pool_selector.py
import logging
import copy
import random
from typing import List, Set, Dict, Any
import numpy as np

from ...LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer

logger = logging.getLogger(__name__)


class GeneticMethodPoolSelector:
    """
    使用遗传算法（Genetic Algorithm, GA）从完整方法池中自动筛选出一个高性能的子方法池。

    该选择器以子方法池在轻量级 LayerGrow 训练任务上的**最高验证准确率**作为适应度（fitness），
    通过初始化随机种群、精英保留、交叉与变异等操作迭代优化，最终返回历史最优子方法池。

    评估过程完全复用 ``LayerGrowTrainer`` 的逻辑，但：
      - 仅使用小批量数据（由 ``data_fraction`` 控制）以加速；
      - 限制最大层数（``layer_grow_layers``）和每层尝试次数（``layer_grow_attempts``）；
      - 不保存模型，仅提取准确率用于适应度计算。

    适用于在多方法、异构操作环境中自动发现对当前任务最有效的操作子集。
    """

    def __init__(
        self,
        base_adaptoflux,
        input_data: np.ndarray,
        target: np.ndarray,
        population_size: int = 20,
        generations: int = 10,
        subpool_size: int = 10,
        layer_grow_layers: int = 2,
        layer_grow_attempts: int = 3,
        data_fraction: float = 0.2,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        verbose: bool = True,
        fitness_metric: str = "accuracy"
    ):
        """
        初始化遗传方法池选择器。

        Args:
            base_adaptoflux (object): 
                一个已配置的 AdaptoFlux 实例，其 ``methods`` 属性包含完整的方法字典（str -> callable）。
                此实例将被深拷贝用于评估，不会被修改。
            input_data (np.ndarray): 
                输入特征数组，形状为 ``(N, ...)``，用于子方法池的性能评估。
            target (np.ndarray): 
                目标标签或值，形状通常为 ``(N,)`` 或 ``(N, D)``，与 ``input_data`` 对齐。
            population_size (int, optional): 
                遗传算法的种群大小。默认为 20。更大的种群可提升多样性但增加计算开销。
            generations (int, optional): 
                遗传算法的迭代代数。默认为 10。控制搜索的深度。
            subpool_size (int, optional): 
                每个个体（即候选子方法池）包含的方法数量。默认为 10。
                必须满足 ``subpool_size <= len(base_adaptoflux.methods)``。
            layer_grow_layers (int, optional): 
                评估个体时，LayerGrow 允许构建的最大网络层数。默认为 2。
                用于限制评估的复杂度，实现快速打分。
            layer_grow_attempts (int, optional): 
                每层 LayerGrow 中尝试添加新方法的最大次数。默认为 3。
                较小的值可显著加速评估，但可能略微降低打分精度。
            data_fraction (float, optional): 
                用于评估的训练/验证数据比例（0.0 ~ 1.0）。默认为 0.2。
                实际采样数量不少于 10 个样本，以保证评估稳定性。
            elite_ratio (float, optional): 
                精英保留比例（0.0 ~ 1.0）。默认为 0.2。
                每代保留前 ``int(elite_ratio * population_size)`` 个最优个体，至少保留 1 个。
            mutation_rate (float, optional): 
                变异率（0.0 ~ 1.0）。默认为 0.1。
                表示每个个体中大约有多少比例的方法会被随机替换（至少替换 1 个）。
            crossover_rate (float, optional): 
                交叉操作发生的概率（0.0 ~ 1.0）。默认为 0.8。
                若未触发交叉，则子代直接复制一个父代。
            verbose (bool, optional): 
                是否启用日志输出（如每代最佳准确率）。默认为 True。
            fitness_metric (str, optional): 
                适应度指标，可选 "accuracy" 或 "loss"。
                - "accuracy": 使用准确率（越大越好）；
                - "loss": 使用损失（越小越好，内部转换为 -loss 作为适应度）。
                默认为 "accuracy"。

        Raises:
            ValueError: 若 ``subpool_size`` 超过可用方法总数。
        """
        self.base_af = base_adaptoflux
        self.full_method_names = list(self.base_af.methods.keys())
        self.subpool_size = subpool_size
        self.population_size = population_size
        self.generations = generations

        # 数据子集用于快速评估
        n_total = input_data.shape[0]
        n_eval = max(10, int(n_total * data_fraction))
        indices = np.random.RandomState().choice(n_total, n_eval, replace=False)
        self.eval_input = input_data[indices]
        self.eval_target = target[indices]

        # LayerGrow 配置（轻量）
        self.lg_layers = layer_grow_layers
        self.lg_attempts = layer_grow_attempts

        # 遗传参数
        self.elite_size = max(1, int(elite_ratio * population_size))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbose = verbose
        self.fitness_metric = fitness_metric.lower()
        

        # 验证 fitness_metric
        if self.fitness_metric not in ("accuracy", "loss"):
            raise ValueError("fitness_metric must be 'accuracy' or 'loss'")

        if self.subpool_size > len(self.full_method_names):
            raise ValueError(f"subpool_size ({subpool_size}) > total methods ({len(self.full_method_names)})")

    def _create_individual(self) -> Set[str]:
        """
        随机创建一个个体（即一个候选子方法池）。

        返回一个包含 ``subpool_size`` 个唯一方法名的集合。

        Returns:
            Set[str]: 一个方法名集合，代表一个候选子方法池。
        """
        return set(random.sample(self.full_method_names, self.subpool_size))

    def _evaluate_individual(self, method_subset: Set[str]) -> float:
        """
        评估一个子方法池的性能，返回其适应度。

        根据 self.fitness_metric 选择指标：
          - "accuracy": 返回最佳准确率（越大越好）；
          - "loss": 返回 -最小损失（即损失越小，适应度越高）。

        若评估失败，返回 0.0（对 accuracy）或一个极小值（如 -1e6，对 loss）。
        但为统一接口，我们始终返回一个“越大越好”的标量。

        Returns:
            float: 适应度值（越大越好）。
        """
        af_copy = copy.deepcopy(self.base_af)
        af_copy.methods = {k: v for k, v in af_copy.methods.items() if k in method_subset}
        
        lg_trainer = LayerGrowTrainer(
            adaptoflux_instance=af_copy,
            max_attempts=self.lg_attempts,
            decision_threshold=0.0,
            verbose=False
        )

        try:
            results = lg_trainer.train(
                input_data=self.eval_input,
                target=self.eval_target,
                max_layers=self.lg_layers,
                max_total_attempts=self.lg_layers * 100,
                save_model=False,
            )

            if self.fitness_metric == "accuracy":
                best_acc = results.get("best_model_accuracy", -1.0)
                if best_acc < 0:
                    best_acc = results.get("final_model_accuracy", 0.0)
                return max(0.0, best_acc)

            elif self.fitness_metric == "loss":
                best_loss = results.get("best_model_loss", float('inf'))
                if not isinstance(best_loss, (int, float)) or best_loss == float('inf'):
                    best_loss = results.get("final_model_loss", float('inf'))
                # 转换为“越大越好”：损失越小，-loss 越大
                if best_loss == float('inf'):
                    return -1e6  # 极差适应度
                return -float(best_loss)

        except Exception as e:
            if self.verbose:
                logger.warning(f"Individual evaluation failed: {e}")
            # 统一返回一个极低适应度
            return 0.0 if self.fitness_metric == "accuracy" else -1e6

    def _crossover(self, parent1: Set[str], parent2: Set[str]) -> Set[str]:
        """
        对两个父代个体执行交叉操作，生成一个子代。

        采用**集合并集采样**策略：
          1. 若随机数 > ``crossover_rate``，直接返回 parent1 的副本；
          2. 否则，合并两个父代的方法集；
          3. 若并集大小不足 ``subpool_size``，从剩余方法中补充；
          4. 最终从中随机采样 ``subpool_size`` 个方法构成子代。

        Args:
            parent1 (Set[str]): 第一个父代个体。
            parent2 (Set[str]): 第二个父代个体。

        Returns:
            Set[str]: 生成的子代个体。
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        union = parent1 | parent2
        if len(union) < self.subpool_size:
            # 补充随机方法
            candidates = list(set(self.full_method_names) - union)
            needed = self.subpool_size - len(union)
            union |= set(random.sample(candidates, min(needed, len(candidates))))
        return set(random.sample(list(union), self.subpool_size))

    def _mutate(self, individual: Set[str]) -> Set[str]:
        """
        对个体执行变异操作：随机替换部分方法。

        具体步骤：
          1. 从个体中随机移除 ``num_mutate`` 个方法（至少 1 个）；
          2. 从不在当前个体中的剩余方法中随机选择相同数量的方法加入。

        Args:
            individual (Set[str]): 待变异的个体。

        Returns:
            Set[str]: 变异后的个体。
        """
        mutated = set(individual)
        num_mutate = max(1, int(self.mutation_rate * self.subpool_size))
        to_remove = random.sample(list(mutated), min(num_mutate, len(mutated)))
        for m in to_remove:
            mutated.remove(m)
        # 补充新方法
        candidates = list(set(self.full_method_names) - mutated)
        if candidates:
            to_add = random.sample(candidates, min(len(to_remove), len(candidates)))
            mutated.update(to_add)
        return mutated

    def select(self) -> Dict[str, Any]:
        """
        执行完整的遗传算法流程，返回最优子方法池及其相关信息。

        算法流程：
          - 初始化随机种群；
          - 迭代多代，每代：
              * 评估所有个体适应度；
              * 记录当前代最佳和全局历史最佳；
              * 保留精英个体；
              * 通过交叉+变异生成新个体，填满种群；
          - 返回全局历史最佳结果（非仅最后一代）。

        Returns:
            Dict[str, Any]: 包含以下键的结果字典：
                - "best_subpool" (List[str]): 选中的最佳方法名列表。
                - "best_fitness" (float): 对应的最佳适应度（最高验证准确率）。
                - "fitness_history" (List[float]): 每代最佳个体的适应度序列。
                - "full_method_pool_size" (int): 原始方法池的总方法数。
                - "subpool_size" (int): 每个子池的目标方法数量。
        """
        # 初始化种群
        population = [self._create_individual() for _ in range(self.population_size)]
        fitness_history = []

        # 记录全局最佳个体和适应度
        best_overall_individual = None
        best_overall_fitness = -float('inf')

        for gen in range(self.generations):
            # 评估适应度
            fitness_scores = [(ind, self._evaluate_individual(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            best_in_gen = fitness_scores[0]
            fitness_history.append(best_in_gen[1])

            # 更新全局最佳（如果当前代最佳优于全局最佳）
            if best_in_gen[1] > best_overall_fitness:
                best_overall_fitness = best_in_gen[1]
                best_overall_individual = best_in_gen[0].copy()

            if self.verbose:
                logger.info(f"Generation {gen+1}/{self.generations} | Best Acc: {best_in_gen[1]:.4f}")

            # 精英保留
            elite = [ind for ind, _ in fitness_scores[:self.elite_size]]

            # 生成下一代
            next_gen = elite[:]
            while len(next_gen) < self.population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_gen.append(child)

            population = next_gen

        # 使用记录的全局最佳（避免最后一代退化）
        final_best = (best_overall_individual, best_overall_fitness)

        return {
            "best_subpool": list(final_best[0]),
            "best_fitness": final_best[1],
            "fitness_history": fitness_history,
            "full_method_pool_size": len(self.full_method_names),
            "subpool_size": self.subpool_size
        }