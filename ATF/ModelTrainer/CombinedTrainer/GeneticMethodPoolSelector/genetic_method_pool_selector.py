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
    使用遗传算法从完整方法池中筛选高性能子方法池。
    适应度 = 在子方法池上运行轻量 LayerGrow（如 2-3 层）得到的最佳准确率。
    完全复用 LayerGrowTrainer 的内部机制，但不保存模型，仅评估性能。
    """

    def __init__(
        self,
        base_adaptoflux,
        input_data: np.ndarray,
        target: np.ndarray,
        population_size: int = 20,
        generations: int = 10,
        subpool_size: int = 10,          # 每个个体包含的方法数量
        layer_grow_layers: int = 2,      # 用于评估的 LayerGrow 最大层数
        layer_grow_attempts: int = 3,    # 每层最大尝试次数（加速）
        data_fraction: float = 0.2,      # 用于评估的小批量数据比例
        elite_ratio: float = 0.2,        # 精英保留比例
        mutation_rate: float = 0.1,      # 变异概率
        crossover_rate: float = 0.8,
        random_seed: int = 42,
        verbose: bool = True
    ):
        self.base_af = base_adaptoflux
        self.full_method_names = list(self.base_af.methods.keys())
        self.subpool_size = subpool_size
        self.population_size = population_size
        self.generations = generations

        # 数据子集用于快速评估
        n_total = input_data.shape[0]
        n_eval = max(10, int(n_total * data_fraction))
        indices = np.random.RandomState(random_seed).choice(n_total, n_eval, replace=False)
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

        # 随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        if self.subpool_size > len(self.full_method_names):
            raise ValueError(f"subpool_size ({subpool_size}) > total methods ({len(self.full_method_names)})")

    def _create_individual(self) -> Set[str]:
        """随机选择 subpool_size 个方法名"""
        return set(random.sample(self.full_method_names, self.subpool_size))

    def _evaluate_individual(self, method_subset: Set[str]) -> float:
        """
        在给定子方法池上运行轻量 LayerGrow，返回最佳准确率。
        复用 LayerGrowTrainer 的核心逻辑，但临时替换方法池。
        """
        # 深拷贝避免污染
        af_copy = copy.deepcopy(self.base_af)
        # 仅保留子集中的方法
        af_copy.methods = {k: v for k, v in af_copy.methods.items() if k in method_subset}
        
        # 创建轻量 LayerGrowTrainer
        lg_trainer = LayerGrowTrainer(
            adaptoflux_instance=af_copy,
            max_attempts=self.lg_attempts,
            decision_threshold=0.0,  # 贪心
            verbose=False
        )

        try:
            # 执行轻量训练（不保存模型）
            results = lg_trainer.train(
                input_data=self.eval_input,
                target=self.eval_target,
                max_layers=self.lg_layers,
                max_total_attempts=self.lg_layers * 100,  # 防止卡死
                save_model=False,
            )

            best_acc = results.get("best_model_accuracy", -1.0)
            if best_acc < 0:
                best_acc = results.get("final_model_accuracy", 0.0)
            return max(0.0, best_acc)
        except Exception as e:
            if self.verbose:
                logger.warning(f"Individual evaluation failed: {e}")
            return 0.0

    def _crossover(self, parent1: Set[str], parent2: Set[str]) -> Set[str]:
        """基于集合的交叉：取并集后随机采样"""
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
        """变异：随机替换部分方法"""
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
        执行遗传算法，返回最佳子方法池及相关信息。
        """
        # 初始化种群
        population = [self._create_individual() for _ in range(self.population_size)]
        fitness_history = []

        # 记录全局最佳个体和适应度
        best_overall_individual = None
        best_overall_fitness = -float('inf') # 初始化为负无穷，确保任何合理的准确率都能更新它

        for gen in range(self.generations):
            # 评估适应度
            fitness_scores = [(ind, self._evaluate_individual(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            best_in_gen = fitness_scores[0]
            fitness_history.append(best_in_gen[1])

            # 更新全局最佳（如果当前代最佳优于全局最佳）
            if best_in_gen[1] > best_overall_fitness:
                best_overall_fitness = best_in_gen[1]
                best_overall_individual = best_in_gen[0].copy() # 确保是副本，防止后续变异等操作影响

            if self.verbose:
                logger.info(f"Generation {gen+1}/{self.generations} | Best Acc: {best_in_gen[1]:.4f}")

            # 精英保留
            elite = [ind for ind, _ in fitness_scores[:self.elite_size]]

            # 生成下一代
            next_gen = elite[:]
            while len(next_gen) < self.population_size:
                parent1 = random.choice(elite)  # 轮盘赌或锦标赛可选，这里简化
                parent2 = random.choice(elite)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_gen.append(child)

            population = next_gen

        # 返回进化过程中记录到的最佳结果，避免最后的重复评估
        # final_best = max(
        #     [(ind, self._evaluate_individual(ind)) for ind in population],
        #     key=lambda x: x[1]
        # )

        # 使用记录的全局最佳
        final_best = (best_overall_individual, best_overall_fitness)

        return {
            "best_subpool": list(final_best[0]),
            "best_fitness": final_best[1],
            "fitness_history": fitness_history,
            "full_method_pool_size": len(self.full_method_names),
            "subpool_size": self.subpool_size
        }