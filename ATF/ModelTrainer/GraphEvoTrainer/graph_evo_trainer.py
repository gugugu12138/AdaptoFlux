# graph_evo_trainer.py
from ..model_trainer import ModelTrainer
import numpy as np
import logging
import random
import copy
from typing import Optional, List, Dict, Any, Tuple
import os
import json
import traceback
from ...PathGenerator.path_generator import PathGenerator
from ...GraphProcessor.graph_processor import GraphProcessor
from ...core.evolved_method import EvolvedMethod
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph
from datetime import datetime

from .Components import (
    BFSSubgraphSampler,
    SubgraphIOExtractor,
    SubgraphReplacer,
    MSEEquivalenceChecker
)

# 设置日志
logger = logging.getLogger(__name__)


class GraphEvoTrainer(ModelTrainer):
    """
    一个继承自 ModelTrainer 的具体训练器。
    该训练器实现了 AdaptoFlux 的图演（GraphEvo）优化框架。
    通过“多样初始化 → 逐节点精炼 → 模块化压缩 → 方法池进化”的闭环流程，实现图结构的自进化优化。
    """

    def __init__(
        self,
        adaptoflux_instance,
        num_initial_models: int = 5,
        max_refinement_steps: int = 100,
        compression_threshold: float = 0.95,
        max_init_layers: int = 3,
        init_mode: str = "fixed",
        init_layers_list: Optional[List[int]] = None,
        frozen_nodes: Optional[List[str]] = None,
        frozen_methods: Optional[List[str]] = None,
        refinement_strategy: str = "random_single",
        candidate_pool_mode: str = "group",
        fallback_mode: Optional[str] = None,
        enable_compression: bool = False,
        enable_evolution: bool = True,
        evolution_sampling_frequency: int = 1,
        evolution_trigger_count: int = 3,
        evolution_cleanup_mode: str = "full",
        consensus_threshold: Optional[float] = None,  # <-- 新增
        methods_per_evolution: int = 1,
        min_subgraph_size_for_evolution: int = 2,  # <-- 新增参数
        verbose: bool = True,
        **kwargs
    ):
        """
        初始化 GraphEvoTrainer，用于执行 AdaptoFlux 的图结构自进化优化流程。

        本训练器通过四个核心阶段实现图结构的迭代优化：
        1. **多样初始化**：生成多个随机初始图，选择性能最优者作为起点；
        2. **逐节点精炼**：对图中可优化节点尝试方法替换，局部搜索更优结构；
        3. **方法池进化**（可选）：基于记录的高性能图结构，抽象新方法注入方法池。
        4. **模块化压缩**（可选）：识别并替换等效的高效子图，实现结构简化；
        

        参数控制说明：
        - **初始化控制**：通过 `num_initial_models`、`init_mode` 等控制初始多样性；
        - **精炼控制**：通过 `refinement_strategy`、`frozen_nodes` 等控制局部搜索行为；
        - **进化控制**：通过 `evolution_sampling_frequency`、`evolution_trigger_count` 等控制进化触发机制。
        - **压缩控制**：通过 `enable_compression`、`compression_threshold` 控制是否启用及等效性标准；
        :param adaptoflux_instance: 已初始化的 AdaptoFlux 实例，作为优化的基础模板。
        :param num_initial_models: 多样初始化阶段生成的候选模型数量。
        :param max_refinement_steps: 单次精炼阶段允许的最大优化步数。
        :param max_total_refinement_attempts: 整个训练过程中允许的最大候选方法评估次数（用于限制计算资源消耗）。若为 None 则不限制。
        :param compression_threshold: 模块化压缩阶段判定子图等效的 MSE 相似度阈值（值越小要求越严格）。
        :param max_init_layers: 初始化时每个候选模型最多添加的随机层数（仅在 init_mode="fixed" 时生效）。
        :param init_mode: 初始化模式，"fixed" 表示所有模型添加相同层数，"list" 表示按 init_layers_list 指定。
        :param init_layers_list: 当 init_mode="list" 时，指定每个候选模型的初始化层数，长度需 ≥ num_initial_models。
        :param frozen_nodes: 显式冻结的节点名称列表（如 ["root", "collapse"]），这些节点在精炼阶段不会被修改。
        :param frozen_methods: 显式冻结的方法名称列表（如 ["return_value"]），使用这些方法的节点将自动被冻结。
        :param refinement_strategy: 精炼策略，"random_single"（随机单点优化）或 "full_sweep"（全图遍历优化）。
        :param candidate_pool_mode: 构建候选方法池的策略，"all"（所有方法）、"group"（同组方法）或 "self"（仅自身）。
        :param fallback_mode: 当无类型兼容方法时的兜底策略，"all"/"group_first"/"self"/"error"。
        :param enable_compression: 是否启用模块化压缩阶段。
        :param enable_evolution: 是否启用方法池进化阶段。
        :param evolution_sampling_frequency: 每隔多少个训练轮次（即一次完整的「精炼+压缩」循环）记录一次当前图结构快照，用于后续方法池进化。例如设为 2 表示每 2 轮保存一次快照。
        :param evolution_trigger_count: 当累计记录的图快照数量达到此值时，触发一次方法池进化。
        :param evolution_cleanup_mode: 进化完成后如何清理已记录的快照，"full"（清空全部）或 "oldest"（仅移除最早的一个）。
        :param methods_per_evolution: 每次方法池进化时，最多从记录的图结构中抽象并添加的新方法数量。
        :param verbose: 是否输出详细日志信息。
        :param kwargs: 其他可选组件，如自定义的 subgraph_sampler、io_extractor 等。
        """

        super().__init__(adaptoflux_instance)
        self.num_initial_models = num_initial_models
        self.max_refinement_steps = max_refinement_steps
        self.compression_threshold = compression_threshold
        self.max_init_layers = max_init_layers
        self.init_mode = init_mode
        self.init_layers_list = init_layers_list
        self.frozen_nodes = set(frozen_nodes) if frozen_nodes else set()
        self.frozen_methods = set(frozen_methods) if frozen_methods else set()  # <-- 保存为集合
        self.candidate_pool_mode = candidate_pool_mode
        self.fallback_mode = fallback_mode or 'self'  
        self.refinement_strategy = refinement_strategy
        self.enable_compression = enable_compression
        self.enable_evolution = enable_evolution
        self.evolution_sampling_frequency = evolution_sampling_frequency
        self.evolution_trigger_count = evolution_trigger_count
        self.evolution_cleanup_mode = evolution_cleanup_mode
        self.consensus_threshold = consensus_threshold
        self.methods_per_evolution = methods_per_evolution    
        self.verbose = verbose

        self.subgraph_sampler = kwargs.get('subgraph_sampler') or BFSSubgraphSampler(max_nodes=4)
        self.io_extractor = kwargs.get('io_extractor') or SubgraphIOExtractor()
        self.replacer = kwargs.get('replacer') or SubgraphReplacer()
        self.equivalence_checker = kwargs.get('equivalence_checker') or MSEEquivalenceChecker(
            threshold=self.compression_threshold
        )

        self.min_subgraph_size_for_evolution = min_subgraph_size_for_evolution
        if self.min_subgraph_size_for_evolution < 1:
            raise ValueError("min_subgraph_size_for_evolution must be at least 1")

        # 校验参数
        if self.methods_per_evolution < 1:
            raise ValueError("methods_per_evolution must be >= 1")
        if self.init_mode == "list":
            if self.init_layers_list is None:
                raise ValueError("init_mode='list' requires init_layers_list to be provided.")
            if len(self.init_layers_list) < self.num_initial_models:
                raise ValueError(f"init_layers_list length ({len(self.init_layers_list)}) must be >= num_initial_models ({self.num_initial_models})")
        if self.consensus_threshold is not None:
            if not (0.0 <= self.consensus_threshold <= 1.0):
                raise ValueError("consensus_threshold must be in [0.0, 1.0] or None")

        # 用于记录完整图结构快照（用于方法池进化）
        self.graph_snapshots: List[Any] = []
        
        self._strategy_map = {
            "random_single": self._refine_random_single_step,
            "full_sweep": self._refine_full_sweep_step,
            # 未来可加："weighted_sample": self._refine_weighted_sample_step,
        }
        if self.refinement_strategy not in self._strategy_map:
            raise ValueError(f"Unknown refinement_strategy: {self.refinement_strategy}. "
                            f"Available: {list(self._strategy_map.keys())}")

        # 校验
        valid_pool_modes = {"all", "group", "self"}
        valid_fallback_modes = {"all", "group_first", "self", "error"}
        if self.candidate_pool_mode not in valid_pool_modes:
            raise ValueError(f"Invalid candidate_pool_mode: {self.candidate_pool_mode}")
        if self.fallback_mode not in valid_fallback_modes:
            raise ValueError(f"Invalid fallback_mode: {self.fallback_mode}")

        if self.enable_compression == True and self.verbose:
            logger.warning("模块化压缩为未完成的实验性功能，使用时大概率有严重bug，如无自行修改建议不要使用！")

    def _phase_diverse_initialization(self, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        阶段一：多样初始化 (Diverse Initialization)
        随机生成多组初始模型，通过快速评估筛选出最优者作为优化起点。

        :param input_data: 用于评估的输入数据
        :param target: 对应的标签
        :return: 包含最优模型信息的字典 {'best_model': AdaptoFlux实例, 'best_loss': float, 'best_acc': float}
        """
        if self.verbose:
            logger.info(f"[Phase 1] Diverse Initialization: Generating {self.num_initial_models} candidate models...")

        candidates = []
        for i in range(self.num_initial_models):
            # 创建一个新的、独立的 AdaptoFlux 实例副本
            # 注意：这里假设 AdaptoFlux 类有一个 `clone()` 方法或类似的机制
            # 如果没有，您需要实现一个深拷贝逻辑，确保 graph 和 methods 都被复制
            try:
                candidate_af = self.adaptoflux.clone()
            except AttributeError:
                # 如果没有 clone 方法，则进行深拷贝
                candidate_af = copy.deepcopy(self.adaptoflux)
                # 重置其内部状态，确保独立性
                candidate_af.graph_processor.graph = copy.deepcopy(self.adaptoflux.graph_processor.graph)
                candidate_af.methods = copy.deepcopy(self.adaptoflux.methods)

            # 根据初始化模式决定添加层数
            if self.init_mode == "fixed":
                layers_to_add = self.max_init_layers
            elif self.init_mode == "list":
                layers_to_add = self.init_layers_list[i]  # 第 i 个候选模型使用列表中第 i 项
            else:
                raise ValueError(f"Unsupported init_mode: {self.init_mode}")

            # 对候选模型进行随机初始化
            self._randomly_initialize_graph(candidate_af, num_layers_to_add=layers_to_add)

            # 评估候选模型
            loss = self._evaluate_loss(input_data, target, adaptoflux_instance=candidate_af)
            acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=candidate_af)

            candidates.append({
                'model': candidate_af,
                'loss': loss,
                'accuracy': acc,
                'id': i
            })

            if self.verbose:
                logger.info(f"  Candidate {i+1}/{self.num_initial_models}: Loss={loss:.6f}, Acc={acc:.4f}")

        # 选择损失最低的模型作为最优模型
        best_candidate = min(candidates, key=lambda x: x['loss'])

        if self.verbose:
            logger.info(f"[Phase 1] Selected best initial model (ID: {best_candidate['id']}) with Loss={best_candidate['loss']:.6f}, Acc={best_candidate['accuracy']:.4f}")

        return {
            'best_model': best_candidate['model'],
            'best_loss': best_candidate['loss'],
            'best_accuracy': best_candidate['accuracy']
        }

    def _randomly_initialize_graph(self, adaptoflux_instance, num_layers_to_add: int = 3):
        """
        一个辅助方法，用于对给定的 AdaptoFlux 实例进行随机的图结构初始化。
        通过调用其 `append_nx_layer` 方法随机添加几层。

        :param adaptoflux_instance: 要初始化的 AdaptoFlux 实例
        :param num_layers_to_add: 要添加的层数
        """
        while adaptoflux_instance.graph_processor.layer > 0:
            adaptoflux_instance.remove_last_nx_layer()
        for _ in range(num_layers_to_add):
            candidate_plan = adaptoflux_instance.process_random_method()
            if candidate_plan["valid_groups"]:
                try:
                    adaptoflux_instance.append_nx_layer(
                        candidate_plan,
                        discard_unmatched='to_discard',
                        discard_node_method_name="null"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add random layer during initialization: {e}", exc_info=True)
                    # 如果添加失败，跳过，不影响整体流程

    def _phase_node_refinement(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        阶段二：逐节点精炼 (Node-wise Refinement)
        
        目标：
            对图中每个“可处理节点”（processing node），尝试用兼容的替代方法进行替换，
            以降低损失（或提升准确率）。这是一个**局部搜索优化过程**。
        
        策略：
            支持多种优化策略（如随机单点、完整遍历），由 `self.refinement_strategy` 控制。
            每次替换都基于小批量数据快速评估，避免全量计算开销。
        
        冻结机制：
            - `frozen_nodes`: 显式冻结的节点名列表（如 "root", "collapse"）
            - `frozen_methods`: 显式冻结的方法名列表（如 ["return_value"]），自动冻结使用这些方法的节点
        
        返回：
            包含最终模型、性能指标、步数等信息的字典。
        """
        # 打印日志：开始精炼阶段
        if self.verbose:
            logger.info(f"[Phase 2] Node-wise Refinement: Starting refinement with strategy '{self.refinement_strategy}'...")


        # 获取图处理器和当前性能指标
        gp = adaptoflux_instance.graph_processor
        current_loss = self._evaluate_loss(input_data, target, adaptoflux_instance=adaptoflux_instance)
        current_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

        # 初始化状态变量
        improvement_made = False  # 标记本轮是否发生过改进
        steps_taken = 0           # 实际执行的“优化步数”（注意：不同策略步数定义不同）

        # === 1. 获取所有“可处理节点”并应用冻结规则 ===
        # `_is_processing_node(node)` 是 GraphProcessor 的方法，用于判断节点是否是“中间计算节点”
        # （通常排除 "root"、"collapse" 等特殊节点）
        processing_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]

        # 合并显式冻结的节点和基于方法名冻结的节点
        final_frozen_nodes = set(self.frozen_nodes)  # 显式冻结的节点名集合

        # 如果用户指定了冻结的方法名（如 ["return_value"]），则自动冻结所有使用这些方法的节点
        if self.frozen_methods:
            method_frozen_nodes = {
                node for node in processing_nodes
                if gp.graph.nodes[node].get('method_name') in self.frozen_methods
            }
            final_frozen_nodes.update(method_frozen_nodes)  # 合并到冻结集合

        # 从可处理节点中移除所有冻结节点
        if final_frozen_nodes:
            processing_nodes = [node for node in processing_nodes if node not in final_frozen_nodes]

        # 打印日志：报告可优化节点数量
        if self.verbose:
            logger.info(f"  Found {len(processing_nodes)} processing nodes to refine "
                        f"(excluding {len(final_frozen_nodes)} frozen nodes).")

        # === 2. 获取优化策略函数 ===
        # `_strategy_map` 将字符串策略名映射到具体实现函数
        strategy_func = self._strategy_map[self.refinement_strategy]

        # === 3. 主优化循环 ===
        for step in range(self.max_refinement_steps):

            if (self.max_total_refinement_attempts is not None and
                self._total_refinement_attempts >= self.max_total_refinement_attempts):
                if self.verbose:
                    logger.info(
                        f"[Phase 2] Stopped early: reached max_total_refinement_attempts="
                        f"{self.max_total_refinement_attempts} (at step {step})."
                    )
                break

            # 安全退出：防止超过最大步数（虽然策略函数可能一次执行多步）
            if steps_taken >= self.max_refinement_steps:
                break

            # 若无可优化节点，提前退出
            if not processing_nodes:
                break

            # 调用具体策略函数执行一次优化尝试
            # 返回值说明（见策略函数文档）：
            #   imp: bool → 是否发生改进
            #   new_loss, new_acc: 改进后的指标
            #   step_inc: int → 本次实际消耗的“步数”（如 full_sweep 一次可能替换多个节点）
            #   updated_nodes: List[str] → 更新后的可处理节点列表（图结构可能变化）
            imp, new_loss, new_acc, step_inc, updated_nodes = strategy_func(
                adaptoflux_instance, input_data, target, processing_nodes, current_loss, gp
            )

            if imp:
                # 如果发生改进，更新全局状态
                improvement_made = True
                current_loss = new_loss
                current_acc = new_acc
                steps_taken += step_inc
                processing_nodes = updated_nodes  # 使用最新节点列表

                # 注意：日志已在策略函数内部打印（避免重复），所以这里不做额外输出
            else:
                # 无改进：继续下一轮尝试（不提前终止，因为后续可能有改进）
                # 也可在此处加入“早停”逻辑（如连续 N 次无改进则退出）
                pass

        # === 4. 打印最终结果日志 ===
        if self.verbose:
            if improvement_made:
                logger.info(f"[Phase 2] Refinement completed in {steps_taken} steps. Final Loss: {current_loss:.6f}, Acc: {current_acc:.4f}")
            else:
                logger.info(f"[Phase 2] Refinement completed. No improvements found within {self.max_refinement_steps} steps.")

        # === 5. 返回结果 ===
        return {
            'final_model': adaptoflux_instance,      # 优化后的模型（原地修改）
            'final_loss': current_loss,              # 最终损失
            'final_accuracy': current_acc,           # 最终准确率
            'steps_taken': steps_taken,              # 实际执行步数
            'improvement_made': improvement_made,     # 是否有改进
            'total_refinement_attempts': self._total_refinement_attempts,  # 前向推理次数
        }

    def _get_compatible_methods_for_node(
        self, 
        adaptoflux_instance, 
        node_name: str, 
    ) -> List[str]:
        """
        获取与图中指定节点兼容的候选方法列表。

        新增参数:
            allow_fallback_on_empty (bool): 
                当类型兼容方法为空时，是否允许回退到非类型安全的兜底策略。
                - True（默认）：启用兜底，保证返回非空列表；
                - False：若无类型兼容方法，直接抛出 RuntimeError。

        其余参数说明见原注释。
        """
        
        gp = adaptoflux_instance.graph_processor
        methods = adaptoflux_instance.methods
        all_method_names = list(methods.keys())

        if not all_method_names:
            return []

        # === 1. 获取原始方法 ===
        node_data = gp.graph.nodes[node_name]
        original_method_name = node_data.get("method_name")
        if original_method_name is None or original_method_name not in methods:
            error_msg = (
                f"Node '{node_name}' has no valid 'method_name'.\n"
                f"  - Current method_name: {original_method_name}\n"
                f"  - Available methods: {sorted(methods.keys())}\n"
                f"  - Node data: {node_data}"
            )
            logger.error("CRITICAL GRAPH ERROR:\n%s", error_msg)
            # 打印完整堆栈（便于定位是哪个调用链导致的）
            logger.error("Full traceback:\n%s", traceback.format_exc())
            # 抛出异常，终止程序（除非外层有特殊处理）
            raise RuntimeError(error_msg)

        # === 2. 提取原始类型 ===
        orig_info = methods[original_method_name]
        orig_input_types = orig_info.get("input_types", []) or []
        orig_output_types = orig_info.get("output_types", []) or []
        def is_type_compatible(method_name: str) -> bool:
            info = methods[method_name]
            m_input = info.get("input_types", []) or []
            m_output = info.get("output_types", []) or []
            return m_input == orig_input_types and m_output == orig_output_types

        # === 3. 构建候选池 ===
        if self.candidate_pool_mode == "all":
            candidate_pool = all_method_names
        elif self.candidate_pool_mode == "self":
            candidate_pool = [original_method_name]
        else:  # "group"
            node_group = methods[original_method_name].get("group", "default")
            candidate_pool = [
                name for name, info in methods.items()
                if info.get("group", "default") == node_group
            ]

        # === 4. 筛选类型兼容方法 ===
        compatible_methods = [name for name in candidate_pool if is_type_compatible(name)]
        # print(f"Node '{node_name}' compatible methods: {compatible_methods}")

        # === 5. 处理空结果 ===
        if not compatible_methods:
            log_msg = (f"Node '{node_name}' has no compatible methods.")
            logger.debug(log_msg)

            if self.fallback_mode == "error":
                raise RuntimeError(f"Strict mode: no type-compatible methods for node '{node_name}'...")

            # === 执行兜底回退 ===
            if self.fallback_mode == "all":
                result = all_method_names
            elif self.fallback_mode == "group_first":
                node_group = methods[original_method_name].get("group", "default")
                group_methods = [name for name, info in methods.items() if info.get("group") == node_group]
                result = group_methods[:1] if group_methods else all_method_names[:1]
            elif self.fallback_mode == "self":
                result = [original_method_name]  # 👈 关键：只返回自己
            else:
                result = all_method_names  # fallback

            logger.warning("Falling back to non-type-safe methods for node '%s'...", node_name)
            return result

        return compatible_methods

    def _phase_modular_compression(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        阶段四：模块化压缩 (Modular Compression)
        识别图中可被替换的高效子图，用更小或更快的等效结构进行替代。

        本阶段执行以下流程：
        1. **子图采样**：使用配置的采样器（如 BFS）随机选取一个连通子图；
        2. **I/O 提取**：执行原图，记录该子图的输入与输出数据；
        3. **替代生成**：（当前简化为）选择一个候选方法（如 "add"）作为替代结构；
        4. **等效性验证**：比较替代结构与原子图在相同输入下的输出是否足够相似；
        5. **图结构替换**：若验证通过，则用替代结构（当前为单节点）替换原子图；
        6. **记录高性能子图**：将被替换的原子图保存至 `high_performance_subgraphs`，供进化阶段使用。

        注意：当前实现中，替代结构为**单个节点**，且候选方法固定为 "add"。
        后续可扩展为训练一个小型替代子图以实现更优压缩。

        :param adaptoflux_instance: 待优化的 AdaptoFlux 实例
        :param input_data: 用于评估等效性的输入数据（小批量）
        :param target: 对应的标签（用于最终性能评估，非等效性验证必需）
        :return: 包含压缩后模型信息和压缩情况的字典，字段包括：
                 - 'final_model': 压缩后的 AdaptoFlux 实例（可能未变）
                 - 'final_loss': 压缩后的损失值
                 - 'final_accuracy': 压缩后的准确率
                 - 'compression_applied': bool，是否成功执行了压缩
                 - 'compressed_subgraphs': int，本次压缩的子图数量（0 或 1）
        """
        if not self.enable_compression:
            return self._return_original_result(adaptoflux_instance, input_data, target)

        gp = adaptoflux_instance.graph_processor

        # 1. 采样子图
        subgraph = self.subgraph_sampler.sample(gp.graph)
        if subgraph is None:
            return self._return_original_result(adaptoflux_instance, input_data, target)

        # 2. 提取 I/O
        try:
            sub_inputs, sub_outputs = self.io_extractor.extract(adaptoflux_instance, subgraph, input_data)
        except Exception as e:
            logger.warning(f"IO extraction failed: {e}")
            return self._return_original_result(adaptoflux_instance, input_data, target)

        # 3. 简化：直接选一个候选方法（如 "add"）作为替代（跳过训练）
        candidate_method = "add"  # 后续可替换为训练逻辑
        if candidate_method not in adaptoflux_instance.methods:
            return self._return_original_result(adaptoflux_instance, input_data, target)

        # 4. 验证等效性
        # 构建临时单节点图并测试
        temp_af = self._create_single_node_graph(adaptoflux_instance, candidate_method)
        rep_output = temp_af.infer_with_graph(input_data)
        orig_output = list(sub_outputs.values())[0]

        if not self.equivalence_checker.is_equivalent(orig_output, rep_output):
            return self._return_original_result(adaptoflux_instance, input_data, target)

        # 5. 执行替换
        try:
            new_node_id = self.replacer.replace_with_node(gp, subgraph, candidate_method)
            logger.info(f"Replaced subgraph with node '{new_node_id}' ({candidate_method})")

            # 返回新结果
            new_loss = self._evaluate_loss(input_data, target, adaptoflux_instance=adaptoflux_instance)
            new_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)
            return {
                'final_model': adaptoflux_instance,
                'final_loss': new_loss,
                'final_accuracy': new_acc,
                'compression_applied': True,
                'compressed_subgraphs': 1
            }

        except Exception as e:
            logger.error(f"Replacement failed: {e}")
            return self._return_original_result(adaptoflux_instance, input_data, target)

    def _return_original_result(self, adaptoflux_instance, input_data, target):
        """
        辅助方法：返回未进行任何压缩的原始模型评估结果。

        当模块化压缩因以下原因跳过时调用：
        - 压缩功能被禁用（enable_compression=False）
        - 未能采样到有效子图
        - I/O 提取失败
        - 候选替代方法不可用
        - 等效性验证未通过
        - 图替换过程中发生异常

        该方法确保压缩阶段始终返回结构一致的结果字典，便于主训练循环统一处理。

        :param adaptoflux_instance: 当前的 AdaptoFlux 实例（未修改）
        :param input_data: 用于评估的输入数据
        :param target: 对应的标签
        :return: 表示“无压缩”的标准结果字典
        """
        loss = self._evaluate_loss(input_data, target, adaptoflux_instance=adaptoflux_instance)
        acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)
        return {
            'final_model': adaptoflux_instance,
            'final_loss': loss,
            'final_accuracy': acc,
            'compression_applied': False,
            'compressed_subgraphs': 0
        }

    def _extract_node_signature(self, graph, node_id: str) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        """
        从图中提取节点的拓扑签名：(layer, sorted_in_coords, sorted_out_coords)
        
        :param graph: NetworkX 图
        :param node_id: 节点 ID，格式如 "L_I_method"
        :return: (layer, in_coords_tuple, out_coords_tuple)
        """
        # 1. 提取 layer
        if node_id in ("root", "collapse"):
            raise ValueError("Skip special nodes")
        try:
            layer = int(node_id.split('_', 1)[0])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse layer from node ID: {node_id}")

        # 2. 收集所有输入边的 data_coord（指向该节点的边）
        in_coords = []
        for src, _, edge_data in graph.in_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                in_coords.append(coord)
        in_coords = tuple(sorted(in_coords))

        # 3. 收集所有输出边的 data_coord（从该节点出发的边）
        out_coords = []
        for _, dst, edge_data in graph.out_edges(node_id, data=True):
            coord = edge_data.get('data_coord')
            if coord is not None:
                out_coords.append(coord)
        out_coords = tuple(sorted(out_coords))

        return (layer, in_coords, out_coords)
    
    def _phase_method_pool_evolution(
        self,
        adaptoflux_instance,
        snapshots: List[Any],
        max_methods: int = 1,
        enable_graph_isomorphism_clustering: bool = True,
        evolved_methods_save_dir: Optional[str] = None,
        subgraph_selection_policy: str = "largest"
    ) -> Dict[str, Any]:
        # 后续可能添加参数允许选择初始化的多个模型进入后面阶段单个任务构建多个实例，实现一种在单任务上实现类多任务的知识提取，这种方法效果会更好但是消耗性能更多
        # 后续可能添加其他种类的选取连图子图的策略
        # 打印日志：开始方法池进化阶段，说明将基于拓扑签名对齐节点
        # 对齐依据：(层号, 输入数据坐标集合, 输出数据坐标集合)
        # 这种签名能唯一标识节点在数据流图中的拓扑角色，与节点ID或方法名无关
        if self.verbose:
            logger.warning(f"该函数（_phase_method_pool_evolution）仍在开发中，如出现与预期不符请联系开发者")
            logger.info(f"[Phase 3] Method Pool Evolution: Aligning nodes via (layer, in_coords, out_coords) "
                        f"across {len(snapshots)} snapshots...")

        # 安全检查：若无快照可供分析，直接返回空结果
        if not snapshots:
            return {'methods_added': 0, 'new_method_names': []}

        # Step 1: 构建拓扑签名到方法频次的映射
        # 结构：signature_freq[signature][method_name] = count
        # 其中 signature = (layer, (in_coord1, in_coord2, ...), (out_coord1, out_coord2, ...))
        signature_freq = defaultdict(lambda: defaultdict(int))

        # 遍历每个快照（即每次保存的图结构）
        for snap in snapshots:
            graph = snap.graph_processor.graph
            # 遍历图中每个节点
            for node_id in graph.nodes():
                try:
                    # 提取该节点的拓扑签名（基于层号和边的 data_coord）
                    sig = self._extract_node_signature(graph, node_id)
                    # 获取该节点当前使用的方法名，若缺失则标记为 'unknown'
                    method = graph.nodes[node_id].get('method_name', 'unknown')
                    # 累加该方法在该拓扑位置出现的次数
                    signature_freq[sig][method] += 1
                except ValueError:
                    # 跳过无法解析的节点（如 "root"、"collapse" 等特殊节点）
                    continue

        # Step 2: 为每个拓扑位置选择高频方法（构建共识）（可以按照不同节点方法构建多个共识图，但目前来说没有必要）
        consensus_map = {}
        for sig, method_counts in signature_freq.items():
            total = sum(method_counts.values())
            max_method, max_count = max(method_counts.items(), key=lambda x: x[1])
            if self.consensus_threshold is None or (max_count / total) >= self.consensus_threshold:
                consensus_map[sig] = max_method

        if self.verbose:
            logger.debug(f"Selected consensus methods for {len(consensus_map)} node roles.")

        # Step 3: 重建共识图（节点为 signature）
        template_graph = snapshots[0].graph_processor.graph
        consensus_graph = nx.MultiDiGraph()
        node_id_to_sig = {}

        # 添加节点
        for node_id in template_graph.nodes():
            if node_id in ("root", "collapse"):
                continue
            try:
                sig = self._extract_node_signature(template_graph, node_id)
                if sig in consensus_map:
                    # ✅ 获取原始节点的完整属性
                    orig_attrs = template_graph.nodes[node_id]
                    
                    # ✅ 构建新节点属性：至少包含 method_name 和 is_passthrough
                    new_attrs = {
                        'method_name': consensus_map[sig],
                        'is_passthrough': orig_attrs.get('is_passthrough', False),  # ← 关键修复！
                    }
                    
                    # 可选：保留其他你认为重要的属性
                    # 例如：
                    # if 'output_count' in orig_attrs:
                    #     new_attrs['output_count'] = orig_attrs['output_count']
                    
                    consensus_graph.add_node(sig, **new_attrs)
                    node_id_to_sig[node_id] = sig
            except ValueError:
                continue

        # 添加边
        for src_id, dst_id, edge_data in template_graph.edges(data=True):
            if src_id in node_id_to_sig and dst_id in node_id_to_sig:
                src_sig = node_id_to_sig[src_id]
                dst_sig = node_id_to_sig[dst_id]
                consensus_graph.add_edge(src_sig, dst_sig, **edge_data)

        # Step 4: 分割为连通分量
        connected_components = list(nx.weakly_connected_components(consensus_graph))

        comp_sizes = [len(comp) for comp in connected_components]
        if self.verbose:
            logger.debug(
                f"[Connectivity Analysis] Found {len(connected_components)} weakly connected components. "
                f"连通分量数量：{len(connected_components)}，各分量大小：{comp_sizes}"
            )

        # 过滤：只保留节点数 >= 阈值的连通子图
        connected_components = [
            comp for comp in connected_components 
            if len(comp) >= self.min_subgraph_size_for_evolution
        ]
        
        num_components = len(connected_components)

        if self.verbose:
            filtered_sizes = [len(comp) for comp in connected_components]
            logger.info(
                f"[Component Filtering] Kept {num_components} components (min size ≥ {self.min_subgraph_size_for_evolution}). "
                f"保留连通分量数量：{num_components}（最小尺寸 ≥ {self.min_subgraph_size_for_evolution}），尺寸列表：{filtered_sizes}"
            )


        if self.verbose:
            logger.debug(f"Consensus graph split into {num_components} connected components.")

        if num_components == 0:
            return {'methods_added': 0, 'new_method_names': []}

        # Step 5: 决定是否进行图同构聚类
        if enable_graph_isomorphism_clustering:
            # ===== 启用图匹配聚类 =====
            from networkx.algorithms.isomorphism import DiGraphMatcher

            unique_graphs = []      # 存储唯一子图
            graph_counts = []       # 对应出现次数

            for comp in connected_components:
                g = consensus_graph.subgraph(comp).copy()
                matched = False
                for i, rep_g in enumerate(unique_graphs):
                    nm = lambda n1, n2: n1.get('method_name') == n2.get('method_name')
                    em = None  # 或启用边属性匹配
                    if DiGraphMatcher(rep_g, g, node_match=nm, edge_match=em).is_isomorphic():
                        graph_counts[i] += 1
                        matched = True
                        break
                if not matched:
                    unique_graphs.append(g)
                    graph_counts.append(1)

            # 按频次排序，取 top-k
            # ===== 根据策略选择排序键 =====
            if subgraph_selection_policy == "most_frequent":
                # 仅按频次（原始行为）
                sort_key = lambda x: (x[1],)
            elif subgraph_selection_policy == "largest":
                # 频次优先，节点数次之（默认推荐）
                sort_key = lambda x: (x[1], x[0].number_of_nodes(), x[0].number_of_edges())
            elif subgraph_selection_policy == "smallest":
                # 频次优先，但节点数越小越好
                sort_key = lambda x: (x[1], -x[0].number_of_nodes(), -x[0].number_of_edges())
            elif subgraph_selection_policy == "balanced":
                # 频次为主，节点数为辅（同 largest，但显式命名）
                sort_key = lambda x: (x[1], x[0].number_of_nodes(), x[0].number_of_edges())
            else:
                raise ValueError(f"Unknown subgraph_selection_policy: {subgraph_selection_policy}")

            # 执行排序（频次高的在前，同频次按策略排）
            sorted_pairs = sorted(zip(unique_graphs, graph_counts), key=sort_key, reverse=True)

            top_k = sorted_pairs[:max_methods]
            selected_graphs = [g for g, _ in top_k]
            selected_counts = [c for _, c in top_k]

            if self.verbose:
                logger.debug(
                    f"Graph isomorphism clustering: {len(sorted_pairs)} unique structures; "
                    f"selected top-{len(selected_graphs)}."
                )
                logger.info(
                    f"[Graph Clustering] Identified {len(unique_graphs)} unique graph structures via isomorphism. "
                    f"图同构聚类结果：共发现 {len(unique_graphs)} 种唯一子图结构。"
                )
                for idx, (g, cnt) in enumerate(sorted_pairs):
                    logger.debug(
                        f"  Structure {idx+1}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, frequency = {cnt}. "
                        f"  结构 {idx+1}：{g.number_of_nodes()} 个节点，{g.number_of_edges()} 条边，出现频次 = {cnt}"
                    )
                logger.info(
                    f"[Subgraph Selection] Using policy '{subgraph_selection_policy}'. "
                    f"子图选择策略：'{subgraph_selection_policy}'"
                )
                logger.info(
                    f"[Top-K Selection] Selected top-{len(selected_graphs)} most frequent structures for evolution. "
                    f"最终选取：频次最高的前 {len(selected_graphs)} 个结构用于方法进化。"
                )

        else:
            # ===== 禁用聚类：每个连通分量独立处理 =====
            selected_graphs = [consensus_graph.subgraph(comp).copy() for comp in connected_components]
            selected_counts = [1] * len(selected_graphs)  # 无频次意义，统一为1
            # 但仍受 max_methods 限制
            if len(selected_graphs) > max_methods:
                selected_graphs = selected_graphs[:max_methods]
                selected_counts = selected_counts[:max_methods]
                if self.verbose:
                    logger.info(
                        f"[No Clustering] Clustering disabled; using first {max_methods} components out of {len(connected_components)}. "
                        f"未启用聚类：从 {len(connected_components)} 个分量中直接选取前 {max_methods} 个。"
                    )

        # Step 6: 对 selected_graphs 还原（添加 root/collapse + 重命名）（未完成）
        reconstructed_graphs = []

        # 构建反向映射：signature -> 原始 node_id（任选一个即可，因签名相同）
        sig_to_original_node = {}
        for node_id, sig in node_id_to_sig.items():
            if sig not in sig_to_original_node:
                sig_to_original_node[sig] = node_id

        for g in selected_graphs:
            g_full = g.copy()
            g_full.add_node("root")
            g_full.add_node("collapse")
            
            internal_nodes = [n for n in g_full.nodes() if n not in ("root", "collapse")]
            entry_nodes = [n for n in internal_nodes if g_full.in_degree(n) == 0]
            exit_nodes = [n for n in internal_nodes if g_full.out_degree(n) == 0]
            
            # === 连接 root 到 entry_nodes，使用原始边属性 ===
            edge_counter = 0

            for sig in entry_nodes:
                orig_node = sig_to_original_node.get(sig)
                
                if orig_node is None or not list(template_graph.in_edges(orig_node, data=True)):
                    logger.warning("理论上不该进入该分支，请联系开发者。")
                    g_full.add_edge("root", sig,
                                    data_type='scalar',
                                    output_index=edge_counter,
                                    data_coord=edge_counter)
                    edge_counter += 1
                    continue

                # 构建并排序入边列表
                edge_info_list = []
                for u, v, attrs in template_graph.in_edges(orig_node, data=True):
                    if 'data_coord' not in attrs or 'data_type' not in attrs:
                        raise ValueError(f"Edge {u}->{v} missing required attrs: {attrs}")
                    layer_u = self.get_node_layer(u)
                    edge_info_list.append((layer_u, attrs['data_coord'], u, v, attrs))
                
                edge_info_list.sort(key=lambda x: (x[0], x[1]))

                # 为每条原始入边创建一条 root -> sig 的边
                for (_, _, u, v, orig_attrs) in edge_info_list:
                    g_full.add_edge("root", sig,
                                    data_type=orig_attrs['data_type'],
                                    output_index=edge_counter,
                                    data_coord=edge_counter)
                    edge_counter += 1
            
            # === 连接 exit_nodes 到 collapse，使用原始边属性并重新排序 ===
            # === 全局收集所有 exit -> collapse 的候选边 ===
            all_exit_edges = []  # (sig, orig_output_index, data_type, layer_u, orig_data_coord)

            for sig in exit_nodes:
                orig_node = sig_to_original_node.get(sig)
                if orig_node is None:
                    logger.warning("理论上不该进入该分支，请联系开发者。")
                    raise ValueError(f"No original node found for exit signature: {sig}")

                out_edges = list(template_graph.out_edges(orig_node, data=True))
                if not out_edges:
                    logger.warning("理论上不该进入该分支，请联系开发者。")
                    # 添加默认输出（注意：这也应纳入全局编号！）
                    all_exit_edges.append((sig, 0, 'scalar', self.get_node_layer(orig_node), 0))
                    continue

                for u, v, attrs in out_edges:
                    if not {'data_type', 'data_coord', 'output_index'} <= attrs.keys():
                        raise ValueError(f"Edge {u}->{v} missing required attrs: {attrs}")
                    layer_u = self.get_node_layer(u)
                    all_exit_edges.append((
                        sig,
                        attrs['output_index'],
                        attrs['data_type'],
                        layer_u,
                        attrs['data_coord']
                    ))

            # === 全局排序：高层级优先，同层按原 data_coord 升序 ===
            all_exit_edges.sort(key=lambda x: (-x[3], x[4]))  # x[3]=layer_u, x[4]=orig_data_coord

            # === 全局分配 data_coord（0,1,2,...）并添加边 ===
            for data_coord, (sig, orig_output_index, data_type, _, _) in enumerate(all_exit_edges):
                g_full.add_edge(sig, "collapse",
                                output_index=orig_output_index,  # ✅ 保留原始输出索引
                                data_coord=data_coord,           # ✅ 全局唯一
                                data_type=data_type)
            
            # === 重命名节点：使用与 append_nx_layer 一致的命名格式 ===
            if self.verbose:
                logger.info("Renaming nodes...")
                logger.warning("图的重命名系统根据节点到root的路径层数进行命名，理论上不该出现问题，如出现问题，请联系开发者")

            # 在 g_full 中（已添加 root），计算每个节点到 root 的最短路径长度
            # 注意：g_full 是 DiGraph，边方向 root → node
            try:
                lengths = nx.single_source_shortest_path_length(g_full, "root")
            except nx.NetworkXError:
                # 如果 root 不连通（理论上不应发生）
                lengths = {node: 0 for node in g_full.nodes()}
                lengths["root"] = 0

            mapping = {}

            # 按方法名分组，为每个方法分配局部索引
            layer_method_counter = defaultdict(lambda: defaultdict(int))

            for node in internal_nodes:
                method = g_full.nodes[node].get('method_name', 'unknown')
                local_layer = lengths.get(node, 1)
                
                # 按 (layer, method) 计数
                local_index = layer_method_counter[local_layer][method]
                layer_method_counter[local_layer][method] += 1
                
                new_name = f"{local_layer}_{local_index}_{method}"
                mapping[node] = new_name

            g_full = nx.relabel_nodes(g_full, mapping)

            # === 修正：按边起点所在层重新分配 data_coord，确保每层内唯一且从0开始 ===
            # 提取所有非 root/collapse 的边（包括内部边和到 collapse 的边）
            edges_to_update = []
            layer_edges = defaultdict(list)  # layer -> list of (src, dst, edge_attrs)

            for src, dst, key, attrs in list(g_full.edges(keys=True, data=True)):
                # 跳过 root 的出边（它们的 data_coord 已在前面正确设置为 input_slot）
                if src == "root":
                    continue
                # 获取起点的 layer（从节点名解析）
                if src == "collapse":
                    continue  # collapse 不应有出边
                try:
                    layer = int(src.split('_')[0])
                except (ValueError, IndexError):
                    logger.warning(f"Cannot parse layer from node name: {src}. Skipping edge {src}->{dst}.")
                    continue
                layer_edges[layer].append((src, dst, key, attrs))

            # 对每一层的边重新分配 data_coord
            for layer, edge_list in layer_edges.items():
                # 可选：按目标节点名排序以保证确定性
                edge_list.sort(key=lambda x: x[1])  # 按 dst 节点名排序
                for new_coord, (src, dst, key, attrs) in enumerate(edge_list):
                    # 更新边的 data_coord
                    g_full.edges[src, dst, key]['data_coord'] = new_coord

            internal_nodes = [n for n in g_full.nodes() if n not in ("root", "collapse")]

            if self.verbose:
                logger.debug(
                    f"[Node Renaming] Renamed {len(internal_nodes)} internal nodes using layer-method-index scheme. "
                    f"节点重命名：已按 (层_局部索引_方法名) 格式重命名 {len(internal_nodes)} 个内部节点。"
                )

            reconstructed_graphs.append(g_full)


        # Step 7: 生成新方法并可选保存子图（未完成）
        new_method_names = []

        # 确定保存目录
        save_dir = evolved_methods_save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for i, (g_orig, g_full, count) in enumerate(zip(selected_graphs, reconstructed_graphs, selected_counts)):
            name = f"evolved_method_{len(adaptoflux_instance.methods) + i + 1}"

            # 推断输入/输出类型（从 root/collapse 边）
            input_types = [edge['data_type'] for _, _, edge in g_full.out_edges("root", data=True)]
            output_types = [edge['data_type'] for _, _, edge in g_full.in_edges("collapse", data=True)]

            meta = {
                'name': name,
                'occurrence_count': count,
                'subgraph_node_count': g_full.number_of_nodes(),
                'is_clustered': enable_graph_isomorphism_clustering,
                'selection_policy': subgraph_selection_policy,
                'evolved_at': datetime.now().isoformat(),
                'input_count': len(input_types),
                'input_types': input_types,
                'output_types': output_types,
                'output_count': len(output_types),
                'group': 'evolved',
                'weight': 1.0,
                'vectorized': True,
                'is_evolved': True
            }

            # 创建可调用对象
            evolved_method = EvolvedMethod(
                name=name,
                graph=g_full,
                methods_ref=adaptoflux_instance.methods,
                metadata=meta
            )

            # 注册到主类 methods（关键！）
            adaptoflux_instance.methods[name] = {
                'function': evolved_method,           # ← 可调用对象
                'input_count': meta['input_count'],
                'output_count': meta['output_count'],
                'input_types': meta['input_types'],
                'output_types': meta['output_types'],
                'group': meta['group'],
                'weight': meta['weight'],
                'vectorized': meta['vectorized'],
                'is_evolved': True
            }

            # 6. 【关键】重建 graph_processor 以识别新方法，并保留当前 layer 状态
            adaptoflux_instance.path_generator.methods = adaptoflux_instance.methods # 确保 methods 已更新
            current_layer = adaptoflux_instance.graph_processor.layer
            adaptoflux_instance.graph_processor = adaptoflux_instance._create_graph_processor(
                graph=adaptoflux_instance.graph
            )
            adaptoflux_instance.graph_processor.layer = current_layer

            new_method_names.append(name)

            # 保存到磁盘
            if save_dir is not None:
                evolved_method.save(save_dir)
                if self.verbose:
                    logger.info(f"[Saved] Evolved method {name} to {save_dir}")

        return {
            'methods_added': len(new_method_names),
            'new_method_names': new_method_names
        }

    def train(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        max_evo_cycles: int = 5,
        enable_early_stop: bool = True,      # ← 新增开关
        early_stop_eps: float = 1e-6,        # ← 建议改名避免和数值计算库的 eps 冲突
        save_model: bool = True,
        model_save_path: Optional[str] = None,
        save_best_model: bool = True,
        best_model_subfolder: str = "best",
        final_model_subfolder: str = "final",
        subgraph_selection_policy: str = "largest",
        skip_initialization: bool = False,
        max_total_refinement_attempts: Optional[int] = None,  # <-- 新增参数
        **kwargs
    ) -> dict:
        """
        实现基类的 train 方法。
        执行完整的“图演”优化循环。

        :param input_data: 用于快速评估的输入数据（小批量）
        :param target: 对应的标签
        :param max_evo_cycles: 最多执行的训练轮次数。
        :param save_model: 是否在训练结束后保存模型
        :param model_save_path: 模型保存的文件夹路径
        :param save_best_model: 是否保存过程中遇到的最佳模型
        :param best_model_subfolder: 最佳模型保存的子目录
        :param final_model_subfolder: 最终模型保存的子目录
        :param skip_initialization: 若为 True，则跳过多样初始化阶段，直接使用当前 adaptoflux 实例的图结构作为优化起点。
                            适用于 CombinedTrainer 或人工预设模型场景。
        :param kwargs: 其他可选参数
        :return: 一个包含训练过程信息的字典
        """

        self.max_total_refinement_attempts = max_total_refinement_attempts
        self._total_refinement_attempts = 0  # <-- 新增计数器

        # 后面可能编写单任务使用多个实例的知识提取（消耗性能更高，但提取出来的知识应该效果更好），以及多任务适配
        # 后面添加可选参数，控制该轮训练得到的新知识是否直接加入方法池，或保存为图使用。

        if self.verbose:
            init_msg = "Skipping diverse initialization" if skip_initialization else "Starting diverse initialization"
            logger.info(f"Starting GraphEvoTrainer. {init_msg}. Max evolution cycles: {max_evo_cycles}")

        results = {
            "evo_cycles_completed": 0,
            "phase_results": [],
            "total_refinement_steps": 0,
            "total_compressions": 0,
            "total_methods_evolved": 0,
            "best_accuracy": -1.0,
            "best_accuracy_cycle": -1,
            "best_model_snapshot": None,
            "best_model_cycle": -1
        }

        # === 阶段一：多样初始化（可选）===
        if not skip_initialization:
            init_result = self._phase_diverse_initialization(input_data, target)
            current_af = init_result['best_model']
            current_loss = init_result['best_loss']
            current_acc = init_result['best_accuracy']

            # 更新全局状态（即使跳过，后续也会用 current_af 覆盖）
            self.adaptoflux = current_af

            # 记录初始化结果
            if current_acc > results['best_accuracy']:
                results['best_accuracy'] = current_acc
                results['best_accuracy_cycle'] = 0
                results['best_model_snapshot'] = copy.deepcopy(current_af)
                results['best_model_cycle'] = 0

            results['phase_results'].append({
                'cycle': 0,
                'phase': 'initialization',
                'result': init_result
            })
        else:
            # 直接使用当前实例的模型
            current_af = copy.deepcopy(self.adaptoflux)
            current_loss = self._evaluate_loss(input_data, target)
            current_acc = self._evaluate_accuracy(input_data, target)

            # 初始化即为当前状态
            if current_acc > results['best_accuracy']:
                results['best_accuracy'] = current_acc
                results['best_accuracy_cycle'] = 0
                results['best_model_snapshot'] = copy.deepcopy(current_af)
                results['best_model_cycle'] = 0

            results['phase_results'].append({
                'cycle': 0,
                'phase': 'initialization_skipped',
                'result': {
                    'loss': current_loss,
                    'accuracy': current_acc,
                    'model_used': 'current_adaptoflux'
                }
            })

        # 开始进化循环
        for cycle in range(1, max_evo_cycles + 1):
            if self.verbose:
                logger.info(f"--- Starting Evolution Cycle {cycle}/{max_evo_cycles} ---")

            cycle_results = {'cycle': cycle}

            # 阶段二：逐节点精炼
            refinement_result = self._phase_node_refinement(current_af, input_data, target)
            current_af = refinement_result['final_model']
            current_loss = refinement_result['final_loss']
            current_acc = refinement_result['final_accuracy']

            # 【✅ 新增早停逻辑】
            if enable_early_stop and current_acc >= 1.0 - early_stop_eps:
                if self.verbose:
                    logger.info(f"🎯 Early stopping triggered at cycle {cycle}: accuracy={current_acc:.6f} >= {1.0 - eps}")
                results['evo_cycles_completed'] = cycle
                break  # 立即终止后续进化轮次

            results['total_refinement_steps'] += refinement_result['steps_taken']

            cycle_results['refinement'] = refinement_result

            # 阶段三：方法池进化（基于快照触发）
            if self.enable_evolution:
                # 1. 按频率记录图快照
                if cycle % self.evolution_sampling_frequency == 0:
                    snapshot = copy.deepcopy(current_af)
                    self.graph_snapshots.append(snapshot)
                    if self.verbose:
                        logger.debug(f"Saved graph snapshot #{len(self.graph_snapshots)} at cycle {cycle}")

                # 2. 检查是否触发进化
                if len(self.graph_snapshots) >= self.evolution_trigger_count:

                    evolved_dir = os.path.join(model_save_path, "evolved_methods")

                    # 3. 执行进化
                    evolution_result = self._phase_method_pool_evolution(
                        current_af,
                        snapshots=self.graph_snapshots,
                        max_methods=self.methods_per_evolution,
                        evolved_methods_save_dir=evolved_dir,
                        subgraph_selection_policy=subgraph_selection_policy
                    )
                    results['total_methods_evolved'] += evolution_result['methods_added']
                    cycle_results['evolution'] = evolution_result

                    # 4. 清理快照
                    if self.evolution_cleanup_mode == "full":
                        self.graph_snapshots.clear()
                    elif self.evolution_cleanup_mode == "oldest":
                        if self.graph_snapshots:
                            self.graph_snapshots.pop(0)
                else:
                    cycle_results['evolution'] = {
                        'skipped': True,
                        'reason': 'insufficient_snapshots'
                    }
            else:
                cycle_results['evolution'] = {
                    'skipped': True,
                    'reason': 'disabled'
                }

            # 更新全局状态
            self.adaptoflux = current_af

            # 检查是否为新的最佳模型
            if current_acc > results['best_accuracy']:
                results['best_accuracy'] = current_acc
                results['best_accuracy_cycle'] = cycle
                results['best_model_snapshot'] = copy.deepcopy(current_af)
                results['best_model_cycle'] = cycle

            results['evo_cycles_completed'] += 1
            results['phase_results'].append(cycle_results)

            if self.verbose:
                logger.info(f"--- Cycle {cycle} completed. Current Acc: {current_acc:.4f}, Best Acc: {results['best_accuracy']:.4f} ---")

        if self.verbose:
            logger.info(f"GraphEvoTrainer finished after {results['evo_cycles_completed']} cycles.")


        # === 阶段四（后处理）：模块化压缩（仅执行一次）===
        final_compression_result = None
        if self.enable_compression:
            if self.verbose:
                logger.info("[Post-Training] Applying Modular Compression once on final model...")
                logger.warning(f"该函数（_phase_modular_compression）仍在开发中，大概率无法达到预期效果，不建议开启，如需使用建议在ATF.ModelTrainer.GraphEvoTrainer中修改或使用规则化方法替代")
            final_compression_result = self._phase_modular_compression(self.adaptoflux, input_data, target)
            # 更新最终模型
            self.adaptoflux = final_compression_result['final_model']
            results['total_compressions'] = final_compression_result['compressed_subgraphs']
        else:
            if self.verbose:
                logger.info("[Post-Training] Modular Compression: SKIPPED (disabled by enable_compression=False)")
            results['total_compressions'] = 0
        
        if final_compression_result:
            results['final_compression_applied'] = final_compression_result['compression_applied']
        else:
            results['final_compression_applied'] = False

        # 保存模型
        if save_model:
            try:
                base_save_path = model_save_path or "models"
                os.makedirs(base_save_path, exist_ok=True)

                # 保存最终模型
                final_path = os.path.join(base_save_path, final_model_subfolder)
                self.adaptoflux.save_model(folder=final_path)
                if self.verbose:
                    logger.info(f"Final model saved to '{final_path}'")
                results["final_model_saved"] = final_path

                # 保存最佳模型
                if save_best_model and results['best_model_snapshot'] is not None:
                    best_path = os.path.join(base_save_path, best_model_subfolder)

                    # 临时切换
                    original_af = self.adaptoflux
                    self.adaptoflux = results['best_model_snapshot']
                    try:
                        self.adaptoflux.save_model(folder=best_path)
                        if self.verbose:
                            logger.info(f"Best model (Cycle {results['best_model_cycle']}, Acc={results['best_accuracy']:.4f}) saved to '{best_path}'")
                    finally:
                        self.adaptoflux = original_af

                    results["best_model_saved"] = best_path

                # 保存训练日志
                log_filename = kwargs.get("log_filename", "graph_evo_training_log.json")
                log_path = os.path.join(base_save_path, log_filename)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, default=str)
                results["training_log_saved"] = log_path

            except Exception as e:
                logger.error(f"Failed to save model(s): {e}")
                import traceback
                logger.error(traceback.format_exc())
                
        # 记录总精炼尝试次数（无论是否成功）
        results['total_refinement_attempts'] = self._total_refinement_attempts

        return results
    
    def _refine_random_single_step(
        self,
        adaptoflux_instance,
        input_data: np.ndarray,
        target: np.ndarray,
        processing_nodes: List[str],
        current_loss: float,
        gp: Any  # GraphProcessor 实例
    ) -> Tuple[bool, float, float, int, List[str]]:
        """
        【策略：随机单点优化】
        在当前图中随机选择一个可优化的处理节点，尝试将其方法替换为所有兼容方法中的最优者（基于损失下降）。
        仅执行一次替换尝试（即一个节点的一次优化），适用于轻量级、低开销的局部搜索。
        注意：本函数会 in-place 修改 adaptoflux_instance 的图结构！

        返回值说明：
        - bool: 本轮是否成功改进（即找到更优方法）
        - float: 改进后的损失值（若未改进则返回原损失）
        - float: 改进后的准确率（若未改进则返回 0.0，调用方应忽略）
        - int: 本次实际执行的优化步数（0 或 1）
        - List[str]: 更新后的可处理节点列表（因图结构可能变化，需重新获取）

        :param adaptoflux_instance: 当前待优化的 AdaptoFlux 实例
        :param input_data: 用于评估的小批量输入数据
        :param target: 对应的真实标签
        :param processing_nodes: 当前图中所有可被优化的处理节点名称列表（已排除冻结节点）
        :param current_loss: 当前模型的损失值（用于比较）
        :param gp: 图处理器实例，用于访问和修改图结构
        :return: (是否改进, 新损失, 新准确率, 步数增量, 更新后的节点列表)
        """
        # 若无可优化节点，直接返回无改进
        if not processing_nodes:
            return False, current_loss, 0.0, 0, processing_nodes

        # 随机选择一个待优化节点
        target_node = random.choice(processing_nodes)
        original_method_name = gp.graph.nodes[target_node]['method_name']
        
        # 获取与该节点兼容的候选方法列表（基于组别或类型匹配）
        candidate_methods = self._get_compatible_methods_for_node(
            adaptoflux_instance, 
            target_node
        )

        best_candidate = None
        best_loss = current_loss  # 初始化为当前损失，用于比较

        attempts_in_step = 0
        
        # 遍历所有候选方法（跳过当前方法）
        for candidate_method_name in candidate_methods:
            if candidate_method_name == original_method_name:
                continue

            # --- 检查是否已达总尝试上限 ---
            if (self.max_total_refinement_attempts is not None and
                self._total_refinement_attempts >= self.max_total_refinement_attempts):
                break

            # --- 计数 +1 ---
            self._total_refinement_attempts += 1
            attempts_in_step += 1

            # 创建临时副本，避免污染原模型
            temp_af = copy.deepcopy(adaptoflux_instance)
            temp_gp = temp_af.graph_processor
            # 尝试替换节点方法
            temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

            # 评估替换后的损失
            new_loss = self._evaluate_loss(input_data, target, adaptoflux_instance=temp_af)


            # 记录损失更低的最优候选
            if new_loss < best_loss:
                best_loss = new_loss
                best_candidate = candidate_method_name

        # 如果找到更优方法，则应用到原图
        if best_candidate and best_loss < current_loss:
            # === 替换节点（自动更新 ID 和边） ===
            new_node_id = gp.replace_node_method(target_node, best_candidate)
            
            # 注意：target_node 已被删除，后续操作应使用 new_node_id（但本策略不需要）
            # 如果作者有空而且没忘记可能会在gp里面加一个刷新图节点id的方法提升可读性

            # 重新评估准确率
            new_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

            # 重新获取处理节点列表（因为节点 ID 已变）
            updated_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
            
            return True, best_loss, new_acc, 1, updated_nodes
        
        else:
            # 无改进，返回原始状态
            return False, current_loss, 0.0, 0, processing_nodes

    def _refine_full_sweep_step(
        self,
        adaptoflux_instance,
        input_data: np.ndarray,
        target: np.ndarray,
        processing_nodes: List[str],
        current_loss: float,
        gp: Any
    ) -> Tuple[bool, float, float, int, List[str]]:
        """
        【策略：完整遍历优化】
        对当前图中所有可优化的处理节点进行一轮完整遍历。
        对每个节点，尝试所有兼容方法，若发现能降低损失的替换，则立即应用（贪心策略）。
        一轮中可能多次修改图结构，适用于更彻底的局部优化，但计算开销较大。
        注意：本函数会 in-place 修改 adaptoflux_instance 的图结构！

        返回值说明：
        - bool: 本轮是否至少有一次成功改进
        - float: 本轮结束后的最终损失值
        - float: 本轮结束后的最终准确率
        - int: 本轮总共执行的方法替换次数
        - List[str]: 最终的可处理节点列表（可能因方法变更而动态变化）

        :param adaptoflux_instance: 当前待优化的 AdaptoFlux 实例
        :param input_data: 用于评估的小批量输入数据
        :param target: 对应的真实标签
        :param processing_nodes: 初始的可优化节点列表
        :param current_loss: 当前损失（作为起点）
        :param gp: 图处理器实例
        :return: (是否改进, 最终损失, 最终准确率, 替换次数, 最终节点列表)
        """
        if not processing_nodes:
            return False, current_loss, 0.0, 0, processing_nodes

        # 随机打乱节点顺序，避免顺序偏差（例如总是先优化靠前的节点）
        nodes_to_try = random.sample(processing_nodes, len(processing_nodes))
        improvement_made = False
        total_replacements = 0
        current_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

        # 遍历每一个待优化节点
        for target_node in nodes_to_try:
            # 检查节点是否仍然存在于图中（可能被之前的替换操作删除或重命名）
            if target_node not in gp.graph.nodes:
                if self.verbose:
                    logger.debug(f"Node '{target_node}' no longer exists in graph; skipping.")
                continue

            original_method_name = gp.graph.nodes[target_node].get('method_name')
            if original_method_name is None:
                if self.verbose:
                    logger.warning(f"Node '{target_node}' has no 'method_name'; skipping.")
                continue

            # 获取与该节点兼容的候选方法列表（考虑组别、类型兼容性及冻结规则）
            candidate_methods = self._get_compatible_methods_for_node(
                adaptoflux_instance,
                target_node
            )

            best_candidate = None
            best_loss = current_loss  # 以当前全局损失为基准进行比较

            # 尝试所有兼容方法（跳过当前方法）
            for candidate_method_name in candidate_methods:
                if candidate_method_name == original_method_name:
                    continue

                # --- 检查并计数 ---
                if (self.max_total_refinement_attempts is not None and
                    self._total_refinement_attempts >= self.max_total_refinement_attempts):
                    # 提前退出双层循环
                    return improvement_made, current_loss, current_acc, total_replacements, processing_nodes

                self._total_refinement_attempts += 1

                # 创建临时副本进行安全评估，避免污染原模型
                try:
                    temp_af = copy.deepcopy(adaptoflux_instance)
                    temp_gp = temp_af.graph_processor
                    # 安全替换：使用图处理器的标准方法（会处理输入/输出类型变化）
                    # 注意：这里我们模拟替换，但不实际调用 replace_node_method（因为只是评估）
                    # 所以直接修改 method_name 是安全的，前提是不依赖边结构变化
                    temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

                    new_loss = self._evaluate_loss(input_data, target, adaptoflux_instance=temp_af)

                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_candidate = candidate_method_name
                except Exception as e:
                    logger.warning(f"Failed to evaluate candidate method '{candidate_method_name}' "
                                   f"for node '{target_node}': {e}")
                    continue

            # 如果找到更优方法，立即应用到原图（贪心策略）
            if best_candidate and best_loss < current_loss:
                try:
                    # 使用图处理器的标准替换方法，确保图结构一致性（如边更新、ID刷新等）
                    new_node_id = gp.replace_node_method(target_node, best_candidate)
                    # 更新当前损失和准确率
                    current_loss = best_loss
                    current_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)
                    improvement_made = True
                    total_replacements += 1

                    if self.verbose:
                        logger.info(f"  Replacement {total_replacements}: Node '{target_node}' "
                                    f"{original_method_name} → {best_candidate}, Loss: {current_loss:.6f}")

                    # 重要：节点替换后，原 target_node 可能已被删除或重命名（new_node_id）
                    # 因此需要重新获取当前所有处理节点，确保后续操作基于最新图状态
                    processing_nodes = [
                        node for node in gp.graph.nodes()
                        if gp._is_processing_node(node)
                    ]
                    # 注意：nodes_to_try 是旧列表，但仅包含节点名，后续节点若仍存在仍可处理；
                    # 若需更严格的一致性，可考虑 break 并重启本轮，但会增加开销，此处暂不处理。

                except Exception as e:
                    logger.error(f"Failed to apply method replacement for node '{target_node}': {e}")
                    # 替换失败，跳过该节点，继续优化其他节点
                    continue

        return improvement_made, current_loss, current_acc, total_replacements, processing_nodes
        
    def build_candidate_pool(compatibility_mode, methods, original_method_name, all_method_names):
        if compatibility_mode == "all":
            return all_method_names
        else:
            node_group = methods[original_method_name].get("group", "default")
            group_methods = [
                name for name, info in methods.items()
                if info.get("group", "default") == node_group
            ]
            if compatibility_mode == "group_only":
                return group_methods if group_methods else all_method_names[:1]
            else:  # group_with_fallback
                return group_methods if len(group_methods) >= 2 else all_method_names

    def get_node_layer(self, node_id: str) -> int:
        if node_id in ("root", "collapse"):
            return -1  # 特殊节点设为 -1（优先级最高）
        try:
            return int(node_id.split('_')[0]) # 感觉获取节点再获取元属性可能更慢，先这样吧
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse layer from node_id: {node_id}. Using layer=0.")
            return 0