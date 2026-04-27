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

from .method_pool_evolver import MethodPoolEvolver

from .refinement_strategies import (
    refine_random_single_step,
    refine_full_sweep_step,
    refine_multi_node_joint_step,
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
        # 父类参数
        adaptoflux_instance,
        loss_fn='mse',
        task_type='regression',
        use_pipeline=False,      # ← 新增
        num_workers=4,           # ← 新增
        custom_loss_evaluator=None,      # ← 新增：自定义损失评估器
        custom_accuracy_evaluator=None,   # ← 新增：自定义准确率评估器
        acceptance_strategy=None,  # ← 新增：自定义接受策略

        # 本类参数
        num_initial_models: int = 5,
        max_refinement_steps: int = 100,
        compression_threshold: float = 0.95,
        max_init_layers: int = 3,
        init_mode: str = "fixed",
        init_layers_list: Optional[List[int]] = None,
        frozen_nodes: Optional[List[str]] = None,
        frozen_methods: Optional[List[str]] = None,
        refinement_strategy: str = "random_single",
        custom_refinement_strategy_func: Optional[callable] = None,  # ← 新增
        candidate_pool_mode: str = "group",
        fallback_mode: Optional[str] = None,
        enable_evolution: bool = True,
        evolution_sampling_frequency: int = 1,
        evolution_trigger_count: int = 3,
        evolution_cleanup_mode: str = "full",
        consensus_threshold: Optional[float] = None,  # <-- 新增
        methods_per_evolution: int = 1,
        min_subgraph_size_for_evolution: int = 2,  # <-- 新增参数

        verbose: bool = True,

        enable_compression: bool = False,
        compression_mode: str = "symbolic",  # 新增：默认为 symbolic
        symbolic_compression_rules: Optional[List[Tuple[Any, Any]]] = None,  # 新增

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
        :param refinement_strategy: 精炼策略，可选值：
            - "random_single"：随机单点优化；
            - "full_sweep"：全图遍历优化；
            - "multi_node_joint"：多节点联合优化（未实现，占位）；
            - "custom"：使用 custom_refinement_strategy_func 提供的函数；
            - 或直接传入一个 callable 函数对象。
        :param custom_refinement_strategy_func: 当 refinement_strategy="custom" 时，
            提供的自定义策略函数。函数签名需与 `_refine_random_single_step` 一致。
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

        super().__init__(adaptoflux_instance, loss_fn, task_type, use_pipeline, num_workers,
                         custom_loss_evaluator, custom_accuracy_evaluator, acceptance_strategy)
        self.method_pool_evolver = MethodPoolEvolver(self)
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
        self.custom_refinement_strategy_func = custom_refinement_strategy_func

        self.enable_evolution = enable_evolution
        self.evolution_sampling_frequency = evolution_sampling_frequency
        self.evolution_trigger_count = evolution_trigger_count
        self.evolution_cleanup_mode = evolution_cleanup_mode

        self.methods_per_evolution = methods_per_evolution
        self.verbose = verbose

        # 模块化压缩相关
        self.enable_compression = enable_compression
        self.compression_mode = compression_mode
        self.consensus_threshold = consensus_threshold
        self.symbolic_compression_rules = symbolic_compression_rules or []

        # 校验 compression_mode
        if self.compression_mode not in {"symbolic", "numerical"}:
            raise ValueError("compression_mode must be 'symbolic' or 'numerical'")
            

        self.subgraph_sampler = kwargs.get('subgraph_sampler')
        self.io_extractor = kwargs.get('io_extractor')
        # self.replacer = kwargs.get('replacer') or SubgraphReplacer()
        self.equivalence_checker = kwargs.get('equivalence_checker')

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
        
        # 原有 _strategy_map
        self._strategy_map = {
            "random_single": refine_random_single_step,
            "full_sweep": refine_full_sweep_step,
            "multi_node_joint": refine_multi_node_joint_step,
            # TODO: 多节点联合优化（见下文）
        }

        # 新增：支持自定义函数
        if custom_refinement_strategy_func is not None:
            self._strategy_map["custom"] = custom_refinement_strategy_func
            if refinement_strategy == "custom":
                pass  # 合法
        else:
            if refinement_strategy == "custom":
                raise ValueError("custom strategy requires custom_refinement_strategy_func")

        # 校验
        valid_pool_modes = {"all", "group", "self"}
        valid_fallback_modes = {"all", "group_first", "self", "error"}
        if self.candidate_pool_mode not in valid_pool_modes:
            raise ValueError(f"Invalid candidate_pool_mode: {self.candidate_pool_mode}")
        if self.fallback_mode not in valid_fallback_modes:
            raise ValueError(f"Invalid fallback_mode: {self.fallback_mode}")


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
                self, adaptoflux_instance, input_data, target, processing_nodes, current_loss, gp
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
        if not self.enable_compression:
            return self._return_original_result(adaptoflux_instance, input_data, target)

        if self.compression_mode == "symbolic":
            return self._phase_symbolic_compression(adaptoflux_instance, input_data, target)
        elif self.compression_mode == "numerical":
            if self.verbose:
                logger.warning(
                    "⚠️ Numerical modular compression is UNFINISHED and UNSAFE. "
                    "It uses hard-coded 'add' replacement and is likely broken. "
                    "Use symbolic compression with explicit rules instead."
                )
            return self._phase_numerical_compression_legacy(adaptoflux_instance, input_data, target)
        else:
            raise ValueError(f"Unknown compression_mode: {self.compression_mode}")

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

    def _phase_method_pool_evolution(
        self,
        adaptoflux_instance,
        snapshots: List[Any],
        max_methods: int = 1,
        enable_graph_isomorphism_clustering: bool = True,
        evolved_methods_save_dir: Optional[str] = None,
        subgraph_selection_policy: str = "largest"
    ) -> Dict[str, Any]:
        """
        薄层包装器：保持 API 兼容性，委托给 MethodPoolEvolver
        Thin wrapper: maintains API compatibility, delegates to MethodPoolEvolver
        """
        return self.method_pool_evolver.evolve(
            adaptoflux_instance=adaptoflux_instance,
            snapshots=snapshots,
            max_methods=max_methods,
            enable_graph_isomorphism_clustering=enable_graph_isomorphism_clustering,
            evolved_methods_save_dir=evolved_methods_save_dir,
            subgraph_selection_policy=subgraph_selection_policy
        )

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
                    logger.info(f"🎯 Early stopping triggered at cycle {cycle}: accuracy={current_acc:.6f} >= {1.0 - early_stop_eps}")
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

        # === 显式记录最终模型和最佳模型的 loss/acc 到 results 根层级 ===
        # 1. 最终模型指标（使用当前 self.adaptoflux）
        final_loss = self._evaluate_loss(input_data, target)
        final_acc = self._evaluate_accuracy(input_data, target)
        results['final_loss'] = final_loss
        results['final_accuracy'] = final_acc

        # 2. 最佳模型指标（如果存在）
        if results['best_model_snapshot'] is not None:
            best_af_temp = self.adaptoflux  # 临时保存
            self.adaptoflux = results['best_model_snapshot']
            try:
                best_loss = self._evaluate_loss(input_data, target)
                best_acc = self._evaluate_accuracy(input_data, target)
                results['best_loss'] = best_loss
                results['best_accuracy'] = best_acc  # 可能已存在，但确保一致性
            finally:
                self.adaptoflux = best_af_temp  # 恢复
        else:
            results['best_loss'] = float('inf')
            results['best_accuracy'] = -1.0
            
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

    def _phase_numerical_compression_legacy(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        【DEPRECATED】旧的数值压缩逻辑（MSE + 单节点替换）。
        仅用于兼容性保留，不推荐使用。
        """
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
        temp_af = self._create_single_node_graph(adaptoflux_instance, candidate_method)
        rep_output = temp_af.infer_with_graph(input_data)
        orig_output = list(sub_outputs.values())[0]
        if not self.equivalence_checker.is_equivalent(orig_output, rep_output):
            return self._return_original_result(adaptoflux_instance, input_data, target)
        # 5. 执行替换
        try:
            new_node_id = self.replacer.replace_with_node(gp, subgraph, candidate_method)
            logger.info(f"Replaced subgraph with node '{new_node_id}' ({candidate_method})")
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

    def _phase_symbolic_compression(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        if not self.symbolic_compression_rules:
            if self.verbose:
                logger.info("Symbolic compression skipped: no rules provided.")
            return self._return_original_result(adaptoflux_instance, input_data, target)

        gp = adaptoflux_instance.graph_processor
        graph = gp.graph
        total_compressed = 0

        from networkx.algorithms.isomorphism import DiGraphMatcher

        for src_subgraph, tgt_spec in self.symbolic_compression_rules:
            root_ph = "root"
            collapse_ph = "collapse"

            # === 1. 预处理 pattern：分离内部图 + 接口规范 ===
            if root_ph not in src_subgraph or collapse_ph not in src_subgraph:
                logger.warning("Pattern must contain 'root' and 'collapse'")
                continue

            # 内部图（用于匹配）
            internal_nodes_pattern = [n for n in src_subgraph.nodes() if n not in (root_ph, collapse_ph)]
            match_graph = src_subgraph.subgraph(internal_nodes_pattern).copy()

            # 输入端口规范: port_name -> (target_node, input_slot)
            input_ports = {}
            for _, tgt, data in src_subgraph.out_edges(root_ph, data=True):
                port_name = data.get('port_name')
                input_slot = data.get('input_slot')
                if port_name is None or input_slot is None:
                    raise ValueError("Pattern input edge must have 'port_name' and 'input_slot'")
                if port_name in input_ports:
                    raise ValueError(f"Duplicate input port: {port_name}")
                input_ports[port_name] = (tgt, input_slot)

            # 输出端口规范: port_name -> (source_node, output_index)
            output_ports = {}
            for src, _, data in src_subgraph.in_edges(collapse_ph, data=True):
                port_name = data.get('port_name')
                output_index = data.get('output_index')
                if port_name is None or output_index is None:
                    raise ValueError("Pattern output edge must have 'port_name' and 'output_index'")
                if port_name in output_ports:
                    raise ValueError(f"Duplicate output port: {port_name}")
                output_ports[port_name] = (src, output_index)

            # === 2. 匹配内部图 ===
            def node_match(n1, n2):
                return n1.get('method_name') == n2.get('method_name')
            
            def edge_match(e1, e2):
                return (
                    e1.get('output_index') == e2.get('output_index') and
                    e1.get('data_type') == e2.get('data_type')
                )

            matcher = DiGraphMatcher(graph, match_graph, node_match=node_match, edge_match=edge_match)
            matches = list(matcher.subgraph_isomorphisms_iter())
            if self.verbose:
                print(f"Found {len(matches)} matches for compression rule.")
            if not matches:
                continue

            # 贪心非重叠
            used_nodes = set()
            valid_matches = []
            for mapping in sorted(matches, key=lambda m: -len(m)):
                main_nodes = set(mapping.keys())
                if main_nodes & used_nodes:
                    continue
                valid_matches.append(mapping)
                used_nodes.update(main_nodes)

            # === 3. 处理每个匹配 ===
            for mapping in valid_matches:
                subgraph_nodes = set(mapping.keys())

                # 🔧 关键修复：构建 pattern_node -> host_node 的反向映射
                pattern_to_host = {pattern_node: host_node for host_node, pattern_node in mapping.items()}

                # --- 构建 input_port_bindings ---
                input_bindings = {}
                for port_name, (pattern_tgt, expected_slot) in input_ports.items():
                    # ✅ 使用反向映射获取 host 节点
                    host_tgt = pattern_to_host[pattern_tgt]
                    candidates = []
                    for src, _, key, data in graph.in_edges(host_tgt, keys=True, data=True):
                        if src in subgraph_nodes:
                            continue
                        if data.get('input_slot') == expected_slot:
                            candidates.append((src, key, data))
                    if len(candidates) != 1:
                        raise ValueError(f"Input port {port_name} (slot={expected_slot}): expected 1 edge, got {len(candidates)}")
                    input_bindings[port_name] = candidates[0]

                # --- 构建 output_port_bindings ---
                output_bindings = {}
                for port_name, (pattern_src, expected_idx) in output_ports.items():
                    # ✅ 使用反向映射获取 host 节点
                    host_src = pattern_to_host[pattern_src]
                    candidates = []
                    for _, dst, key, data in graph.out_edges(host_src, keys=True, data=True):
                        if dst in subgraph_nodes:
                            continue
                        if data.get('output_index') == expected_idx:
                            candidates.append((dst, key, data))
                    if len(candidates) != 1:
                        raise ValueError(f"Output port {port_name} (output={expected_idx}): expected 1 edge, got {len(candidates)}")
                    output_bindings[port_name] = candidates[0]

                # === 4. 调用替换 ===
                try:
                    if isinstance(tgt_spec, nx.DiGraph):
                        new_nodes = gp.replace_subgraph_with_graph(
                            subgraph_nodes=subgraph_nodes,
                            replacement_graph=tgt_spec,
                            input_port_bindings=input_bindings,
                            output_port_bindings=output_bindings
                        )
                        logger.info(f"✅ Replaced subgraph with graph: {len(new_nodes)} nodes inserted")
                        total_compressed += 1
                    else:
                        logger.warning(f"Unsupported target spec type: {type(tgt_spec)}")
                except Exception as e:
                    logger.warning(f"Compression failed: {e}", exc_info=True)

        # 评估
        new_loss = self._evaluate_loss(input_data, target, adaptoflux_instance=adaptoflux_instance)
        new_acc = self._evaluate_accuracy(input_data, target, adaptoflux_instance=adaptoflux_instance)

        return {
            'final_model': adaptoflux_instance,
            'final_loss': new_loss,
            'final_accuracy': new_acc,
            'compression_applied': total_compressed > 0,
            'compressed_subgraphs': total_compressed
        }

    