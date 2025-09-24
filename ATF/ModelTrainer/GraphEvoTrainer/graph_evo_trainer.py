# graph_evo_trainer.py
from ..model_trainer import ModelTrainer
import numpy as np
import logging
import random
import copy
from typing import Optional, List, Dict, Any, Tuple
import os
import json
from ...PathGenerator.path_generator import PathGenerator
from ...GraphManager.graph_processor import GraphProcessor

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
        evolution_frequency: int = 3,
        max_init_layers: int = 3,
        init_mode: str = "fixed",
        init_layers_list: Optional[List[int]] = None,
        frozen_nodes: Optional[List[str]] = None,
        frozen_methods: Optional[List[str]] = None,
        refinement_strategy: str = "random_single",
        compatibility_mode: str = "group_with_fallback",  # <-- 新增
        verbose: bool = True
    ):
        """
        初始化 GraphEvoTrainer。

        :param adaptoflux_instance: 已初始化的 AdaptoFlux 对象
        :param num_initial_models: 在“多样初始化”阶段生成的候选模型数量
        :param max_refinement_steps: 在“逐节点精炼”阶段，单次优化的最大步数
        :param compression_threshold: 在“模块化压缩”阶段，子图等效性判定的相似度阈值 (0~1)
        :param evolution_frequency: 每进行多少次完整的“精炼-压缩”循环后，执行一次“方法池进化”
        :param max_init_layers: 在“多样初始化”阶段，每个候选模型最多随机添加的层数（仅在 fixed 模式生效）
        :param init_mode: 初始化模式，"fixed"=所有模型添加固定 max_init_layers 层数，"list"=按 init_layers_list 指定层数
        :param init_layers_list: 当 init_mode="list" 时，指定每个候选模型的层数列表，长度应 >= num_initial_models
        :param verbose: 是否打印详细日志
        """
        super().__init__(adaptoflux_instance)
        self.num_initial_models = num_initial_models
        self.max_refinement_steps = max_refinement_steps
        self.compression_threshold = compression_threshold
        self.evolution_frequency = evolution_frequency
        self.max_init_layers = max_init_layers
        self.init_mode = init_mode
        self.init_layers_list = init_layers_list
        self.frozen_nodes = set(frozen_nodes) if frozen_nodes else set()
        self.frozen_methods = set(frozen_methods) if frozen_methods else set()  # <-- 保存为集合
        self.compatibility_mode = compatibility_mode
        self.refinement_strategy = refinement_strategy
        self.verbose = verbose

        # 校验参数
        if self.init_mode == "list":
            if self.init_layers_list is None:
                raise ValueError("init_mode='list' requires init_layers_list to be provided.")
            if len(self.init_layers_list) < self.num_initial_models:
                raise ValueError(f"init_layers_list length ({len(self.init_layers_list)}) must be >= num_initial_models ({self.num_initial_models})")

        # 用于记录高性能子图，供“方法池进化”阶段使用
        self.high_performance_subgraphs: List[Dict[str, Any]] = []

        
        self._strategy_map = {
            "random_single": self._refine_random_single_step,
            "full_sweep": self._refine_full_sweep_step,
            # 未来可加："weighted_sample": self._refine_weighted_sample_step,
        }
        if self.refinement_strategy not in self._strategy_map:
            raise ValueError(f"Unknown refinement_strategy: {self.refinement_strategy}. "
                            f"Available: {list(self._strategy_map.keys())}")

        if self.compatibility_mode not in {"group_only", "group_with_fallback", "all"}:
            raise ValueError(f"Invalid compatibility_mode: {self.compatibility_mode}")

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
            loss = self._evaluate_loss_with_instance(candidate_af, input_data, target)
            acc = self._evaluate_accuracy_with_instance(candidate_af, input_data, target)

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
                    logger.warning(f"Failed to add random layer during initialization: {e}")
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
        current_loss = self._evaluate_loss_with_instance(adaptoflux_instance, input_data, target)
        current_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)

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
            'improvement_made': improvement_made     # 是否有改进
        }

    def _get_compatible_methods_for_node(
        self, 
        adaptoflux_instance, 
        node_name: str, 
        compatibility_mode: str = "group_with_fallback",
        allow_fallback_on_empty: bool = True
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
        supported_modes = {"group_only", "group_with_fallback", "all"}
        if compatibility_mode not in supported_modes:
            raise ValueError(
                f"Unsupported compatibility_mode: '{compatibility_mode}'. "
                f"Supported modes: {sorted(supported_modes)}"
            )

        gp = adaptoflux_instance.graph_processor
        methods = adaptoflux_instance.methods
        all_method_names = list(methods.keys())

        if not all_method_names:
            return []

        # === 1. 获取原始方法 ===
        node_data = gp.graph.nodes[node_name]
        original_method_name = node_data.get("method_name")
        if original_method_name is None or original_method_name not in methods:
            logger.warning(
                "Node '%s' has no valid 'method_name'; falling back to all methods.",
                node_name
            )
            return all_method_names

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
        if compatibility_mode == "all":
            candidate_pool = all_method_names
        else:
            node_group = node_data.get("group", "default")
            group_methods = [
                name for name, info in methods.items()
                if info.get("group", "default") == node_group
            ]
            if compatibility_mode == "group_only":
                candidate_pool = group_methods if group_methods else all_method_names[:1]
            else:  # group_with_fallback
                candidate_pool = group_methods if len(group_methods) >= 2 else all_method_names

        # === 4. 筛选类型兼容方法 ===
        compatible_methods = [name for name in candidate_pool if is_type_compatible(name)]

        # === 5. 处理空结果 ===
        if not compatible_methods:
            log_msg = (
                f"No type-compatible methods found for node '{node_name}' "
                f"(original method: {original_method_name}, "
                f"input_types={orig_input_types}, output_types={orig_output_types}). "
                f"Candidate pool ({len(candidate_pool)}): {candidate_pool}"
            )
            logger.debug(log_msg)

            # 可选：打印每个候选的实际类型（仅在调试级别）
            if logger.isEnabledFor(logging.DEBUG):
                for name in candidate_pool:
                    info = methods[name]
                    inp = info.get("input_types", []) or []
                    out = info.get("output_types", []) or []
                    logger.debug("  %s: input=%s, output=%s", name, inp, out)

            if not allow_fallback_on_empty:
                raise RuntimeError(
                    f"Strict mode: no type-compatible methods for node '{node_name}'. "
                    f"Expected input={orig_input_types}, output={orig_output_types}."
                )

            # === 执行兜底回退 ===
            if compatibility_mode == "all":
                result = all_method_names
            elif compatibility_mode == "group_only":
                node_group = node_data.get("group", "default")
                group_methods = [
                    name for name, info in methods.items()
                    if info.get("group", "default") == node_group
                ]
                result = group_methods[:1] if group_methods else all_method_names[:1]
            else:  # group_with_fallback
                result = all_method_names

            logger.warning(
                "Falling back to non-type-safe methods for node '%s' due to empty compatible set. "
                "Returned %d methods: %s",
                node_name, len(result), result[:3]  # 只显示前3个避免日志过长
            )
            return result

        return compatible_methods

    def _phase_modular_compression(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        阶段三：模块化压缩 (Modular Compression)
        识别图中可被替换的高效子图，用更小或更快的等效结构进行替代。

        :param adaptoflux_instance: 待优化的 AdaptoFlux 实例
        :param input_data: 用于评估等效性的输入数据
        :param target: 对应的标签（用于评估性能，非必需）
        :return: 包含压缩后模型信息和压缩情况的字典
        """
        if self.verbose:
            logger.info(f"[Phase 3] Modular Compression: Searching for compressible subgraphs...")

        gp = adaptoflux_instance.graph_processor
        original_loss = self._evaluate_loss_with_instance(adaptoflux_instance, input_data, target)
        original_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)

        # 此阶段的实现高度依赖于具体的“子图模式”定义。
        # 一个简化示例：查找连续的、功能单一的节点序列（如 "add -> multiply"）并尝试用一个复合节点替换。
        # 由于实现复杂，这里先提供一个占位逻辑，返回原模型。

        # --- 占位逻辑开始 ---
        # 假设我们识别到一个子图可以被压缩
        # subgraph_to_compress = self._find_compressible_subgraph(gp.graph)
        # if subgraph_to_compress:
        #     compressed_subgraph = self._create_compressed_version(subgraph_to_compress)
        #     if self._is_equivalent(subgraph_to_compress, compressed_subgraph, input_data, threshold=self.compression_threshold):
        #         # 执行图替换
        #         self._replace_subgraph(gp.graph, subgraph_to_compress, compressed_subgraph)
        #         # 记录这个高性能子图，用于进化阶段
        #         self.high_performance_subgraphs.append(compressed_subgraph)
        # --- 占位逻辑结束 ---

        # 由于子图压缩逻辑复杂，且依赖于具体的图模式，我们暂时跳过实际压缩，直接返回原模型。
        # 在实际项目中，您需要在此处实现具体的子图发现和替换算法。

        if self.verbose:
            logger.info(f"[Phase 3] Modular Compression: No compression applied in this placeholder implementation.")

        return {
            'final_model': adaptoflux_instance,
            'final_loss': original_loss,
            'final_accuracy': original_acc,
            'compression_applied': False, # 标记本次是否执行了压缩
            'compressed_subgraphs': 0
        }

    def _phase_method_pool_evolution(self, adaptoflux_instance) -> Dict[str, Any]:
        """
        阶段四：方法池进化 (Method Pool Evolution)
        将在“模块化压缩”阶段发现的高性能子结构，抽象为新的方法，并注入到方法池中。

        :param adaptoflux_instance: 当前的 AdaptoFlux 实例，其方法池将被更新
        :return: 包含进化后方法池信息的字典
        """
        if self.verbose:
            logger.info(f"[Phase 4] Method Pool Evolution: Evolving method pool with {len(self.high_performance_subgraphs)} new subgraphs...")

        methods_added = 0
        # 遍历记录的高性能子图
        for subgraph in self.high_performance_subgraphs:
            # 为子图创建一个唯一的方法名
            new_method_name = f"evolved_method_{len(adaptoflux_instance.methods) + methods_added + 1}"

            # 将子图封装为一个新方法
            # 这需要一个复杂的封装过程，将子图的输入输出接口和内部逻辑打包成一个可调用的函数
            # new_method_callable = self._wrap_subgraph_as_method(subgraph)

            # 由于封装逻辑复杂，这里仅模拟添加一个占位方法
            # adaptoflux_instance.add_method(new_method_name, new_method_callable, ...)

            # 模拟添加
            adaptoflux_instance.methods[new_method_name] = {
                'output_count': 1,
                'input_types': ['scalar'],
                'output_types': ['scalar'],
                'group': 'evolved',
                'weight': 1.0,
                'vectorized': True,
                'is_evolved': True, # 标记为进化而来
                'source_subgraph': 'placeholder' # 保存来源信息
            }
            methods_added += 1

            if self.verbose:
                logger.info(f"  Added new evolved method: {new_method_name}")

        # 清空已进化的子图列表，避免重复添加
        self.high_performance_subgraphs.clear()

        if self.verbose:
            logger.info(f"[Phase 4] Method Pool Evolution: Added {methods_added} new methods to the pool.")

        return {
            'methods_added': methods_added,
            'new_method_names': [f"evolved_method_{i}" for i in range(len(adaptoflux_instance.methods) - methods_added + 1, len(adaptoflux_instance.methods) + 1)]
        }

    def _evaluate_loss_with_instance(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> float:
        """
        辅助方法：使用指定的 AdaptoFlux 实例计算损失。
        """
        try:
            output = adaptoflux_instance.infer_with_graph(values=input_data)
            loss = self.loss_fn(output, target)
            return float(loss)
        except Exception as e:
            logger.error(f"Evaluation failed for instance: {e}")
            return float('inf')

    def _evaluate_accuracy_with_instance(self, adaptoflux_instance, input_data: np.ndarray, target: np.ndarray) -> float:
        """
        辅助方法：使用指定的 AdaptoFlux 实例计算准确率。
        """
        try:
            output = adaptoflux_instance.graph_processor.infer_with_graph(values=input_data)
            output = np.array(output)
            if len(output.shape) == 1 or output.shape[1] == 1:
                pred_classes = (output >= 0.5).astype(int).flatten()
            else:
                pred_classes = np.argmax(output, axis=1)
            true_labels = np.array(target).flatten()
            accuracy = float(np.mean(pred_classes == true_labels))
            return accuracy
        except Exception as e:
            logger.error(f"Accuracy evaluation failed for instance: {e}")
            return 0.0

    def train(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        max_evo_cycles: int = 5,
        save_model: bool = True,
        model_save_path: Optional[str] = None,
        save_best_model: bool = True,
        best_model_subfolder: str = "best",
        final_model_subfolder: str = "final",
        **kwargs
    ) -> dict:
        """
        实现基类的 train 方法。
        执行完整的“图演”优化循环。

        :param input_data: 用于快速评估的输入数据（小批量）
        :param target: 对应的标签
        :param max_evo_cycles: 最多执行的“精炼-压缩-进化”循环次数
        :param save_model: 是否在训练结束后保存模型
        :param model_save_path: 模型保存的文件夹路径
        :param save_best_model: 是否保存过程中遇到的最佳模型
        :param best_model_subfolder: 最佳模型保存的子目录
        :param final_model_subfolder: 最终模型保存的子目录
        :param kwargs: 其他可选参数
        :return: 一个包含训练过程信息的字典
        """
        if self.verbose:
            logger.info(f"Starting GraphEvoTrainer. Max evolution cycles: {max_evo_cycles}")

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

        # 阶段一：多样初始化
        init_result = self._phase_diverse_initialization(input_data, target)
        current_af = init_result['best_model']
        current_loss = init_result['best_loss']
        current_acc = init_result['best_accuracy']

        # 更新全局状态
        self.adaptoflux = current_af

        # 记录最佳模型
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
            results['total_refinement_steps'] += refinement_result['steps_taken']

            cycle_results['refinement'] = refinement_result

            # 阶段三：模块化压缩
            compression_result = self._phase_modular_compression(current_af, input_data, target)
            current_af = compression_result['final_model']
            current_loss = compression_result['final_loss'] # 更新损失和准确率
            current_acc = compression_result['final_accuracy']
            if compression_result['compression_applied']:
                results['total_compressions'] += compression_result['compressed_subgraphs']

            cycle_results['compression'] = compression_result

            # 阶段四：方法池进化 (根据频率决定)
            if cycle % self.evolution_frequency == 0:
                evolution_result = self._phase_method_pool_evolution(current_af)
                results['total_methods_evolved'] += evolution_result['methods_added']
                cycle_results['evolution'] = evolution_result
            else:
                cycle_results['evolution'] = {'skipped': True, 'reason': 'frequency'}

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
            target_node, 
            compatibility_mode=self.compatibility_mode
        )

        best_candidate = None
        best_loss = current_loss  # 初始化为当前损失，用于比较

        # 遍历所有候选方法（跳过当前方法）
        for candidate_method_name in candidate_methods:
            if candidate_method_name == original_method_name:
                continue

            # 创建临时副本，避免污染原模型
            temp_af = copy.deepcopy(adaptoflux_instance)
            temp_gp = temp_af.graph_processor
            # 尝试替换节点方法
            temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

            # 评估替换后的损失
            new_loss = self._evaluate_loss_with_instance(temp_af, input_data, target)

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
            new_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)
            
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

        # 随机打乱节点顺序，避免顺序偏差
        nodes_to_try = random.sample(processing_nodes, len(processing_nodes))
        improvement_made = False
        total_replacements = 0
        current_acc = 0.0

        # 遍历每一个待优化节点
        for target_node in nodes_to_try:
            original_method_name = gp.graph.nodes[target_node]['method_name']
            candidate_methods = self._get_compatible_methods_for_node(
                adaptoflux_instance, 
                target_node, 
                compatibility_mode=self.compatibility_mode
            )

            best_candidate = None
            best_loss = current_loss  # 注意：current_loss 在本轮中可能已被更新

            # 尝试所有兼容方法
            for candidate_method_name in candidate_methods:
                if candidate_method_name == original_method_name:
                    continue

                # 创建临时副本进行评估
                temp_af = copy.deepcopy(adaptoflux_instance)
                temp_gp = temp_af.graph_processor
                temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

                new_loss = self._evaluate_loss_with_instance(temp_af, input_data, target)

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_candidate = candidate_method_name

            # 如果找到更优方法，立即应用到原图（贪心）
            if best_candidate and best_loss < current_loss:
                gp.graph.nodes[target_node]['method_name'] = best_candidate
                current_loss = best_loss  # 更新当前损失，供后续节点使用
                current_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)
                improvement_made = True
                total_replacements += 1

                if self.verbose:
                    logger.info(f"  Replacement {total_replacements}: Node '{target_node}' "
                                f"{original_method_name} → {best_candidate}, Loss: {current_loss:.6f}")

                # ⚠️ 重要：方法替换可能改变节点的“处理节点”属性（例如输入/输出类型变化），
                # 因此需要重新获取当前所有处理节点，确保后续操作基于最新图状态。
                processing_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
                # 注意：虽然 `nodes_to_try` 是旧列表，但仅包含节点名，遍历仍安全；
                # 若需更严格的一致性，可考虑在每次修改后 break 并重启本轮，但会增加开销。

        return improvement_made, current_loss, current_acc, total_replacements, processing_nodes
