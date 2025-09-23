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
        init_mode: str = "fixed",           # <-- 新增：初始化模式
        init_layers_list: Optional[List[int]] = None,  # <-- 新增：自定义层数列表
        frozen_nodes: Optional[List[str]] = None,  # 冻结节点的方法名
        frozen_methods: Optional[List[str]] = None,  # <-- 新增参数
        refinement_strategy: str = "random_single",  # 支持 "random_single", "full_sweep"
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
        self.verbose = verbose

        # 校验参数
        if self.init_mode == "list":
            if self.init_layers_list is None:
                raise ValueError("init_mode='list' requires init_layers_list to be provided.")
            if len(self.init_layers_list) < self.num_initial_models:
                raise ValueError(f"init_layers_list length ({len(self.init_layers_list)}) must be >= num_initial_models ({self.num_initial_models})")

        # 用于记录高性能子图，供“方法池进化”阶段使用
        self.high_performance_subgraphs: List[Dict[str, Any]] = []

        self.refinement_strategy = refinement_strategy
        self._strategy_map = {
            "random_single": self._refine_random_single_step,
            "full_sweep": self._refine_full_sweep_step,
            # 未来可加："weighted_sample": self._refine_weighted_sample_step,
        }
        if self.refinement_strategy not in self._strategy_map:
            raise ValueError(f"Unknown refinement_strategy: {self.refinement_strategy}. "
                            f"Available: {list(self._strategy_map.keys())}")

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
        if self.verbose:
            logger.info(f"[Phase 2] Node-wise Refinement: Starting refinement with strategy '{self.refinement_strategy}'...")

        gp = adaptoflux_instance.graph_processor
        current_loss = self._evaluate_loss_with_instance(adaptoflux_instance, input_data, target)
        current_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)

        improvement_made = False
        steps_taken = 0

        # 获取并过滤处理节点
        processing_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
        final_frozen_nodes = set(self.frozen_nodes)
        if self.frozen_methods:
            method_frozen_nodes = {
                node for node in processing_nodes
                if gp.graph.nodes[node].get('method_name') in self.frozen_methods
            }
            final_frozen_nodes.update(method_frozen_nodes)
        if final_frozen_nodes:
            processing_nodes = [node for node in processing_nodes if node not in final_frozen_nodes]

        if self.verbose:
            logger.info(f"  Found {len(processing_nodes)} processing nodes to refine "
                        f"(excluding {len(final_frozen_nodes)} frozen nodes).")

        # 获取策略函数
        strategy_func = self._strategy_map[self.refinement_strategy]

        # 主循环
        for step in range(self.max_refinement_steps):
            if steps_taken >= self.max_refinement_steps:
                break

            if not processing_nodes:
                break

            # 调用策略函数
            imp, new_loss, new_acc, step_inc, updated_nodes = strategy_func(
                adaptoflux_instance, input_data, target, processing_nodes, current_loss, gp
            )

            if imp:
                improvement_made = True
                current_loss = new_loss
                current_acc = new_acc
                steps_taken += step_inc
                processing_nodes = updated_nodes

                if self.verbose and self.refinement_strategy == "random_single":
                    # full_sweep 已在内部打印，避免重复
                    pass  # 日志已在策略函数内打印
            else:
                # 无改进，继续或提前终止（可选）
                # 当前不提前终止，继续尝试
                pass

        if self.verbose:
            if improvement_made:
                logger.info(f"[Phase 2] Refinement completed in {steps_taken} steps. Final Loss: {current_loss:.6f}, Acc: {current_acc:.4f}")
            else:
                logger.info(f"[Phase 2] Refinement completed. No improvements found within {self.max_refinement_steps} steps.")

        return {
            'final_model': adaptoflux_instance,
            'final_loss': current_loss,
            'final_accuracy': current_acc,
            'steps_taken': steps_taken,
            'improvement_made': improvement_made
        }

    def _get_compatible_methods_for_node(self, adaptoflux_instance, node_name: str) -> List[str]:
        """
        一个辅助方法，用于获取与指定节点兼容的候选方法列表。
        兼容性基于输入/输出数据类型匹配。

        :param adaptoflux_instance: AdaptoFlux 实例
        :param node_name: 图中节点的名称
        :return: 兼容的方法名称列表
        """
        gp = adaptoflux_instance.graph_processor
        methods = adaptoflux_instance.methods

        # 简化实现：返回同一组（group）的所有方法，或随机返回方法池中的一部分
        # 理想情况下，应根据节点的输入/输出边的 data_type 进行精确匹配
        node_data = gp.graph.nodes[node_name]
        node_group = node_data.get('group', 'default')  # 假设节点存储了组信息

        compatible_methods = [
            method_name for method_name, method_info in methods.items()
            if method_info.get('group') == node_group
        ]

        # 如果没有同组的，或者为了增加多样性，可以返回所有方法
        if len(compatible_methods) < 2:
            compatible_methods = list(methods.keys())

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
        执行一步随机单点优化。
        返回：(是否改进, 新loss, 新acc, 本次步数增量(0或1), 更新后的processing_nodes)
        """
        if not processing_nodes:
            return False, current_loss, 0.0, 0, processing_nodes

        target_node = random.choice(processing_nodes)
        original_method_name = gp.graph.nodes[target_node]['method_name']
        candidate_methods = self._get_compatible_methods_for_node(adaptoflux_instance, target_node)

        best_candidate = None
        best_loss = current_loss

        for candidate_method_name in candidate_methods:
            if candidate_method_name == original_method_name:
                continue

            temp_af = copy.deepcopy(adaptoflux_instance)
            temp_gp = temp_af.graph_processor
            temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

            new_loss = self._evaluate_loss_with_instance(temp_af, input_data, target)

            if new_loss < best_loss:
                best_loss = new_loss
                best_candidate = candidate_method_name

        if best_candidate and best_loss < current_loss:
            gp.graph.nodes[target_node]['method_name'] = best_candidate
            new_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)
            updated_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
            return True, best_loss, new_acc, 1, updated_nodes
        else:
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
        执行一轮完整遍历所有节点。
        返回：(本轮是否至少有一次改进, 最终loss, 最终acc, 本轮总替换次数, 最终processing_nodes)
        """
        if not processing_nodes:
            return False, current_loss, 0.0, 0, processing_nodes

        nodes_to_try = random.sample(processing_nodes, len(processing_nodes))  # 随机打乱
        improvement_made = False
        total_replacements = 0
        current_acc = 0.0

        for target_node in nodes_to_try:
            original_method_name = gp.graph.nodes[target_node]['method_name']
            candidate_methods = self._get_compatible_methods_for_node(adaptoflux_instance, target_node)

            best_candidate = None
            best_loss = current_loss

            for candidate_method_name in candidate_methods:
                if candidate_method_name == original_method_name:
                    continue

                temp_af = copy.deepcopy(adaptoflux_instance)
                temp_gp = temp_af.graph_processor
                temp_gp.graph.nodes[target_node]['method_name'] = candidate_method_name

                new_loss = self._evaluate_loss_with_instance(temp_af, input_data, target)

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_candidate = candidate_method_name

            if best_candidate and best_loss < current_loss:
                gp.graph.nodes[target_node]['method_name'] = best_candidate
                current_loss = best_loss
                current_acc = self._evaluate_accuracy_with_instance(adaptoflux_instance, input_data, target)
                improvement_made = True
                total_replacements += 1

                if self.verbose:
                    logger.info(f"  Replacement {total_replacements}: Node '{target_node}' "
                                f"{original_method_name} → {best_candidate}, Loss: {current_loss:.6f}")

                # 重要：更新节点列表（方法变更可能影响节点性质）
                processing_nodes = [node for node in gp.graph.nodes() if gp._is_processing_node(node)]
                # 注意：nodes_to_try 是旧列表，但我们继续用它遍历（安全，因为只是名字列表）
                # 如果想更严谨，可 break 并重启本轮，但当前设计允许“本轮内动态更新”

        return improvement_made, current_loss, current_acc, total_replacements, processing_nodes