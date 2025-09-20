# layer_grow_trainer.py
from ..model_trainer import ModelTrainer
import numpy as np
import logging
import random
import copy
from typing import Optional
import os
import json
from ...PathGenerator.path_generator import PathGenerator
from ...GraphManager.graph_processor import GraphProcessor

# 设置日志
logger = logging.getLogger(__name__)

class LayerGrowTrainer(ModelTrainer):
    """
    一个继承自 ModelTrainer 的具体训练器。
    该训练器实现了 AdaptoFlux 的核心构建机制——“层叠式生成-评估-回退”(Layered Generate-Evaluate-Backtrack)。
    它通过在现有图结构上迭代尝试添加新层，并基于性能评估决定是否保留，从而实现图的动态扩展。
    """

    def __init__(
        self,
        adaptoflux_instance,
        max_attempts: int = 5,
        decision_threshold: float = 0.0,
        verbose: bool = True
    ):
        """
        初始化 LayerGrowTrainer。

        :param adaptoflux_instance: 已初始化的 AdaptoFlux 对象
        :param max_attempts: 为添加一层而进行的最大尝试次数
        :param decision_threshold: 决策阈值。若 (旧损失 - 新损失) > threshold，则接受新层。
                                   threshold=0.0 表示贪心策略（必须严格变好）。
        :param verbose: 是否打印详细日志
        """
        super().__init__(adaptoflux_instance)
        self.max_attempts = max_attempts
        self.decision_threshold = decision_threshold
        self.verbose = verbose
        

    def _evaluate_loss(self, input_data: np.ndarray, target: np.ndarray, use_pipeline=False, num_workers=4) -> float:
        """
        核心机制的 "Evaluate" 步骤。
        在当前图结构上执行前向传播并计算损失。

        :param input_data: 用于评估的输入数据（建议使用小批量以加速）
        :param target: 对应的标签
        :param use_pipeline: 是否使用并行流水线推理（多线程）
        :param num_workers: 并行推理使用的线程数（仅在 use_pipeline=True 时有效）
        :return: 计算得到的损失值，若失败返回 float('inf')
        """
        try:
            # 选择推理方式
            gp = self.adaptoflux.graph_processor
            if use_pipeline:
                output = gp.infer_with_graph_pipeline(values=input_data, num_workers=num_workers)
            else:
                output = gp.infer_with_graph(values=input_data)

            # 确保输出和目标形状兼容
            if output.shape[0] != target.shape[0]:
                raise ValueError(f"Output batch size {output.shape[0]} != target batch size {target.shape[0]}")

            # 使用 AdaptoFlux 实例的损失函数
            loss = self.loss_fn(output, target)
            return float(loss)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return float('inf')
    
    def _evaluate_accuracy(self, input_data: np.ndarray, target: np.ndarray, use_pipeline=False, num_workers=4) -> float:
        """
        计算当前图结构的分类准确率。

        :param input_data: 输入数据
        :param target: 真实标签 (shape: [N,] 或 [N, 1])
        :param use_pipeline: 是否使用并行流水线推理
        :param num_workers: 并行推理使用的线程数
        :return: 准确率 (0~1)，若失败返回 0.0
        """
        try:
            # 选择推理方式
            gp = self.adaptoflux.graph_processor
            if use_pipeline:
                output = gp.infer_with_graph_pipeline(values=input_data, num_workers=num_workers)
            else:
                output = gp.infer_with_graph(values=input_data)

            # 确保输出是 NumPy 数组
            output = np.array(output)

            # 预测类别
            if len(output.shape) == 1 or output.shape[1] == 1:
                # 二分类：阈值 0.5
                pred_classes = (output >= 0.5).astype(int).flatten()
            else:
                # 多分类：取最大概率类别
                pred_classes = np.argmax(output, axis=1)

            # 真实标签处理
            true_labels = np.array(target).flatten()

            # 计算准确率
            accuracy = float(np.mean(pred_classes == true_labels))
            return accuracy

        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            return 0.0  # 失败时返回 0

    def _should_accept(self, old_loss: float, new_loss: float) -> bool:
        """
        核心机制的 "Decide" 步骤。
        根据决策策略判断是否接受新层。

        :param old_loss: 添加新层之前的损失
        :param new_loss: 添加新层之后的损失
        :return: True 表示接受，False 表示拒绝
        """
        improvement = old_loss - new_loss
        return improvement > self.decision_threshold

    def train(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        max_layers: int = 10,
        discard_unmatched='to_discard', 
        discard_node_method_name="null",
        save_model: bool = True,
        on_retry_exhausted: str = "stop",  # 新增：失败后策略
        rollback_layers: int = 1,          # 新增：回退层数
        max_total_attempts: Optional[int] = None,  # 👈 新增：全局最大尝试次数
        model_save_path: Optional[str] = None,
        save_best_model: bool = True,           # 👈 新增：是否保存最佳模型
        best_model_subfolder: str = "best",     # 👈 新增：最佳模型子目录
        final_model_subfolder: str = "final",   # 👈 新增：最终模型子目录
        **kwargs
    ) -> dict:
        """
        实现基类的 train 方法。
        执行完整的“层叠式生成-评估-回退”循环，尝试为当前图添加多个新层。
        如果考虑加速可以把上一层处理的结果缓存下来，避免重复计算。不过这个方法在推理阶段使用适配很差，后期再写。

        :param input_data: 用于快速评估的输入数据（小批量）
        :param target: 对应的标签
        :param max_layers: 最多尝试添加的新层数量
        :param discard_unmatched: 是否丢弃不匹配的节点
        :param discard_node_method_name: 丢弃节点的方法名称
        :param save_model: 是否在训练结束后保存模型
        :param on_retry_exhausted: 当所有尝试失败时的策略（如 "stop", "continue"）
        :param rollback_layers: 如果添加失败，回退的层数
        :param max_total_attempts: 全局最大增长尝试次数，防止无限循环。默认为 max_layers * 30
        :param model_save_path: 模型保存的文件夹路径。仅在 save_model=True 时生效。默认为 None（使用 'models'）
        :param kwargs: 其他可选参数
        :return: 一个包含训练过程信息的字典
        """

        best_acc = -1.0
        best_graph_snapshot = None
        best_layer_count = 0
        if self.verbose:
            logger.info(f"Starting LayerGrowTrainer. Max layers to grow: {max_layers}")

        # 初始化 results 时添加
        results = {
            "layers_added": 0,
            "attempt_history": [],
            "total_growth_attempts": 0,
            "total_candidate_attempts": 0,
            "rollback_count": 0,
            "rollback_events": []
        }


        # 设置默认值：如果不指定，则为 max_layers * 30
        if max_total_attempts is None:
            max_total_attempts = max_layers * 30

        iteration_count = 0
        layer_idx = 0
        while layer_idx < max_layers and iteration_count < max_total_attempts:
            iteration_count += 1
            results["total_growth_attempts"] += 1

            if self.verbose:
                logger.info(f"--- Starting to grow layer {layer_idx + 1} ---")

            # 记录当前状态（损失 + 准确率）
            base_loss = self._evaluate_loss(input_data, target)
            base_acc = self._evaluate_accuracy(input_data, target)
            if self.verbose:
                logger.info(f"Base loss before attempt: {base_loss:.6f}, Accuracy: {base_acc:.4f}")

            layer_success = False
            attempt_record = {"layer": layer_idx + 1, "attempts": []}

            # 尝试循环
            for attempt in range(1, self.max_attempts + 1):
                results["total_candidate_attempts"] += 1  # 每次尝试都算一次候选生成
                attempt_info = {"attempt": attempt, "accepted": False, "new_loss": None}
                if self.verbose:
                    logger.info(f"  Attempt {attempt}/{self.max_attempts}")

                # 1. GENERATE: 生成候选方案
                candidate_plan = self.adaptoflux.process_random_method()
                # 生成为空则跳过，这个地方代码逻辑有一丢丢问题，应该要全为空但是问题不大
                if not candidate_plan["valid_groups"]:
                    if self.verbose:
                        logger.warning("  process_random_method is empty. Skipping.")
                    attempt_info["status"] = "empty_plan"
                    attempt_record["attempts"].append(attempt_info)
                    continue

                # 2. EVALUATE: 临时应用候选层
                # 这里直接调用 AdaptoFlux 实例的 append_nx_layer 方法
                try:
                    self.adaptoflux.append_nx_layer(
                        candidate_plan,
                        discard_unmatched=discard_unmatched,
                        discard_node_method_name=discard_node_method_name
                    )
                except Exception as e:
                    logger.error(f"  Failed to append layer: {e}")
                    import traceback
                    logger.error(f"Exception traceback:\n{traceback.format_exc()}")  # 👈 关键：打印完整堆栈
                    attempt_info["status"] = f"append_failed: {e}"
                    attempt_record["attempts"].append(attempt_info)
                    continue

                # 2.2 EVALUATE: 评估新图的性能
                new_loss = self._evaluate_loss(input_data, target)
                attempt_info["new_loss"] = new_loss
                new_acc = self._evaluate_accuracy(input_data, target)
                attempt_info["new_acc"] = new_acc

                # 3. DECIDE: 决定是否接受
                if self._should_accept(base_loss, new_loss):
                    # 4. ACCEPT: 决策成功，新层已通过 append_nx_layer 永久集成
                    if self.verbose:
                        logger.info(f"  ✅ Layer accepted on attempt {attempt}. "
                                    f"Loss: {base_loss:.6f} → {new_loss:.6f}, "
                                    f"Acc: {base_acc:.4f} → {new_acc:.4f}")
                    attempt_info["accepted"] = True
                    attempt_info["status"] = "accepted"
                    attempt_record["attempts"].append(attempt_info)
                    layer_success = True
                    break
                else:
                    # 4. BACKTRACK: 决策失败，撤销上一步
                    try:
                        self.adaptoflux.remove_last_nx_layer()
                    except Exception as e:
                        logger.error(f"  Failed to remove last layer: {e}")
                        # 如果无法回退，图结构可能已损坏，应中断
                        attempt_info["status"] = f"rollback_failed: {e}"
                        attempt_record["attempts"].append(attempt_info)
                        break

                    if self.verbose:
                        logger.info(f"  ❌ Layer rejected. Loss: {new_loss:.6f} (base: {base_loss:.6f}), "
                                    f"Acc: {new_acc:.4f} (base: {base_acc:.4f}). "
                                    f"Reverted to previous state.")

                    attempt_info["status"] = "rejected"
                    attempt_record["attempts"].append(attempt_info)

            # 记录本次层的尝试历史
            results["attempt_history"].append(attempt_record)

            # 更新最终结果
            if layer_success:
                layer_idx += 1
                results["layers_added"] += 1
                base_loss = new_loss  # 更新 base_loss 用于下一层的比较

                if new_acc > best_acc:
                    if self.verbose:
                        logger.info(f"🎉 New best accuracy: {best_acc:.4f} → {new_acc:.4f}, layers={results['layers_added']}")
                    best_acc = new_acc
                    # 保存图结构和方法池的快照
                    best_graph_snapshot = copy.deepcopy(self.adaptoflux.graph)
                    best_methods_snapshot = copy.deepcopy(self.adaptoflux.methods)
                    best_layer_count = results["layers_added"]
            else:
                if on_retry_exhausted == "stop":
                    if self.verbose:
                        logger.info(f"--- Failed to add layer {layer_idx + 1} after {self.max_attempts} attempts. "
                                    f"Stopping growth. ---")
                    break

                elif on_retry_exhausted == "rollback":
                    if self.verbose:
                        logger.info(f"--- Layer {layer_idx + 1} failed after {self.max_attempts} attempts. "
                                    f"Rolling back {rollback_layers} layer(s). ---")

                    results["rollback_count"] += 1
                    rolled_back_success = 0
                    rolled_back_fail = 0

                    current_layers = results["layers_added"]
                    actual_rollback = min(rollback_layers, current_layers)  # 安全限制

                    for _ in range(actual_rollback):
                        try:
                            if layer_idx > 0:  # 确保有层可以回退
                                if self.verbose:
                                    logger.info(f"  Rolling back layer {layer_idx + 1}...")
                                self.adaptoflux.remove_last_nx_layer()
                                results["layers_added"] -= 1
                                layer_idx -= 1
                                rolled_back_success += 1
                        except Exception as e:
                            logger.error(f"Rollback failed: {e}")
                            rolled_back_fail += 1

                    # 记录事件
                    results["rollback_events"].append({
                        "at_layer": layer_idx + 1,
                        "rollback_layers": rollback_layers,
                        "success_count": rolled_back_success,
                        "failed_count": rolled_back_fail,
                        "reason": "retry_exhausted"
                    })

                    if self.verbose:
                        logger.info(f"Rolled back {rolled_back_success} layers (failed: {rolled_back_fail}).")

                    # 👇 关键：更新当前性能基准
                    base_loss = self._evaluate_loss(input_data, target)
                    base_acc = self._evaluate_accuracy(input_data, target)
                    if self.verbose:
                        logger.info(f"  Reset base loss to: {base_loss:.6f}, acc: {base_acc:.4f}")

                else:
                    logger.warning(f"Invalid on_retry_exhausted='{on_retry_exhausted}'. Must be 'stop' or 'rollback'. Stopping.")
                    break

        # 循环结束后，判断终止原因
        if iteration_count >= max_total_attempts:
            if self.verbose:
                logger.info(f"--- Training stopped: reached global maximum attempts ({max_total_attempts}) ---")
        elif layer_idx >= max_layers:
            if self.verbose:
                logger.info(f"--- Training stopped: reached maximum layers ({max_layers}) ---")
        else:
            if self.verbose:
                logger.info("--- Training stopped: unknown reason ---")
        if self.verbose:
            logger.info(f"LayerGrowTrainer finished. Successfully added {results['layers_added']} layers.")

        # 根据参数决定是否保存模型
        if save_model:
            try:
                base_save_path = model_save_path or "models"
                os.makedirs(base_save_path, exist_ok=True)

                # === 保存最终模型 ===
                final_path = os.path.join(base_save_path, final_model_subfolder)
                self.adaptoflux.save_model(folder=final_path)
                if self.verbose:
                    logger.info(f"Final model saved to '{final_path}'")

                # === 保存最佳模型 ===
                if save_best_model and best_graph_snapshot is not None:
                    best_path = os.path.join(base_save_path, best_model_subfolder)

                    # 临时替换当前图结构以保存最佳状态
                    original_graph = self.adaptoflux.graph
                    original_methods = self.adaptoflux.methods

                    self.adaptoflux.graph = best_graph_snapshot
                    self.adaptoflux.methods = best_methods_snapshot
                    try:
                        self.adaptoflux.save_model(folder=best_path)
                        if self.verbose:
                            logger.info(f"Best model saved to '{best_path}' (accuracy={best_acc:.4f}, layers={best_layer_count})")
                    finally:
                        # 恢复原始状态
                        self.adaptoflux.graph = original_graph
                        self.adaptoflux.methods = original_methods

                # 添加到 results
                results["final_model_saved"] = final_path
                results["final_model_accuracy"] = self._evaluate_accuracy(input_data, target)
                results["final_model_layers"] = results["layers_added"]
                if save_best_model and best_graph_snapshot is not None:
                    results["best_model_saved"] = best_path
                    results["best_model_accuracy"] = best_acc
                    results["best_model_layers"] = best_layer_count
                
                # 自动保存训练日志为 JSON
                log_filename = kwargs.get("log_filename", "training_log.json")
                log_path = os.path.join(base_save_path, log_filename)
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, default=str)
                results["training_log_saved"] = log_path

            except Exception as e:
                logger.error(f"Failed to save model(s): {e}")
                import traceback
                logger.error(traceback.format_exc())

        return results
