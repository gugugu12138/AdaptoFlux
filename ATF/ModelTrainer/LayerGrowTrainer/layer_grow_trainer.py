# layer_grow_trainer.py
from ..model_trainer import ModelTrainer
import numpy as np
import logging
import random
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
        

    def _evaluate_loss(self, input_data: np.ndarray, target: np.ndarray) -> float:
        """
        核心机制的 "Evaluate" 步骤。
        在当前图结构上执行前向传播并计算损失。

        :param input_data: 用于评估的输入数据（建议使用小批量以加速）
        :param target: 对应的标签
        :return: 计算得到的损失值
        """
        try:
            # 使用 AdaptoFlux 实例的 infer_with_graph 方法
            output = self.adaptoflux.graph_processor.infer_with_graph(input_data)
            # 确保输出和目标形状兼容
            if output.shape[0] != target.shape[0]:
                raise ValueError(f"Output batch size {output.shape[0]} != target batch size {target.shape[0]}")
            # 使用 AdaptoFlux 实例的损失函数
            loss = self.loss_fn(output, target)
            return loss
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return float('inf')
    
    def _evaluate_accuracy(self, input_data: np.ndarray, target: np.ndarray) -> float:
        """
        计算当前图结构的分类准确率。
        
        :param input_data: 输入数据
        :param target: 真实标签 (shape: [N,] 或 [N, 1])
        :return: 准确率 (0~1)
        """
        try:
            output = self.adaptoflux.graph_processor.infer_with_graph(input_data)  # [N, C] 或 [N,]

            # 假设是分类任务
            if len(output.shape) == 1 or output.shape[1] == 1:
                # 二分类，输出是单值
                pred_classes = (output >= 0.5).astype(int).flatten()
            else:
                # 多分类，取最大值索引
                pred_classes = np.argmax(output, axis=1)

            true_labels = np.array(target).flatten()
            accuracy = np.mean(pred_classes == true_labels)
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
        **kwargs
    ) -> dict:
        """
        实现基类的 train 方法。
        执行完整的“层叠式生成-评估-回退”循环，尝试为当前图添加多个新层。

        :param input_data: 用于快速评估的输入数据（小批量）
        :param target: 对应的标签
        :param max_layers: 最多尝试添加的新层数量
        :param kwargs: 其他可选参数
        :return: 一个包含训练过程信息的字典
        """
        if self.verbose:
            logger.info(f"Starting LayerGrowTrainer. Max layers to grow: {max_layers}")

        results = {
            "layers_added": 0,
            "attempt_history": []
        }

        for layer_idx in range(max_layers):
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
                results["layers_added"] += 1
                base_loss = new_loss # 更新 base_loss 用于下一层的比较
            else:
                if self.verbose:
                    logger.info(f"--- Failed to add layer {layer_idx + 1} after {self.max_attempts} attempts. "
                                f"Stopping growth. ---")
                break # 如果某一层失败，则停止继续添加

        if self.verbose:
            logger.info(f"LayerGrowTrainer finished. Successfully added {results['layers_added']} layers.")

        # 根据参数决定是否保存模型
        if save_model:
            try:
                self.adaptoflux.save_model()
                if self.verbose:
                    logger.info("Model saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

        return results
