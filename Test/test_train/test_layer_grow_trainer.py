# test_layer_grow_trainer.py

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import logging

# 假设你的模块路径是正确的，这里导入被测类
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer


# 为测试设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLayerGrowTrainer(unittest.TestCase):

    def setUp(self):
        """
        每个测试前的初始化。
        使用 mock 构建一个最小化的 AdaptoFlux 实例。
        """
        # Mock AdaptoFlux 实例
        self.mock_adaptoflux = MagicMock()

        # Mock graph 和 methods
        self.mock_graph = MagicMock()
        self.mock_methods = {
            "method_A": MagicMock(),
            "method_B": MagicMock()
        }

        # 设置 adaptoflux 的属性
        self.mock_adaptoflux.graph = self.mock_graph
        self.mock_adaptoflux.methods = self.mock_methods

        # Mock GraphProcessor 和 infer_with_graph
        self.mock_graph_processor = MagicMock()
        self.mock_adaptoflux.graph_processor = self.mock_graph_processor

        # Mock loss function
        self.mock_loss_fn = MagicMock()
        self.mock_adaptoflux.loss_fn = self.mock_loss_fn

        # 初始化 trainer
        self.trainer = LayerGrowTrainer(
            adaptoflux_instance=self.mock_adaptoflux,
            max_attempts=3,
            decision_threshold=0.0,
            verbose=True
        )

        # 准备测试数据
        self.input_data = np.random.rand(4, 10)  # (batch_size=4, features=10)
        self.target = np.random.rand(4, 1)       # (batch_size=4, output_dim=1)

    def test_initialization(self):
        """测试 LayerGrowTrainer 的初始化是否正确"""
        self.assertEqual(self.trainer.max_attempts, 3)
        self.assertEqual(self.trainer.decision_threshold, 0.0)
        self.assertTrue(self.trainer.verbose)

        # 检查 PathGenerator 是否被正确创建
        self.assertIsNotNone(self.trainer.path_generator)
        self.assertEqual(self.trainer.path_generator.graph, self.mock_graph)
        self.assertEqual(self.trainer.path_generator.methods, self.mock_methods)

    def test_evaluate_loss_success(self):
        """测试 _evaluate_loss 在正常情况下的行为"""
        # 模拟 infer 输出
        mock_output = np.random.rand(4, 1)
        self.mock_graph_processor.infer_with_graph.return_value = mock_output

        # 模拟 loss 函数返回值
        expected_loss = 0.45
        self.mock_loss_fn.return_value = expected_loss

        loss = self.trainer._evaluate_loss(self.input_data, self.target)

        self.assertEqual(loss, expected_loss)
        self.mock_graph_processor.infer_with_graph.assert_called_once_with(self.input_data)
        self.mock_loss_fn.assert_called_once_with(mock_output, self.target)

    def test_evaluate_loss_shape_mismatch(self):
        """测试 _evaluate_loss 在 batch size 不匹配时返回 inf"""
        mock_output = np.random.rand(5, 1)  # batch size 5 ≠ target's 4
        self.mock_graph_processor.infer_with_graph.return_value = mock_output

        loss = self.trainer._evaluate_loss(self.input_data, self.target)

        self.assertEqual(loss, float('inf'))

    def test_evaluate_loss_exception(self):
        """测试 _evaluate_loss 在推理失败时返回 inf"""
        self.mock_graph_processor.infer_with_graph.side_effect = Exception("Infer error")

        loss = self.trainer._evaluate_loss(self.input_data, self.target)

        self.assertEqual(loss, float('inf'))

    def test_should_accept(self):
        """测试 _should_accept 决策逻辑"""
        # 改进大于阈值（默认0）→ 接受
        self.assertTrue(self.trainer._should_accept(old_loss=1.0, new_loss=0.5))  # 改进 0.5
        self.assertFalse(self.trainer._should_accept(old_loss=1.0, new_loss=1.0))  # 无改进
        self.assertFalse(self.trainer._should_accept(old_loss=1.0, new_loss=1.2))  # 变差

        # 修改阈值测试
        trainer_custom = LayerGrowTrainer(self.mock_adaptoflux, decision_threshold=0.6)
        self.assertFalse(trainer_custom._should_accept(old_loss=1.0, new_loss=0.5))  # 改进 0.5 < 0.6

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_empty_plan(self, mock_gen_plan):
        """测试当生成的候选方案为空时的行为"""
        mock_gen_plan.return_value = {"valid_groups": []}

        results = self.trainer.train(self.input_data, self.target, max_layers=1)

        self.assertEqual(results["layers_added"], 0)
        self.assertEqual(len(results["attempt_history"]), 1)
        attempts = results["attempt_history"][0]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "empty_plan")

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_append_failure(self, mock_gen_plan):
        """测试 append_nx_layer 失败时的处理"""
        mock_gen_plan.return_value = {"valid_groups": ["group1"]}
        self.mock_adaptoflux.append_nx_layer.side_effect = Exception("Append failed")

        results = self.trainer.train(self.input_data, self.target, max_layers=1)

        self.assertEqual(results["layers_added"], 0)
        status = results["attempt_history"][0]["attempts"][0]["status"]
        self.assertIn("append_failed", status)

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_reject_and_backtrack(self, mock_gen_plan):
        """测试新层被拒绝并成功回退的情况"""
        mock_gen_plan.return_value = {"valid_groups": ["group1"]}
        self.mock_adaptoflux.append_nx_layer.return_value = None  # 成功
        self.mock_graph_processor.infer_with_graph.side_effect = [
            np.array([[0.1], [0.2]]),  # 基础输出（batch_size=2）
            np.array([[0.3], [0.4]])   # 新图输出
        ]
        self.mock_loss_fn.side_effect = [0.8, 0.85]  # 损失变差

        results = self.trainer.train(
            np.random.rand(2, 10),
            np.random.rand(2, 1),
            max_layers=1,
            discard_unmatched='to_discard',
            discard_node_method_name='null'
        )

        self.assertEqual(results["layers_added"], 0)
        attempt = results["attempt_history"][0]["attempts"][0]
        self.assertFalse(attempt["accepted"])
        self.assertEqual(attempt["status"], "rejected")
        self.mock_adaptoflux.remove_last_nx_layer.assert_called_once()

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_accept_layer(self, mock_gen_plan):
        """测试新层被接受的情况"""
        mock_gen_plan.return_value = {"valid_groups": ["group1"]}
        self.mock_adaptoflux.append_nx_layer.return_value = None
        self.mock_graph_processor.infer_with_graph.side_effect = [
            np.array([[0.1], [0.2]]),
            np.array([[0.15], [0.25]])
        ]
        self.mock_loss_fn.side_effect = [0.8, 0.6]  # 明确改进

        results = self.trainer.train(
            np.random.rand(2, 10),
            np.random.rand(2, 1),
            max_layers=1
        )

        self.assertEqual(results["layers_added"], 1)
        attempt = results["attempt_history"][0]["attempts"][0]
        self.assertTrue(attempt["accepted"])
        self.assertEqual(attempt["status"], "accepted")
        self.mock_adaptoflux.remove_last_nx_layer.assert_not_called()  # 不应回退

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_rollback_failure(self, mock_gen_plan):
        """测试回退失败时中断尝试"""
        mock_gen_plan.return_value = {"valid_groups": ["group1"]}
        self.mock_adaptoflux.append_nx_layer.return_value = None
        self.mock_graph_processor.infer_with_graph.side_effect = [np.array([[0.1]]), np.array([[0.2]])]
        self.mock_loss_fn.side_effect = [0.8, 0.9]
        self.mock_adaptoflux.remove_last_nx_layer.side_effect = Exception("Rollback failed")

        results = self.trainer.train(np.random.rand(1, 10), np.random.rand(1, 1), max_layers=1)

        # 尝试失败，但因为 rollback 失败，中断该层
        attempt = results["attempt_history"][0]["attempts"][0]
        self.assertIn("rollback_failed", attempt["status"])
        self.assertEqual(results["layers_added"], 0)

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_multiple_attempts_success_on_second(self, mock_gen_plan):
        """测试多次尝试，第二轮才成功"""
        mock_gen_plan.side_effect = [
            {"valid_groups": ["group1"]},  # 第一次尝试
            {"valid_groups": ["group2"]}   # 第二次成功
        ]
        self.mock_adaptoflux.append_nx_layer.return_value = None

        # 第一次评估损失变差，第二次变好
        self.mock_graph_processor.infer_with_graph.side_effect = [
            np.array([[0.1]]),  # base
            np.array([[0.2]]),  # attempt 1
            np.array([[0.15]]), # attempt 2
        ]
        self.mock_loss_fn.side_effect = [0.8, 0.85, 0.6]

        results = self.trainer.train(np.random.rand(1, 10), np.random.rand(1, 1), max_layers=1)

        self.assertEqual(results["layers_added"], 1)
        attempts = results["attempt_history"][0]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["accepted"])
        self.assertTrue(attempts[1]["accepted"])

    @patch.object(LayerGrowTrainer, '_generate_candidate_plan')
    def test_train_stop_on_failure(self, mock_gen_plan):
        """测试某一层失败后停止继续添加后续层"""
        mock_gen_plan.return_value = {"valid_groups": ["group1"]}
        self.mock_adaptoflux.append_nx_layer.return_value = None
        self.mock_graph_processor.infer_with_graph.side_effect = [
            np.array([[0.1]]), np.array([[0.2]]),
            np.array([[0.3]]), np.array([[0.4]]),
            np.array([[0.5]]), np.array([[0.6]])
        ] * 2  # 重复模拟
        self.mock_loss_fn.side_effect = [0.8, 0.85] * 6  # 损失始终变差

        results = self.trainer.train(
            np.random.rand(1, 10),
            np.random.rand(1, 1),
            max_layers=5
        )

        # 所有尝试都失败，只尝试了第一层
        self.assertEqual(results["layers_added"], 0)
        self.assertEqual(len(results["attempt_history"]), 1)  # 只有一层的记录

    def test_train_with_kwargs_forwarding(self):
        """测试 kwargs 正确传递给 append_nx_layer"""
        with patch.object(self.mock_adaptoflux, 'append_nx_layer') as mock_append:
            mock_plan = {"valid_groups": ["g1"]}
            with patch.object(self.trainer, '_generate_candidate_plan', return_value=mock_plan):
                with patch.object(self.trainer, '_evaluate_loss', return_value=0.5):
                    with patch.object(self.trainer, '_should_accept', return_value=True):
                        self.trainer.train(
                            self.input_data,
                            self.target,
                            max_layers=1,
                            discard_unmatched='custom_discard',
                            discard_node_method_name='noop'
                        )

                        mock_append.assert_called_once_with(
                            self.mock_methods,
                            mock_plan,
                            discard_unmatched='custom_discard',
                            discard_node_method_name='noop'
                        )


if __name__ == '__main__':
    unittest.main()