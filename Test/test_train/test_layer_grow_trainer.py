# test_layer_grow_trainer.py

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import logging

# 假设你的模块路径是正确的，这里导入被测类
from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer
from ATF.ModelTrainer.model_trainer import ModelTrainer  # 基类


class TestLayerGrowTrainer(unittest.TestCase):

    def setUp(self):
        self.adaptoflux = MagicMock()
        self.adaptoflux.graph = "mock_graph"
        self.adaptoflux.methods = {
            "method_A": {"input_count": 2, "output_count": 1},
            "method_B": {"input_count": 3, "output_count": 1}
        }

        self.adaptoflux.graph_processor = MagicMock()
        self.adaptoflux.graph_processor.infer_with_graph.side_effect = lambda x: x @ np.random.rand(x.shape[1], 10)

        self.adaptoflux.loss_fn = MagicMock()
        self.adaptoflux.loss_fn.side_effect = lambda output, target: np.mean((output - target) ** 2)

        self.adaptoflux.append_nx_layer = MagicMock()
        self.adaptoflux.remove_last_nx_layer = MagicMock()

        self.trainer = LayerGrowTrainer(
            adaptoflux_instance=self.adaptoflux,
            max_attempts=3,
            decision_threshold=0.0,
            verbose=True
        )

        # ✅ 修复关键：mock 方法
        self.mock_process_random = MagicMock()
        self.trainer.path_generator.process_random_method = self.mock_process_random

        self.input_data = np.random.randn(16, 5)
        self.target = np.random.randn(16, 10)

    def test_generate_candidate_empty_skip(self):
        """
        测试：当 process_random_method 返回空 valid_groups 时，跳过该次尝试
        """
        # 模拟 path_generator 的 process_random_method 返回空
        self.trainer.path_generator.process_random_method.return_value = {"valid_groups": []}

        result = self.trainer.train(self.input_data, self.target, max_layers=1)

        # 断言：append 和 remove 都没有被调用
        self.adaptoflux.append_nx_layer.assert_not_called()
        self.adaptoflux.remove_last_nx_layer.assert_not_called()

        # 断言：尝试了1次，状态为 empty_plan
        attempt = result["attempt_history"][0]["attempts"][0]
        self.assertEqual(attempt["status"], "empty_plan")
        self.assertFalse(attempt["accepted"])
        self.assertEqual(result["layers_added"], 0)

    def test_first_attempt_success(self):
        """
        测试：第一次尝试成功（损失下降），接受并 break
        """
        # 模拟生成有效 plan
        mock_plan = {'index_map': {1: {'method': 'method_A', 'group': (1, 0)}, 0: {'method': 'method_A', 'group': (1, 0)}, 3: {'method': 'unmatched', 'group': (3,)}, 2: {'method': 'unmatched', 'group': (2,)}, 4: {'method': 'unmatched', 'group': (4,)}}, 'valid_groups': {'method_A': [[1, 0]], 'method_B': []}, 'unmatched': [[3], [2], [4]]}
        
        self.trainer.path_generator.process_random_method.return_value = mock_plan

        # 模拟损失：第一次评估 base_loss = 1.0，添加后 new_loss = 0.5
        self.adaptoflux.loss_fn.side_effect = None
        self.adaptoflux.loss_fn.return_value = 1.0  # 第一次 base_loss
        self.adaptoflux.loss_fn.side_effect = [
            1.0,    # base_loss
            0.5     # new_loss after append
        ]

        result = self.trainer.train(self.input_data, self.target, max_layers=1)

        # 断言：append 被调用一次，remove 未被调用
        self.adaptoflux.append_nx_layer.assert_called_once()
        self.adaptoflux.remove_last_nx_layer.assert_not_called()

        # 断言：只尝试了一次，成功
        layer_record = result["attempt_history"][0]
        self.assertEqual(len(layer_record["attempts"]), 1)
        attempt = layer_record["attempts"][0]
        self.assertTrue(attempt["accepted"])
        self.assertEqual(attempt["status"], "accepted")
        self.assertEqual(result["layers_added"], 1)

    def test_reject_then_accept_on_second_attempt(self):
        """
        测试：第一次拒绝，第二次成功
        """
        mock_plan = {'index_map': {1: {'method': 'method_A', 'group': (1, 0)}, 0: {'method': 'method_A', 'group': (1, 0)}, 3: {'method': 'unmatched', 'group': (3,)}, 2: {'method': 'unmatched', 'group': (2,)}, 4: {'method': 'unmatched', 'group': (4,)}}, 'valid_groups': {'method_A': [[1, 0]], 'method_B': []}, 'unmatched': [[3], [2], [4]]}
        self.trainer.path_generator.process_random_method.return_value = mock_plan

        # 损失序列：base=1.0, 尝试1: 1.2（变差）, 尝试2: 0.6（变好）
        self.adaptoflux.loss_fn.side_effect = [
            1.0,    # base_loss
            1.2,    # 尝试1 -> 拒绝
            0.6     # 尝试2 -> 接受
        ]

        result = self.trainer.train(self.input_data, self.target, max_layers=1)

        # 断言：append 被调两次，remove 被调一次
        self.assertEqual(self.adaptoflux.append_nx_layer.call_count, 2)
        self.adaptoflux.remove_last_nx_layer.assert_called_once()

        # 断言：尝试了两次，第二次成功
        attempts = result["attempt_history"][0]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["status"], "rejected")
        self.assertEqual(attempts[1]["status"], "accepted")
        self.assertEqual(result["layers_added"], 1)

    def test_all_attempts_fail_then_stop(self):
        """
        测试：所有尝试都失败，停止生长
        """
        mock_plan = {'index_map': {1: {'method': 'method_A', 'group': (1, 0)}, 0: {'method': 'method_A', 'group': (1, 0)}, 3: {'method': 'unmatched', 'group': (3,)}, 2: {'method': 'unmatched', 'group': (2,)}, 4: {'method': 'unmatched', 'group': (4,)}}, 'valid_groups': {'method_A': [[1, 0]], 'method_B': []}, 'unmatched': [[3], [2], [4]]}
        self.trainer.path_generator.process_random_method.return_value = mock_plan

        # 所有 new_loss 都大于 base_loss
        self.adaptoflux.loss_fn.side_effect = lambda *args: 1.5  # 固定返回高损失

        result = self.trainer.train(self.input_data, self.target, max_layers=5)

        # 断言：只尝试了一层，失败后停止
        self.assertEqual(len(result["attempt_history"]), 1)
        self.assertEqual(result["layers_added"], 0)

        # append 和 remove 被调用次数应相等（每次 reject 都 rollback）
        self.assertEqual(self.adaptoflux.append_nx_layer.call_count, 3)
        self.assertEqual(self.adaptoflux.remove_last_nx_layer.call_count, 3)

    def test_append_layer_failure_handled(self):
        """
        测试：append_nx_layer 抛出异常时，跳过该尝试
        """
        mock_plan = {'index_map': {1: {'method': 'method_A', 'group': (1, 0)}, 0: {'method': 'method_A', 'group': (1, 0)}, 3: {'method': 'unmatched', 'group': (3,)}, 2: {'method': 'unmatched', 'group': (2,)}, 4: {'method': 'unmatched', 'group': (4,)}}, 'valid_groups': {'method_A': [[1, 0]], 'method_B': []}, 'unmatched': [[3], [2], [4]]}
        self.trainer.path_generator.process_random_method.return_value = mock_plan

        # 模拟第一次 append 失败
        self.adaptoflux.append_nx_layer.side_effect = [Exception("Simulated append error"), None]

        # 损失：第一次失败，第二次成功
        self.adaptoflux.loss_fn.side_effect = [
            1.0,    # base
            0.5     # 第二次成功
        ]

        result = self.trainer.train(self.input_data, self.target, max_layers=1)

        # 断言：第一次 append 失败，第二次成功
        self.assertEqual(self.adaptoflux.append_nx_layer.call_count, 2)
        self.assertEqual(self.adaptoflux.remove_last_nx_layer.call_count, 0)  # 第一次没成功，不需要回退；第二次成功也没回退

        attempts = result["attempt_history"][0]["attempts"]
        self.assertEqual(attempts[0]["status"], "append_failed: Simulated append error")
        self.assertEqual(attempts[1]["status"], "accepted")
        self.assertTrue(attempts[1]["accepted"])

    def test_rollback_failure_breaks_loop(self):
        """
        测试：remove_last_nx_layer 失败时，中断该层尝试
        """
        mock_plan = {"valid_groups": [("input", "node1", "Linear")]}
        self.trainer.path_generator.process_random_method.return_value = mock_plan

        # 第一次评估：损失变差，需要回退，但 remove 失败
        self.adaptoflux.loss_fn.side_effect = [1.0, 1.2]
        self.adaptoflux.remove_last_nx_layer.side_effect = Exception("Simulated rollback error")

        result = self.trainer.train(self.input_data, self.target, max_layers=1)

        # 断言：append 被调一次，remove 被调一次并失败
        self.adaptoflux.append_nx_layer.assert_called_once()
        self.adaptoflux.remove_last_nx_layer.assert_called_once()

        attempts = result["attempt_history"][0]["attempts"]
        self.assertEqual(attempts[0]["status"], "rollback_failed: Simulated rollback error")
        self.assertEqual(result["layers_added"], 0)


if __name__ == '__main__':
    unittest.main()