import unittest
import numpy as np
import os
import shutil
import json
from unittest.mock import MagicMock, patch

from ATF.ModelTrainer.LayerGrowTrainer.layer_grow_trainer import LayerGrowTrainer


# ----------------------------
# Mock AdaptoFlux for testing
# ----------------------------
class MockAdaptoFlux:
    def __init__(self, is_classification=False):
        self.graph = {"nodes": [], "edges": []}
        self.methods = []
        self._layer_count = 0
        self.is_classification = is_classification  # 新增标志

    @property
    def loss_fn(self):
        if self.is_classification:
            # 用 cross-entropy 模拟，但为了简单，我们只用 MSE 但配合 accuracy
            return lambda pred, target: np.mean((pred - target) ** 2)
        else:
            return lambda pred, target: np.mean((pred - target) ** 2)

    def compute_accuracy(self, pred, target):
        if not self.is_classification:
            return None
        # 假设 pred 是 logits 或概率，target 是整数标签
        pred_labels = (pred.flatten() > 0.5).astype(int)
        return np.mean(pred_labels == target)

    def process_random_method(self):
        # 每次生成一个有效候选（简化）
        return {"valid_groups": [{"op": "linear", "params": {}}]}

    def append_nx_layer(self, candidate_plan, discard_unmatched='to_discard', discard_node_method_name="null"):
        self._layer_count += 1
        self.graph["nodes"].append(f"layer_{self._layer_count}")
        # 模拟可能失败（可选）
        if hasattr(self, '_fail_next_append') and self._fail_next_append:
            raise RuntimeError("Simulated append failure")

    def remove_last_nx_layer(self):
        if self._layer_count == 0:
            raise RuntimeError("No layer to remove")
        self._layer_count -= 1
        self.graph["nodes"].pop()

    def infer_with_graph(self, values):
        # 简单模拟：输出 = 输入 + 层数（用于区分不同结构）
        return values + float(self._layer_count)

    def infer_with_task_parallel(self, values, num_workers=4):
        return self.infer_with_graph(values)

    def save_model(self, folder):
        os.makedirs(folder, exist_ok=True)
        model_file = os.path.join(folder, "model.json")
        with open(model_file, 'w') as f:
            json.dump({
                "graph": self.graph,
                "methods": self.methods,
                "layer_count": self._layer_count
            }, f)



# ----------------------------
# Test Cases
# ----------------------------
class TestLayerGrowTrainer(unittest.TestCase):

    def setUp(self):
        self.input_data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        # 改为 target = input + 2，这样 layer 1 和 layer 2 都能改善 loss
        self.target = np.array([[3.0], [4.0], [5.0]], dtype=np.float32)  # ideal: output = input + 2
        self.adaptoflux = MockAdaptoFlux()
        self.trainer = LayerGrowTrainer(
            adaptoflux_instance=self.adaptoflux,
            max_attempts=3,
            decision_threshold=0.0,
            verbose=False
        )
        self.test_dir = "test_models"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_successful_layer_addition(self):
        """Test that layers are added when loss improves."""
        results = self.trainer.train(
            input_data=self.input_data,
            target=self.target,
            max_layers=2,
            save_model=False
        )
        self.assertEqual(results["layers_added"], 2)
        self.assertEqual(self.adaptoflux._layer_count, 2)

    def test_layer_rejection_when_no_improvement(self):
        """Force no improvement by making loss worse after adding layer."""
        # Override infer to make output worse
        original_infer = self.adaptoflux.infer_with_graph
        self.adaptoflux.infer_with_graph = lambda values: values - 10  # always bad

        results = self.trainer.train(
            input_data=self.input_data,
            target=self.target,
            max_layers=1,
            save_model=False
        )
        self.assertEqual(results["layers_added"], 0)
        self.assertEqual(self.adaptoflux._layer_count, 0)

        # restore
        self.adaptoflux.infer_with_graph = original_infer

    def test_rollback_on_failure(self):
        call_count = 0

        def mock_process_random_method():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 第一次：成功生成 layer 1
                return {"valid_groups": [{"op": "linear"}]}
            else:
                # 其他所有调用：无有效候选（模拟失败）
                return {"valid_groups": []}

        # 重置 adaptoflux（避免之前测试影响）
        adaptoflux = MockAdaptoFlux()
        trainer = LayerGrowTrainer(
            adaptoflux,
            max_attempts=2,          # 每层最多尝试 2 次
            verbose=False
        )

        with patch.object(adaptoflux, 'process_random_method', side_effect=mock_process_random_method):
            results = trainer.train(
                input_data=self.input_data,
                target=self.target,
                max_layers=2,
                on_retry_exhausted="rollback",
                rollback_layers=1,
                save_model=False,
                max_total_attempts=5  # 👈 防止无限循环！
            )

        # 预期：layer 1 成功添加，layer 2 尝试 2 次失败 → rollback 1 层 → 到 layer1，再失败两次，最终 0 层
        self.assertEqual(adaptoflux._layer_count, 0)
        self.assertEqual(results["layers_added"], 0)
        self.assertEqual(results["rollback_count"], 4)  # 应该只有 1 次 rollback

    def test_save_models(self):
        """Test saving final and best models."""
        results = self.trainer.train(
            input_data=self.input_data,
            target=self.target,
            max_layers=2,
            save_model=True,
            model_save_path=self.test_dir,
            save_best_model=True,
            final_model_subfolder="final",
            best_model_subfolder="best"
        )

        # Check files exist
        final_model_file = os.path.join(self.test_dir, "final", "model.json")
        best_model_file = os.path.join(self.test_dir, "best", "model.json")
        log_file = os.path.join(self.test_dir, "training_log.json")

        self.assertTrue(os.path.exists(final_model_file))
        self.assertTrue(os.path.exists(best_model_file))
        self.assertTrue(os.path.exists(log_file))

        # Check log content
        with open(log_file, 'r') as f:
            log = json.load(f)
        self.assertEqual(log["layers_added"], 2)
        self.assertIn("best_model_saved", log)

    def test_max_total_attempts_prevents_infinite_loop(self):
        with patch.object(self.adaptoflux, 'process_random_method', return_value={"valid_groups": []}):
            results = self.trainer.train(
                input_data=self.input_data,
                target=self.target,
                max_layers=10,
                max_total_attempts=5,
                on_retry_exhausted="stop",  # ✅ valid value
                save_model=False
            )
        self.assertLessEqual(results["total_growth_attempts"], 5)

    def test_accuracy_tracking(self):
        input_data = np.array([[0.1], [0.9], [0.85]], dtype=np.float32)
        target = np.array([[0], [1], [1]], dtype=np.float32)  # (3,1)

        adaptoflux = MockAdaptoFlux(is_classification=True)

        # 关键修改：让不同层数输出不同的值
        # - 0 层：输出偏离 target → loss 高
        # - 1 层：输出接近 target → loss 低，且 accuracy=1.0
        layer_outputs = {
            0: np.array([[0.6], [0.4], [0.4]]),  # 初始状态：全错！acc=0, loss 高
            1: np.array([[0.1], [0.9], [0.85]])  # 加 1 层后：全对！acc=1.0, loss 低
        }
        adaptoflux.infer_with_graph = lambda values: layer_outputs.get(adaptoflux._layer_count, values)

        trainer = LayerGrowTrainer(adaptoflux, max_attempts=1, decision_threshold=0.0, verbose=False)
        results = trainer.train(
            input_data=input_data,
            target=target,
            max_layers=1,
            save_model=True,  # 不需要保存文件
            model_save_path=self.test_dir,
            save_best_model=True
        )

        # 验证：层被成功添加，且 best_model_accuracy=1.0
        self.assertEqual(results["layers_added"], 1)  # 确保层被接受
        self.assertIn("best_model_accuracy", results)
        self.assertAlmostEqual(results["best_model_accuracy"], 1.0, places=4)

        # Check files exist
        final_model_file = os.path.join(self.test_dir, "final", "model.json")
        best_model_file = os.path.join(self.test_dir, "best", "model.json")
        log_file = os.path.join(self.test_dir, "training_log.json")

        self.assertTrue(os.path.exists(final_model_file))
        self.assertTrue(os.path.exists(best_model_file))
        self.assertTrue(os.path.exists(log_file))


if __name__ == '__main__':
    unittest.main()