import unittest
import numpy as np
import copy
from unittest.mock import Mock, patch, MagicMock
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer


class MockAdaptoFlux:
    """模拟 AdaptoFlux 实例，用于测试"""
    def __init__(self):
        self.graph_processor = MockGraphProcessor()
        self.methods = {
            "method_a": {"input_types": ["scalar"], "output_types": ["scalar"], "group": "math"},
            "method_b": {"input_types": ["scalar"], "output_types": ["scalar"], "group": "math"},
            "method_c": {"input_types": ["vector"], "output_types": ["vector"], "group": "linear"},
            "return_value": {"input_types": ["scalar"], "output_types": ["scalar"], "group": "output"},
        }

    def clone(self):
        return copy.deepcopy(self)

    def infer_with_graph(self, values):
        # 简单模拟输出
        return values + 0.1


class MockGraphProcessor:
    def __init__(self):
        # 构建一个简单图：root -> node1 -> node2 -> collapse
        import networkx as nx
        self.graph = nx.DiGraph()
        self.graph.add_node("root", method_name=None)
        self.graph.add_node("node1", method_name="method_a", group="math")
        self.graph.add_node("node2", method_name="method_b", group="math")
        self.graph.add_node("collapse", method_name="return_value", group="output")
        self.graph.add_edges_from([("root", "node1"), ("node1", "node2"), ("node2", "collapse")])

    def _is_processing_node(self, node):
        return node not in {"root", "collapse"}

    def replace_node_method(self, node_name, new_method):
        # 模拟替换方法并返回新节点 ID（实际中可能重命名，这里简化为原名）
        self.graph.nodes[node_name]["method_name"] = new_method
        return node_name


class TestGraphEvoRefinement(unittest.TestCase):

    def setUp(self):
        self.adaptoflux = MockAdaptoFlux()
        self.input_data = np.array([[1.0], [2.0], [3.0]])
        self.target = np.array([1, 0, 1])

        # 模拟损失函数（MSE）
        self.loss_fn = lambda pred, tgt: np.mean((pred.flatten() - tgt) ** 2)

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_refinement_random_single_with_improvement(self, mock_acc, mock_loss):
        # 模拟损失：method_a → loss=0.5, method_b → loss=0.3（更优）
        def side_effect_loss(instance, x, y):
            method_used = instance.graph_processor.graph.nodes["node1"]["method_name"]
            return 0.5 if method_used == "method_a" else 0.3
        mock_loss.side_effect = side_effect_loss
        mock_acc.return_value = 0.8

        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            num_initial_models=1,
            max_refinement_steps=10,
            refinement_strategy="random_single",
            compatibility_mode="group_with_fallback",
            verbose=False
        )
        trainer.loss_fn = self.loss_fn

        # 手动设置初始模型（跳过初始化阶段）
        current_af = self.adaptoflux.clone()
        result = trainer._phase_node_refinement(current_af, self.input_data, self.target)

        self.assertTrue(result['improvement_made'])
        self.assertEqual(result['steps_taken'], 1)
        # 检查 node1 是否被替换为 method_b（损失更低）
        self.assertIn(
            current_af.graph_processor.graph.nodes["node1"]["method_name"],
            ["method_b"]  # 可能选中 node2，但至少有一个被优化
        )

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_refinement_full_sweep_no_improvement(self, mock_acc, mock_loss):
        # 所有方法损失相同，无改进
        mock_loss.return_value = 0.5
        mock_acc.return_value = 0.7

        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            num_initial_models=1,
            max_refinement_steps=5,
            refinement_strategy="full_sweep",
            compatibility_mode="group_only",
            verbose=False
        )
        trainer.loss_fn = self.loss_fn

        current_af = self.adaptoflux.clone()
        result = trainer._phase_node_refinement(current_af, self.input_data, self.target)

        self.assertFalse(result['improvement_made'])
        self.assertEqual(result['steps_taken'], 0)

    def test_frozen_nodes_are_excluded(self):
        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            frozen_nodes=["node1"],
            refinement_strategy="random_single",
            verbose=False
        )

        # 检查内部逻辑：processing_nodes 应只含 node2
        gp = self.adaptoflux.graph_processor
        processing_nodes = [n for n in gp.graph.nodes() if gp._is_processing_node(n)]
        final_frozen = set(trainer.frozen_nodes)
        if trainer.frozen_methods:
            method_frozen = {n for n in processing_nodes if gp.graph.nodes[n].get('method_name') in trainer.frozen_methods}
            final_frozen.update(method_frozen)
        active_nodes = [n for n in processing_nodes if n not in final_frozen]

        self.assertEqual(active_nodes, ["node2"])

    def test_frozen_methods_are_excluded(self):
        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            frozen_methods=["method_b"],
            refinement_strategy="random_single",
            verbose=False
        )

        gp = self.adaptoflux.graph_processor
        processing_nodes = [n for n in gp.graph.nodes() if gp._is_processing_node(n)]
        final_frozen = set(trainer.frozen_nodes)
        if trainer.frozen_methods:
            method_frozen = {n for n in processing_nodes if gp.graph.nodes[n].get('method_name') in trainer.frozen_methods}
            final_frozen.update(method_frozen)
        active_nodes = [n for n in processing_nodes if n not in final_frozen]

        # node2 使用 method_b，应被冻结
        self.assertEqual(active_nodes, ["node1"])

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_compatibility_mode_all(self, mock_acc, mock_loss):
        mock_loss.return_value = 0.4
        mock_acc.return_value = 0.85

        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            refinement_strategy="random_single",
            compatibility_mode="all",  # 应包含所有方法
            verbose=False
        )
        trainer.loss_fn = self.loss_fn

        # Mock _get_compatible_methods_for_node to inspect call
        with patch.object(trainer, '_get_compatible_methods_for_node') as mock_compat:
            mock_compat.return_value = ["method_a", "method_b", "method_c", "return_value"]
            current_af = self.adaptoflux.clone()
            trainer._phase_node_refinement(current_af, self.input_data, self.target)
            mock_compat.assert_called()
            args, kwargs = mock_compat.call_args
            self.assertEqual(kwargs['compatibility_mode'], "all")

    def test_empty_processing_nodes(self):
        # 构建无 processing node 的图
        af = MockAdaptoFlux()
        af.graph_processor.graph.clear()
        af.graph_processor.graph.add_node("root")
        af.graph_processor.graph.add_node("collapse")

        trainer = GraphEvoTrainer(adaptoflux_instance=af, verbose=False)
        result = trainer._phase_node_refinement(af, self.input_data, self.target)
        self.assertFalse(result['improvement_made'])
        self.assertEqual(result['steps_taken'], 0)


if __name__ == '__main__':
    unittest.main()