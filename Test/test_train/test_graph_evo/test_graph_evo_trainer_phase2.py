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
        return values + 0.1


class MockGraphProcessor:
    def __init__(self):
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
        print(f"[DEBUG] Replacing method at node '{node_name}' with '{new_method}'")
        self.graph.nodes[node_name]["method_name"] = new_method
        return node_name

    def get_graph_summary(self):
        """辅助函数：打印图结构摘要"""
        summary = []
        for node, data in self.graph.nodes(data=True):
            method = data.get('method_name', 'N/A')
            group = data.get('group', 'N/A')
            summary.append(f"  - {node}: method={method}, group={group}")
        return "\n".join(summary)


class TestGraphEvoRefinement(unittest.TestCase):

    def setUp(self):
        self.adaptoflux = MockAdaptoFlux()
        self.input_data = np.array([[1.0], [2.0], [3.0]])
        self.target = np.array([1, 0, 1])
        self.loss_fn = lambda pred, tgt: np.mean((pred.flatten() - tgt) ** 2)
        print("\n" + "="*60)
        print(f"SETUP: Input shape={self.input_data.shape}, Target={self.target}")

    def _print_graph_state(self, af, title="Graph State"):
        print(f"\n[INFO] {title}:")
        print(af.graph_processor.get_graph_summary())

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_refinement_random_single_with_improvement(self, mock_acc, mock_loss):
        print("\n>>> TEST: test_refinement_random_single_with_improvement")
        print("  Simulating loss: method_a → 0.5, method_b → 0.3 (better)")

        def side_effect_loss(instance, x, y):
            method_used = instance.graph_processor.graph.nodes["node1"]["method_name"]
            loss_val = 0.5 if method_used == "method_a" else 0.3
            print(f"    [LOSS] node1 uses '{method_used}' → loss = {loss_val:.2f}")
            return loss_val

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

        current_af = self.adaptoflux.clone()
        self._print_graph_state(current_af, "Before refinement")

        result = trainer._phase_node_refinement(current_af, self.input_data, self.target)

        print(f"  Result: improvement_made={result['improvement_made']}, steps_taken={result['steps_taken']}")
        self._print_graph_state(current_af, "After refinement")

        self.assertTrue(result['improvement_made'], "Expected improvement but none occurred")
        self.assertEqual(result['steps_taken'], 1, "Expected exactly 1 refinement step")
        
        final_method = current_af.graph_processor.graph.nodes["node1"]["method_name"]
        print(f"  Final method at node1: '{final_method}'")
        # 注意：由于是 random_single，可能优化 node1 或 node2；但至少有一个应变为 method_b
        # 为简化，我们假设 node1 被选中（实际中可加逻辑确保）
        self.assertIn(final_method, ["method_b"], "node1 should have been replaced with method_b for lower loss")

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_refinement_full_sweep_no_improvement(self, mock_acc, mock_loss):
        print("\n>>> TEST: test_refinement_full_sweep_no_improvement")
        print("  All methods yield same loss (0.5) → no improvement expected")

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
        self._print_graph_state(current_af, "Before refinement")

        result = trainer._phase_node_refinement(current_af, self.input_data, self.target)

        print(f"  Result: improvement_made={result['improvement_made']}, steps_taken={result['steps_taken']}")
        self.assertFalse(result['improvement_made'], "No improvement expected, but one was reported")
        self.assertEqual(result['steps_taken'], 0, "Expected 0 steps when no improvement possible")

    def test_frozen_nodes_are_excluded(self):
        print("\n>>> TEST: test_frozen_nodes_are_excluded")
        print("  Freezing 'node1' → only 'node2' should be active")

        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            frozen_nodes=["node1"],
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

        print(f"  Processing nodes: {processing_nodes}")
        print(f"  Frozen nodes: {trainer.frozen_nodes}")
        print(f"  Active nodes: {active_nodes}")

        self.assertEqual(active_nodes, ["node2"], "Only node2 should be active when node1 is frozen")

    def test_frozen_methods_are_excluded(self):
        print("\n>>> TEST: test_frozen_methods_are_excluded")
        print("  Freezing method 'method_b' → node2 (which uses it) should be excluded")

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

        print(f"  Processing nodes: {processing_nodes}")
        print(f"  Frozen methods: {trainer.frozen_methods}")
        print(f"  Nodes using frozen methods: {[n for n in processing_nodes if gp.graph.nodes[n].get('method_name') in trainer.frozen_methods]}")
        print(f"  Active nodes: {active_nodes}")

        self.assertEqual(active_nodes, ["node1"], "node2 uses method_b and should be frozen")

    @patch.object(GraphEvoTrainer, '_evaluate_loss_with_instance')
    @patch.object(GraphEvoTrainer, '_evaluate_accuracy_with_instance')
    def test_compatibility_mode_all(self, mock_acc, mock_loss):
        print("\n>>> TEST: test_compatibility_mode_all")
        print("  compatibility_mode='all' → should consider all methods including 'return_value'")

        mock_loss.return_value = 0.4
        mock_acc.return_value = 0.85

        trainer = GraphEvoTrainer(
            adaptoflux_instance=self.adaptoflux,
            refinement_strategy="random_single",
            compatibility_mode="all",
            verbose=False
        )
        trainer.loss_fn = self.loss_fn

        with patch.object(trainer, '_get_compatible_methods_for_node') as mock_compat:
            mock_compat.return_value = ["method_a", "method_b", "method_c", "return_value"]
            current_af = self.adaptoflux.clone()
            trainer._phase_node_refinement(current_af, self.input_data, self.target)
            mock_compat.assert_called()
            args, kwargs = mock_compat.call_args
            print(f"  _get_compatible_methods_for_node called with compatibility_mode='{kwargs['compatibility_mode']}'")
            self.assertEqual(kwargs['compatibility_mode'], "all")

    def test_empty_processing_nodes(self):
        print("\n>>> TEST: test_empty_processing_nodes")
        print("  Graph has only 'root' and 'collapse' → no processing nodes")

        af = MockAdaptoFlux()
        af.graph_processor.graph.clear()
        af.graph_processor.graph.add_node("root")
        af.graph_processor.graph.add_node("collapse")

        trainer = GraphEvoTrainer(adaptoflux_instance=af, verbose=False)
        result = trainer._phase_node_refinement(af, self.input_data, self.target)

        print(f"  Result: improvement_made={result['improvement_made']}, steps_taken={result['steps_taken']}")
        self.assertFalse(result['improvement_made'])
        self.assertEqual(result['steps_taken'], 0)


if __name__ == '__main__':
    print("🚀 Starting detailed GraphEvoTrainer refinement tests...\n")
    unittest.main(verbosity=2)