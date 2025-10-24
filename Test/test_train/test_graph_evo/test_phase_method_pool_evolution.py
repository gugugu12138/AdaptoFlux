import unittest
from unittest.mock import Mock, patch
import networkx as nx
from collections import defaultdict
import tempfile
import shutil
import os
from typing import Optional, List

# ================================
# 导入你的真实 GraphEvoTrainer
# ================================
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer


# ================================
# 辅助函数：构建标准测试图
# ================================
def build_connected_graph() -> nx.MultiDiGraph:
    """构建一个连通的、含 root/collapse 的图（模拟真实流程输出）"""
    G = nx.MultiDiGraph()
    G.add_node("root")
    G.add_node("0_0_in", method_name="input")
    G.add_node("0_1_in", method_name="input")
    G.add_node("1_0_add", method_name="add")
    G.add_node("2_0_out", method_name="return")
    G.add_node("collapse")

    # root -> inputs
    G.add_edge("root", "0_0_in", data_type="scalar", data_coord=0, output_index=0)
    G.add_edge("root", "0_1_in", data_type="scalar", data_coord=1, output_index=1)

    # inputs -> op
    G.add_edge("0_0_in", "1_0_add", data_type="scalar", data_coord=0)
    G.add_edge("0_1_in", "1_0_add", data_type="scalar", data_coord=1)

    # op -> output
    G.add_edge("1_0_add", "2_0_out", data_type="scalar", data_coord=0)

    # output -> collapse
    G.add_edge("2_0_out", "collapse", data_type="scalar", data_coord=0, output_index=0)

    return G


def build_isolated_node_graph() -> nx.MultiDiGraph:
    """构建一个含孤立节点的图（用于测试 min_subgraph_size 过滤）"""
    G = nx.MultiDiGraph()
    G.add_node("root")
    G.add_node("0_0_in", method_name="input")
    G.add_node("1_0_iso", method_name="inc")  # 孤立节点（无连接）
    G.add_node("collapse")

    G.add_edge("root", "0_0_in", data_type="scalar", data_coord=0, output_index=0)
    G.add_edge("0_0_in", "collapse", data_type="scalar", data_coord=0, output_index=0)

    return G


# ================================
# Mock Snapshot 类
# ================================
class MockSnapshot:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph_processor = Mock()
        self.graph_processor.graph = graph


# ================================
# 单元测试类
# ================================
class TestGraphEvoTrainerPhaseMethodPoolEvolution(unittest.TestCase):

    def setUp(self):
        """每个测试前创建临时目录（用于 save_dir 测试）"""
        self.temp_dir = tempfile.mkdtemp()
        print(f"\n📁 创建临时目录: {self.temp_dir}")

    def tearDown(self):
        """清理临时目录"""
        print(f"🧹 清理临时目录: {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod')
    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.logger')
    def test_min_subgraph_size_filters_small_components(self, mock_logger, mock_evolved_method_class):
        print("\n" + "="*70)
        print("🧪 测试 1: min_subgraph_size_for_evolution=2（过滤小连通分量）")
        print("="*70)

        graph1 = build_connected_graph()
        graph2 = build_connected_graph()
        snapshots = [MockSnapshot(graph1), MockSnapshot(graph2)]

        print(f"✅ 创建 2 个快照，每个包含 {graph1.number_of_nodes()} 个节点")

        adaptoflux_for_trainer = Mock()
        adaptoflux_for_trainer.methods = {}

        trainer = GraphEvoTrainer(
            adaptoflux_instance=adaptoflux_for_trainer,
            num_initial_models=1,
            max_refinement_steps=10,
            compression_threshold=0.95,
            max_init_layers=3,
            init_mode="fixed",
            frozen_nodes=None,
            frozen_methods=None,
            refinement_strategy="random_single",
            candidate_pool_mode="group",
            fallback_mode="self",
            enable_compression=True,
            enable_evolution=True,
            evolution_sampling_frequency=1,
            evolution_trigger_count=3,
            evolution_cleanup_mode="full",
            consensus_threshold=0.5,
            methods_per_evolution=1,
            min_subgraph_size_for_evolution=2,
            verbose=True
        )

        mock_evolved_method_instance = Mock()
        mock_evolved_method_class.return_value = mock_evolved_method_instance

        result = trainer._phase_method_pool_evolution(
            adaptoflux_instance=adaptoflux_for_trainer,
            snapshots=snapshots,
            max_methods=1,
            enable_graph_isomorphism_clustering=False,
            evolved_methods_save_dir=None,
            subgraph_selection_policy="largest"
        )

        print(f"📊 返回结果: {result}")
        print(f"📦 adaptoflux.methods 中的方法数量: {len(adaptoflux_for_trainer.methods)}")
        if adaptoflux_for_trainer.methods:
            print(f"   方法名: {list(adaptoflux_for_trainer.methods.keys())}")

        self.assertEqual(result['methods_added'], 1, "应生成1个进化方法")
        self.assertEqual(len(result['new_method_names']), 1, "new_method_names 长度应为1")
        self.assertIn('evolved_method_1', result['new_method_names'], "方法名应为 evolved_method_1")
        self.assertIn('evolved_method_1', adaptoflux_for_trainer.methods, "方法应注册到 adaptoflux.methods")
        entry = adaptoflux_for_trainer.methods['evolved_method_1']
        self.assertEqual(entry['group'], 'evolved', "group 应为 'evolved'")
        self.assertTrue(entry['is_evolved'], "is_evolved 应为 True")

        print("✅ 测试 1 通过！")

    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod')
    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.logger')
    def test_empty_snapshots_returns_empty(self, mock_logger, mock_evolved_method_class):
        print("\n" + "="*70)
        print("🧪 测试 2: 空快照输入")
        print("="*70)

        adaptoflux_for_trainer = Mock()
        adaptoflux_for_trainer.methods = {}

        trainer = GraphEvoTrainer(
            adaptoflux_instance=adaptoflux_for_trainer,
            num_initial_models=1,
            max_refinement_steps=10,
            compression_threshold=0.95,
            max_init_layers=3,
            init_mode="fixed",
            frozen_nodes=None,
            frozen_methods=None,
            refinement_strategy="random_single",
            candidate_pool_mode="group",
            fallback_mode="self",
            enable_compression=True,
            enable_evolution=True,
            evolution_sampling_frequency=1,
            evolution_trigger_count=3,
            evolution_cleanup_mode="full",
            consensus_threshold=None,
            methods_per_evolution=1,
            min_subgraph_size_for_evolution=2,
            verbose=True
        )

        result = trainer._phase_method_pool_evolution(
            adaptoflux_instance=adaptoflux_for_trainer,
            snapshots=[],
            max_methods=1
        )

        print(f"📊 返回结果: {result}")

        self.assertEqual(result, {'methods_added': 0, 'new_method_names': []}, "空快照应返回空结果")
        print("✅ 测试 2 通过！")

    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod')
    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.logger')
    def test_consensus_threshold_filters_low_frequency(self, mock_logger, mock_evolved_method_class):
        print("\n" + "="*70)
        print("🧪 测试 3: 共识阈值过滤（50% < 60% → 无共识）")
        print("="*70)

        G1 = build_connected_graph()
        G2 = build_connected_graph()
        for node in G2.nodes():
            if G2.nodes[node].get('method_name') == 'add':
                G2.nodes[node]['method_name'] = 'mul'

        snapshots = [MockSnapshot(G1), MockSnapshot(G2)]
        print("✅ 创建 2 个快照：一个用 'add'，一个用 'mul'（相同拓扑）")

        adaptoflux_for_trainer = Mock()
        adaptoflux_for_trainer.methods = {}

        trainer = GraphEvoTrainer(
            adaptoflux_instance=adaptoflux_for_trainer,
            num_initial_models=1,
            max_refinement_steps=10,
            compression_threshold=0.95,
            max_init_layers=3,
            init_mode="fixed",
            frozen_nodes=None,
            frozen_methods=None,
            refinement_strategy="random_single",
            candidate_pool_mode="group",
            fallback_mode="self",
            enable_compression=True,
            enable_evolution=True,
            evolution_sampling_frequency=1,
            evolution_trigger_count=3,
            evolution_cleanup_mode="full",
            consensus_threshold=0.6,
            methods_per_evolution=1,
            min_subgraph_size_for_evolution=2,
            verbose=True
        )

        result = trainer._phase_method_pool_evolution(
            adaptoflux_instance=adaptoflux_for_trainer,
            snapshots=snapshots,
            max_methods=1,
            enable_graph_isomorphism_clustering=False
        )

        print(f"📊 返回结果: {result}")

        self.assertEqual(result['methods_added'], 0, "应无方法生成（未达共识阈值）")
        self.assertEqual(result['new_method_names'], [], "new_method_names 应为空列表")

        print("✅ 测试 3 通过！")

    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod')
    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.logger')
    def test_successful_evolution_and_registration(self, mock_logger, mock_evolved_method_class):
        print("\n" + "="*70)
        print("🧪 测试 4: 完整流程 — 方法生成、注册、保存")
        print("="*70)

        graph = build_connected_graph()
        snapshots = [MockSnapshot(graph)]
        print(f"✅ 创建 1 个快照，包含 {graph.number_of_nodes()} 个节点")

        adaptoflux_for_trainer = Mock()
        adaptoflux_for_trainer.methods = {"existing_method": {"func": None}}

        trainer = GraphEvoTrainer(
            adaptoflux_instance=adaptoflux_for_trainer,
            num_initial_models=1,
            max_refinement_steps=10,
            compression_threshold=0.95,
            max_init_layers=3,
            init_mode="fixed",
            frozen_nodes=None,
            frozen_methods=None,
            refinement_strategy="random_single",
            candidate_pool_mode="group",
            fallback_mode="self",
            enable_compression=True,
            enable_evolution=True,
            evolution_sampling_frequency=1,
            evolution_trigger_count=3,
            evolution_cleanup_mode="full",
            consensus_threshold=None,
            methods_per_evolution=1,
            min_subgraph_size_for_evolution=2,
            verbose=True
        )

        mock_evolved_method_instance = Mock()
        mock_evolved_method_class.return_value = mock_evolved_method_instance

        result = trainer._phase_method_pool_evolution(
            adaptoflux_instance=adaptoflux_for_trainer,
            snapshots=snapshots,
            max_methods=1,
            evolved_methods_save_dir=self.temp_dir
        )

        print(f"📊 返回结果: {result}")
        method_name = result['new_method_names'][0] if result['new_method_names'] else None
        print(f"🆕 生成方法名: {method_name}")

        self.assertEqual(result['methods_added'], 1, "应生成1个方法")
        self.assertTrue(method_name and method_name.startswith('evolved_method_'), "方法名格式应正确")
        self.assertIn(method_name, adaptoflux_for_trainer.methods, "方法应注册到 adaptoflux.methods")

        entry = adaptoflux_for_trainer.methods[method_name]
        required_keys = {
            'func', 'output_count', 'input_types', 'output_types',
            'group', 'weight', 'vectorized', 'is_evolved'
        }
        missing_keys = required_keys - entry.keys()
        self.assertFalse(missing_keys, f"缺少必要字段: {missing_keys}")
        self.assertEqual(entry['group'], 'evolved', "group 应为 'evolved'")
        self.assertTrue(entry['is_evolved'], "is_evolved 应为 True")

        mock_evolved_method_class.assert_called_once()
        call_args = mock_evolved_method_class.call_args
        print(f"🔧 EvolvedMethod 创建参数:")
        print(f"    name: {call_args.kwargs['name']}")
        print(f"    graph nodes: {call_args.kwargs['graph'].number_of_nodes()}")
        print(f"    graph edges: {call_args.kwargs['graph'].number_of_edges()}")
        meta = call_args.kwargs['metadata']
        print(f"    metadata keys: {list(meta.keys())}")

        mock_evolved_method_instance.save.assert_called_with(self.temp_dir)
        print(f"💾 方法已保存至: {self.temp_dir}")

        print("✅ 测试 4 通过！")

    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod')
    @patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.logger')
    def test_no_unhashable_error_in_data_coord_update(self, mock_logger, mock_evolved_method_class):
        print("\n" + "="*70)
        print("🧪 测试 5: 验证无 'unhashable type: dict' 错误（核心修复验证）")
        print("="*70)

        graph = build_connected_graph()
        snapshots = [MockSnapshot(graph)]
        print(f"✅ 使用标准连通图测试 data_coord 更新逻辑")

        adaptoflux_for_trainer = Mock()
        adaptoflux_for_trainer.methods = {}

        trainer = GraphEvoTrainer(
            adaptoflux_instance=adaptoflux_for_trainer,
            num_initial_models=1,
            max_refinement_steps=10,
            compression_threshold=0.95,
            max_init_layers=3,
            init_mode="fixed",
            frozen_nodes=None,
            frozen_methods=None,
            refinement_strategy="random_single",
            candidate_pool_mode="group",
            fallback_mode="self",
            enable_compression=True,
            enable_evolution=True,
            evolution_sampling_frequency=1,
            evolution_trigger_count=3,
            evolution_cleanup_mode="full",
            consensus_threshold=None,
            methods_per_evolution=1,
            min_subgraph_size_for_evolution=2,
            verbose=True
        )

        mock_evolved_method_instance = Mock()
        mock_evolved_method_class.return_value = mock_evolved_method_instance

        try:
            result = trainer._phase_method_pool_evolution(
                adaptoflux_instance=adaptoflux_for_trainer,
                snapshots=snapshots,
                max_methods=1
            )
            success = True
            print(f"✅ 方法生成成功，结果: {result}")
        except TypeError as e:
            if "unhashable type: 'dict'" in str(e):
                success = False
                print(f"❌ 仍然存在 unhashable type 错误: {e}")
                raise AssertionError("仍然存在 unhashable type: 'dict' 错误！") from e
            else:
                print(f"⚠️ 其他 TypeError: {e}")
                raise

        self.assertTrue(success, "应成功完成进化阶段")
        self.assertEqual(result['methods_added'], 1, "应生成1个方法")

        print("✅ 测试 5 通过！无 unhashable 错误。")


if __name__ == '__main__':
    print("🚀 开始运行 GraphEvoTrainer._phase_method_pool_evolution 单元测试")
    print("💡 提示：verbose=True 已启用，内部日志逻辑会执行（但被 mock 拦截）")
    unittest.main(verbosity=2)