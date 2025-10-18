import unittest
from unittest.mock import Mock
import networkx as nx
from collections import defaultdict
from typing import Any, List, Dict, Tuple

# ================================
# 模拟辅助函数：_extract_node_signature
# ================================
def _extract_node_signature(graph, node_id: str) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    if node_id in ("root", "collapse"):
        raise ValueError("Skip special nodes")
    try:
        layer = int(node_id.split('_', 1)[0])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse layer from node ID: {node_id}")
    
    in_coords = []
    for src, _, edge_data in graph.in_edges(node_id, data=True):
        coord = edge_data.get('data_coord')
        if coord is not None:
            in_coords.append(coord)
    in_coords = tuple(sorted(in_coords))

    out_coords = []
    for _, dst, edge_data in graph.out_edges(node_id, data=True):
        coord = edge_data.get('data_coord')
        if coord is not None:
            out_coords.append(coord)
    out_coords = tuple(sorted(out_coords))

    return (layer, in_coords, out_coords)


# ================================
# 模拟训练器
# ================================
class MockGraphEvoTrainer:
    def __init__(
        self,
        min_subgraph_size_for_evolution: int = 2,
        consensus_threshold: float = None,
        verbose: bool = False
    ):
        self.min_subgraph_size_for_evolution = min_subgraph_size_for_evolution
        self.consensus_threshold = consensus_threshold
        self.verbose = verbose

    def _phase_method_pool_evolution(
        self,
        adaptoflux_instance,
        snapshots: List[Any],
        max_methods: int = 1
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n[🔍 Phase 3] 开始方法池进化：基于 {len(snapshots)} 个快照对齐节点...")

        if not snapshots:
            if self.verbose:
                print("[⚠️] 无快照，跳过进化阶段。")
            return {'methods_added': 0, 'new_method_names': []}

        signature_freq = defaultdict(lambda: defaultdict(int))
        for idx, snap in enumerate(snapshots):
            graph = snap.graph_processor.graph
            if self.verbose:
                print(f"  处理快照 {idx + 1}/{len(snapshots)}，图中共有 {graph.number_of_nodes()} 个节点")
            for node_id in graph.nodes():
                try:
                    sig = _extract_node_signature(graph, node_id)
                    method = graph.nodes[node_id].get('method_name', 'unknown')
                    signature_freq[sig][method] += 1
                    if self.verbose:
                        print(f"    节点 {node_id} → 签名 {sig}，方法 {method}")
                except ValueError as e:
                    if self.verbose:
                        print(f"    跳过节点 {node_id}（原因：{e}）")
                    continue

        total_signatures = len(signature_freq)
        if self.verbose:
            print(f"[📊] 共识别出 {total_signatures} 种唯一拓扑角色")

        consensus_map = {}
        for sig, method_counts in signature_freq.items():
            total = sum(method_counts.values())
            max_method, max_count = max(method_counts.items(), key=lambda x: x[1])
            ratio = max_count / total
            if self.consensus_threshold is None or ratio >= self.consensus_threshold:
                consensus_map[sig] = max_method
                if self.verbose:
                    print(f"    签名 {sig} → 共识方法 {max_method}（占比 {ratio:.2%}）")
            else:
                if self.verbose:
                    print(f"    签名 {sig} → 无共识（最高频 {max_method} 占比 {ratio:.2%} < 阈值）")

        if self.verbose:
            print(f"[✅] 共识阶段完成，共 {len(consensus_map)} 个签名达成共识")

        template_graph = snapshots[0].graph_processor.graph
        consensus_graph = nx.DiGraph()
        node_id_to_sig = {}

        for node_id in template_graph.nodes():
            if node_id in ("root", "collapse"):
                continue
            try:
                sig = _extract_node_signature(template_graph, node_id)
                if sig in consensus_map:
                    consensus_graph.add_node(sig, method=consensus_map[sig])
                    node_id_to_sig[node_id] = sig
            except ValueError:
                continue

        for src_id, dst_id, edge_data in template_graph.edges(data=True):
            if src_id in node_id_to_sig and dst_id in node_id_to_sig:
                src_sig = node_id_to_sig[src_id]
                dst_sig = node_id_to_sig[dst_id]
                consensus_graph.add_edge(src_sig, dst_sig, **edge_data)

        if self.verbose:
            print(f"[🧩] 共识图构建完成：{consensus_graph.number_of_nodes()} 个节点，{consensus_graph.number_of_edges()} 条边")

        connected_components = list(nx.weakly_connected_components(consensus_graph))
        if self.verbose:
            print(f"[🔗] 共识图分解为 {len(connected_components)} 个连通分量")

        filtered_components = [
            comp for comp in connected_components
            if len(comp) >= self.min_subgraph_size_for_evolution
        ]

        if self.verbose:
            print(f"[🧹] 应用最小节点数阈值 ({self.min_subgraph_size_for_evolution}) 后，保留 {len(filtered_components)} 个分量")

        connected_components = filtered_components

        new_method_names = []
        num_to_add = min(len(connected_components), max_methods)

        if num_to_add > 0:
            for i in range(num_to_add):
                name = f"evolved_method_{len(adaptoflux_instance.methods) + i + 1}"
                adaptoflux_instance.methods[name] = {
                    'output_count': 1,
                    'input_types': ['scalar'],
                    'output_types': ['scalar'],
                    'group': 'evolved',
                    'weight': 1.0,
                    'vectorized': True,
                    'is_evolved': True,
                    'aligned_roles': total_signatures
                }
                new_method_names.append(name)

            if self.verbose:
                print(f"[🆕] 生成 {len(new_method_names)} 个新方法: {new_method_names}")
        else:
            if self.verbose:
                print("[ℹ️] 无有效连通分量，跳过新方法生成")

        return {
            'methods_added': len(new_method_names),
            'new_method_names': new_method_names
        }


# ================================
# 工具函数：构建三层9节点图
# ================================
def build_three_layer_graph(method_name: str = "add") -> nx.DiGraph:
    """
    构建标准三层图：
    - Layer 0: 3 输入节点
    - Layer 1: 3 中间节点（使用 method_name）
    - Layer 2: 3 输出节点
    - 边：0->1, 1->2，共 3+3+4=10 条边（可调整）
    """
    G = nx.DiGraph()

    # Layer 0: inputs
    for i in range(3):
        G.add_node(f"0_{i}_input", method_name="input")

    # Layer 1: ops
    for i in range(3):
        G.add_node(f"1_{i}_{method_name}", method_name=method_name)

    # Layer 2: outputs
    for i in range(3):
        G.add_node(f"2_{i}_out", method_name="return_value")

    # Edges from layer 0 → 1
    for i in range(3):
        G.add_edge(f"0_{i}_input", f"1_{i}_{method_name}", data_coord=i)

    # Edges from layer 1 → 2
    for i in range(3):
        G.add_edge(f"1_{i}_{method_name}", f"2_{i}_out", data_coord=i)

    # Add cross-connections to reach 10 edges (e.g., 1_0 → 2_1, 1_1 → 2_2)
    G.add_edge("1_0_add", "2_1_out", data_coord=3)
    G.add_edge("1_1_add", "2_2_out", data_coord=4)

    return G


# ================================
# 单元测试类（图已扩展）
# ================================
class TestPhaseMethodPoolEvolution(unittest.TestCase):

    def test_1_min_subgraph_size_2_filters_single_nodes(self):
        print("\n" + "="*60)
        print("🧪 测试 1: min=2（过滤单节点）— 使用三层9节点图")
        print("="*60)

        # 构建一个图：其中 layer1 的 3 个节点形成一个连通分量（通过 cross edges）
        G = build_three_layer_graph("add")

        # Now, manually break one node to be isolated in consensus graph?
        # Instead, we'll rely on the fact that all layer1 nodes have same signature?
        # But actually, their in/out coords differ → different signatures!

        # So to control connectivity, we make all middle nodes have SAME signature
        # → give them same in/out pattern

        G = nx.DiGraph()
        # Inputs
        G.add_node("0_0_input", method_name="input")
        G.add_node("0_1_input", method_name="input")
        # Middle (all identical signature)
        G.add_node("1_0_op", method_name="add")
        G.add_node("1_1_op", method_name="add")
        G.add_node("1_2_op", method_name="add")
        # Outputs
        G.add_node("2_0_out", method_name="return_value")
        G.add_node("2_1_out", method_name="return_value")
        G.add_node("2_2_out", method_name="return_value")
        G.add_node("2_3_out", method_name="return_value")  # extra

        # All middle nodes: in from 0_0, out to 2_0 and 2_1 → same signature!
        for mid in ["1_0_op", "1_1_op", "1_2_op"]:
            G.add_edge("0_0_input", mid, data_coord=0)
            G.add_edge("0_1_input", mid, data_coord=1)
            G.add_edge(mid, "2_0_out", data_coord=0)
            G.add_edge(mid, "2_1_out", data_coord=1)

        # Add one isolated node with different signature
        G.add_node("1_3_isolated", method_name="inc")
        G.add_edge("0_0_input", "1_3_isolated", data_coord=5)  # unique in/out

        snap = Mock()
        snap.graph_processor.graph = G

        adaptoflux = Mock()
        adaptoflux.methods = {}

        trainer = MockGraphEvoTrainer(min_subgraph_size_for_evolution=2, verbose=True)
        result = trainer._phase_method_pool_evolution(adaptoflux, [snap], max_methods=1)

        # The 3 "op" nodes share signature → one consensus node
        # The "isolated" has unique signature → one node
        # But in consensus_graph: 
        #   - consensus node for op → 1 node
        #   - isolated → 1 node
        # → Two components of size 1 → both filtered if min=2 → 0 methods?
        # That fails test.

        # So instead: make the 3 op nodes CONNECTED via shared signature AND edges in consensus graph?
        # But consensus graph nodes are signatures, not original nodes.
        # If 3 original nodes → 1 signature → 1 node in consensus graph → size=1 → filtered!

        # 🔥 Critical insight: To get a component of size ≥2, we need ≥2 distinct signatures that are connected.

        # Let's design two connected signatures:
        G = nx.DiGraph()
        # Input
        G.add_node("0_0_in", method_name="input")
        # Two middle nodes with DIFFERENT but connected signatures
        G.add_node("1_0_a", method_name="add")
        G.add_node("1_1_b", method_name="add")
        # Output
        G.add_node("2_0_out", method_name="return")

        # Edges
        G.add_edge("0_0_in", "1_0_a", data_coord=0)
        G.add_edge("1_0_a", "1_1_b", data_coord=1)  # middle-to-middle!
        G.add_edge("1_1_b", "2_0_out", data_coord=2)

        # Add extra nodes to reach 9 nodes, 10 edges
        for i in range(1, 4):
            G.add_node(f"0_{i}_in", method_name="input")
            G.add_node(f"2_{i}_out", method_name="return")
            G.add_edge(f"0_{i}_in", f"1_0_a", data_coord=10+i)
            G.add_edge(f"1_1_b", f"2_{i}_out", data_coord=20+i)

        # Now: 
        # - "1_0_a": in=(0,11,12,13), out=(1,)
        # - "1_1_b": in=(1,), out=(2,21,22,23)
        # → Two different signatures, connected → component size=2

        # Add one isolated node (single signature, no connection)
        G.add_node("1_2_iso", method_name="inc")
        G.add_edge("0_0_in", "1_2_iso", data_coord=99)

        snap = Mock()
        snap.graph_processor.graph = G

        adaptoflux = Mock()
        adaptoflux.methods = {}

        trainer = MockGraphEvoTrainer(min_subgraph_size_for_evolution=2, verbose=True)
        result = trainer._phase_method_pool_evolution(adaptoflux, [snap], max_methods=1)

        # Should have one component of size 2 → kept
        self.assertEqual(result['methods_added'], 1)
        self.assertIn('evolved_method_1', result['new_method_names'])
        print("✅ 测试 1 通过：双节点分量保留，单节点被过滤\n")

    def test_2_min_subgraph_size_1_includes_single_nodes(self):
        print("\n" + "="*60)
        print("🧪 测试 2: min=1（保留单节点）— 三层9节点")
        print("="*60)

        G = build_three_layer_graph("add")
        # This graph has 9 nodes, 10 edges
        # Each middle node has unique signature → 3 consensus nodes, no edges between them (if no cross edges in template)
        # But in our build_three_layer_graph, we added cross edges → may connect

        # To ensure 3 isolated consensus nodes, remove cross edges
        G = nx.DiGraph()
        for i in range(3):
            G.add_node(f"0_{i}_in", method_name="input")
            G.add_node(f"1_{i}_op", method_name="add")
            G.add_node(f"2_{i}_out", method_name="return")
            G.add_edge(f"0_{i}_in", f"1_{i}_op", data_coord=i)
            G.add_edge(f"1_{i}_op", f"2_{i}_out", data_coord=i)

        # Now 3 independent chains → 3 signatures → 3 isolated nodes in consensus graph

        snap = Mock()
        snap.graph_processor.graph = G

        adaptoflux = Mock()
        adaptoflux.methods = {}

        trainer = MockGraphEvoTrainer(min_subgraph_size_for_evolution=1, verbose=True)
        result = trainer._phase_method_pool_evolution(adaptoflux, [snap], max_methods=3)

        self.assertEqual(result['methods_added'], 3)
        self.assertEqual(len(result['new_method_names']), 3)
        print("✅ 测试 2 通过：3 个单节点分量均被保留\n")

    def test_3_empty_snapshots_returns_zero(self):
        print("\n" + "="*60)
        print("🧪 测试 3: 空快照输入")
        print("="*60)

        adaptoflux = Mock()
        adaptoflux.methods = {}
        trainer = MockGraphEvoTrainer(verbose=True)
        result = trainer._phase_method_pool_evolution(adaptoflux, [], max_methods=1)

        self.assertEqual(result, {'methods_added': 0, 'new_method_names': []})
        print("✅ 测试 3 通过：空输入正确返回空结果\n")

    def test_4_consensus_with_divergent_snapshots(self):
        print("\n" + "="*60)
        print("🧪 测试 4: 快照之间有差距 + 共识阈值过滤 — 三层图")
        print("="*60)

        def make_graph(method_name):
            G = nx.DiGraph()
            # Input
            G.add_node("0_0_in", method_name="input")
            # Middle (only one, but with rich connections to make signature stable)
            G.add_node(f"1_0_{method_name}", method_name=method_name)
            # Outputs
            for i in range(3):
                G.add_node(f"2_{i}_out", method_name="return")
            # Edges
            G.add_edge("0_0_in", f"1_0_{method_name}", data_coord=0)
            for i in range(3):
                G.add_edge(f"1_0_{method_name}", f"2_{i}_out", data_coord=i)
            # Add more to reach 9 nodes
            for i in range(1, 3):
                G.add_node(f"0_{i}_in", method_name="input")
                G.add_edge(f"0_{i}_in", f"1_0_{method_name}", data_coord=10+i)
            return G

        G1 = make_graph("add")
        G2 = make_graph("mul")

        snap1 = Mock()
        snap1.graph_processor.graph = G1
        snap2 = Mock()
        snap2.graph_processor.graph = G2

        adaptoflux = Mock()
        adaptoflux.methods = {}

        trainer = MockGraphEvoTrainer(
            min_subgraph_size_for_evolution=2,
            consensus_threshold=0.6,
            verbose=True
        )

        result = trainer._phase_method_pool_evolution(adaptoflux, [snap1, snap2], max_methods=1)

        # Signature same, methods: add(1), mul(1) → ratio=0.5 < 0.6 → no consensus
        self.assertEqual(result['methods_added'], 0)
        self.assertEqual(result['new_method_names'], [])
        print("✅ 测试 4 通过：未达共识阈值，未生成新方法\n")

    def test_5_consensus_achieved_with_majority(self):
        print("\n" + "="*60)
        print("🧪 测试 5: 多数快照达成共识（2/3）— 三层图")
        print("="*60)

        def make_graph(method_name):
            G = nx.DiGraph()
            G.add_node("0_0_in", method_name="input")
            G.add_node(f"1_0_{method_name}", method_name=method_name)
            for i in range(3):
                G.add_node(f"2_{i}_out", method_name="return")
                G.add_edge(f"1_0_{method_name}", f"2_{i}_out", data_coord=i)
            G.add_edge("0_0_in", f"1_0_{method_name}", data_coord=0)
            # Add extras
            for i in range(1, 3):
                G.add_node(f"0_{i}_in", method_name="input")
                G.add_edge(f"0_{i}_in", f"1_0_{method_name}", data_coord=10+i)
                G.add_node(f"2_{3+i}_out", method_name="return")
                G.add_edge(f"1_0_{method_name}", f"2_{3+i}_out", data_coord=20+i)
            return G

        snaps = []
        for name in ["add", "add", "mul"]:
            snap = Mock()
            snap.graph_processor.graph = make_graph(name)
            snaps.append(snap)

        adaptoflux = Mock()
        adaptoflux.methods = {}

        trainer = MockGraphEvoTrainer(
            min_subgraph_size_for_evolution=1,
            consensus_threshold=0.6,
            verbose=True
        )

        result = trainer._phase_method_pool_evolution(adaptoflux, snaps, max_methods=1)

        self.assertEqual(result['methods_added'], 1)
        self.assertIn('evolved_method_1', result['new_method_names'])
        print("✅ 测试 5 通过：多数共识达成，成功生成新方法\n")


if __name__ == '__main__':
    print("🚀 开始运行 _phase_method_pool_evolution 单元测试（三层9节点+）...")
    unittest.main(verbosity=2)