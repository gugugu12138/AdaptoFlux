# test_phase_method_pool_evolution.py
import os
import tempfile
from unittest.mock import MagicMock, patch
import networkx as nx
from datetime import datetime

# ====== 替换为你的真实类路径 ======
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from ATF.ModelTrainer.GraphEvoTrainer.method_pool_evolver import MethodPoolEvolver


class MockSnapshot:
    def __init__(self, graph):
        self.graph_processor = MagicMock()
        self.graph_processor.graph = graph


class MockAdaptoFluxInstance:
    def __init__(self):
        self.methods = {}
        self.path_generator = MagicMock()
        self.graph_processor = MagicMock()
        self.graph = nx.MultiDiGraph()

    def _create_graph_processor(self, graph):
        gp = MagicMock()
        gp.layer = getattr(self.graph_processor, 'layer', 0)
        return gp


class MockEvolvedMethod:
    def __init__(self, name, graph, methods_ref, metadata):
        self.name = name
        self.graph = graph
        self.metadata = metadata

    def save(self, dir_path):
        path = os.path.join(dir_path, f"{self.name}.pkl")
        with open(path, 'w') as f:
            f.write("mock saved")


# ----------------------------
# Helper: 创建最小 GraphEvoTrainer 实例（用于调用方法）
# ----------------------------

def make_minimal_trainer():
    trainer = GraphEvoTrainer.__new__(GraphEvoTrainer)  # 绕过 __init__
    trainer.method_pool_evolver = MethodPoolEvolver(trainer)  # 注入 evolver
    trainer.verbose = False
    trainer.consensus_threshold = 0.6
    trainer.min_subgraph_size_for_evolution = 1
    # Mock 依赖方法
    trainer.method_pool_evolver._extract_node_signature = lambda g, nid: (1, (0,), (0,))
    return trainer


# ----------------------------
# 测试用例
# ----------------------------

def test_empty_snapshots():
    trainer = make_minimal_trainer()
    instance = MockAdaptoFluxInstance()
    result = trainer._phase_method_pool_evolution(
        adaptoflux_instance=instance,
        snapshots=[],
        max_methods=1
    )
    # 验证必需字段存在且值正确（允许额外字段）
    assert result.get('methods_added') == 0, f"Expected methods_added=0, got {result.get('methods_added')}"
    assert result.get('new_method_names') == [], f"Expected new_method_names=[], got {result.get('new_method_names')}"
    # 可选：验证至少包含这两个字段（防止字段缺失）
    assert 'methods_added' in result, "Missing required field: methods_added"
    assert 'new_method_names' in result, "Missing required field: new_method_names"


@patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod', MockEvolvedMethod)
def test_single_node_consensus():
    # 构建图
    G = nx.MultiDiGraph()
    G.add_node("root")
    G.add_node("A", method_name="add", is_passthrough=False, layer=1)
    G.add_node("collapse")
    G.add_edge("root", "A", data_type='scalar', data_coord=0, output_index=0)
    G.add_edge("A", "collapse", data_type='scalar', data_coord=0, output_index=0)

    snap = MockSnapshot(G)
    instance = MockAdaptoFluxInstance()
    trainer = make_minimal_trainer()
    trainer.consensus_threshold = 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        result = trainer._phase_method_pool_evolution(
            adaptoflux_instance=instance,
            snapshots=[snap],
            max_methods=1,
            evolved_methods_save_dir=tmpdir
        )

    assert result['methods_added'] == 1
    name = result['new_method_names'][0]
    assert name in instance.methods
    assert instance.methods[name]['is_evolved'] is True


@patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod', MockEvolvedMethod)
def test_consensus_threshold_filtering():
    def make_graph(method):
        G = nx.MultiDiGraph()
        G.add_nodes_from(["root", "A", "collapse"])
        G.nodes["A"]['method_name'] = method
        G.add_edge("root", "A", data_type='scalar', data_coord=0, output_index=0)
        G.add_edge("A", "collapse", data_type='scalar', data_coord=0, output_index=0)
        return G

    snaps = [
        MockSnapshot(make_graph("add")),
        MockSnapshot(make_graph("mul")),
        MockSnapshot(make_graph("add")),
    ]

    instance = MockAdaptoFluxInstance()
    trainer = make_minimal_trainer()
    trainer.consensus_threshold = 0.6

    result = trainer._phase_method_pool_evolution(
        adaptoflux_instance=instance,
        snapshots=snaps,
        max_methods=1
    )

    assert result['methods_added'] == 1
    name = result['new_method_names'][0]

@patch('ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer.EvolvedMethod', MockEvolvedMethod)
def test_graph_isomorphism_clustering_largest():
    def make_big():
        G = nx.MultiDiGraph()
        G.add_node("X", method_name="op1")
        G.add_node("Y", method_name="op2")
        G.add_edge("X", "Y")
        return G

    small = nx.MultiDiGraph()
    small.add_node("Z", method_name="small")

    def assemble_with_both():
        G = nx.MultiDiGraph()
        G.add_nodes_from(["root", "collapse"])
        # 合并 big 和 small（无连接）
        big = make_big()
        G = nx.compose(G, big)
        G = nx.compose(G, small)
        # 连接 root -> X, root -> Z（两个入口）
        G.add_edge("root", "X", data_type='scalar', data_coord=0, output_index=0)
        G.add_edge("root", "Z", data_type='scalar', data_coord=1, output_index=1)
        # 连接 Y -> collapse, Z -> collapse（两个出口）
        G.add_edge("Y", "collapse", data_type='scalar', data_coord=0, output_index=0)
        G.add_edge("Z", "collapse", data_type='scalar', data_coord=1, output_index=1)
        return G

    # 所有快照都包含 big + small
    full1 = assemble_with_both()
    full2 = assemble_with_both()
    full3 = assemble_with_both()

    snaps = [MockSnapshot(full1), MockSnapshot(full2), MockSnapshot(full3)]

    instance = MockAdaptoFluxInstance()
    trainer = make_minimal_trainer()
    trainer.consensus_threshold = 0.5  # 50%
    trainer.min_subgraph_size_for_evolution = 1

    # 唯一签名（避免冲突）
    node_to_sig = {
        "X": (1, (0,), (0,)),
        "Y": (2, (0,), (0,)),
        "Z": (1, (1,), (1,)),  # 注意：data_coord=1，与 X 不同
    }
    trainer.method_pool_evolver._extract_node_signature = lambda g, nid: node_to_sig.get(nid, (0, (), ()))

    result = trainer._phase_method_pool_evolution(
        adaptoflux_instance=instance,
        snapshots=snaps,
        max_methods=2,
        enable_graph_isomorphism_clustering=True,
        subgraph_selection_policy="largest"
    )

    assert result['methods_added'] == 2
    names = result['new_method_names']
    g1 = instance.methods[names[0]]['function'].graph
    g2 = instance.methods[names[1]]['function'].graph
    internal1 = len([n for n in g1.nodes() if n not in ("root", "collapse")])
    internal2 = len([n for n in g2.nodes() if n not in ("root", "collapse")])
    assert internal1 > internal2