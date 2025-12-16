# test_symbolic_compression_structure_only.py
import numpy as np
import networkx as nx
from ATF.ModelTrainer.GraphEvoTrainer import GraphEvoTrainer
from ATF.core.adaptoflux import AdaptoFlux
from ATF.methods.decorators import method_profile

# === 1. 定义方法（仅注册，不执行）===
@method_profile(input_types=['scalar','scalar'], output_types=['scalar','scalar'], output_count=2, group='math')
def add_values(a, b): return [a + b, a + b]

@method_profile(input_types=['scalar','scalar'], output_types=['scalar'], output_count=1, group='math')
def multiply_values(a, b): return [a * b]

@method_profile(input_types=['scalar','scalar'], output_types=['scalar'], output_count=1, group='math')
def fused_add_mul(a, b): return [(a + b) * (a + b)]

# === 2. 构建测试图 ===
def build_test_graph(gp):
    graph = gp.graph

    # 清空图
    nodes_to_remove = [n for n in graph.nodes() if n not in ("root", "collapse")]
    for n in nodes_to_remove:
        graph.remove_node(n)

    # 添加 add 节点
    add_node = "1_0_add_values"
    graph.add_node(add_node, method_name="add_values")
    graph.add_edge("root", add_node, data_type='scalar', output_index=0, data_coord=0)  # a
    graph.add_edge("root", add_node, data_type='scalar', output_index=1, data_coord=1)  # b


    # 添加 multiply 节点：需要两个输入！
    mul_node = "2_0_multiply_values"
    graph.add_node(mul_node, method_name="multiply_values")

    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=0, data_coord=0)
    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=1, data_coord=1)  # b

    # multiply → collapse
    graph.add_edge(mul_node, "collapse", data_type='scalar', output_index=0, data_coord=0)

    gp.layer = 2

    print("=== Main Graph ===")
    for n, attr in gp.graph.nodes(data=True):
        print(f"Node: {n}, attr: {attr}")
    for u, v, attr in gp.graph.edges(data=True):
        print(f"Edge: {u} -> {v}, attr: {attr}")

# === 3. 构造符号规则 ===
def make_rule():
    graph = nx.MultiDiGraph()
    add_node = "add"
    mul_node = "mul"
    graph.add_node(add_node, method_name="add_values")
    graph.add_node(mul_node, method_name="multiply_values")
    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=0, data_coord=0)
    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=1, data_coord=1)

    print("=== Rule Graph ===")
    for n, attr in graph.nodes(data=True):
        print(f"Node: {n}, attr: {attr}")
    for u, v, attr in graph.edges(data=True):
        print(f"Edge: {u} -> {v}, attr: {attr}")
    return [(graph, "fused_add_mul")]

# === 4. 测试主函数 ===
def test_symbolic_compression_structure():
    # 初始化 AdaptoFlux（只需类型信息）
    X = np.array([[0.0, 0.0]])
    y = np.array([0.0])
    af = AdaptoFlux(values=X, labels=y, input_types_list=['scalar', 'scalar'])

    # 注册方法
    af.add_method("add_values", add_values, input_count=2, output_count=2, input_types=['scalar','scalar'], output_types=['scalar','scalar'], group='math', weight=1.0, vectorized=False)
    af.add_method("multiply_values", multiply_values, input_count=2, output_count=1, input_types=['scalar','scalar'], output_types=['scalar'],  group='math', weight=1.0, vectorized=False)
    af.add_method("fused_add_mul", fused_add_mul, input_count=2, output_count=1, input_types=['scalar','scalar'], output_types=['scalar'], group='math', weight=1.0, vectorized=False)

    # 构建图
    build_test_graph(af.graph_processor)
    print("原始图节点:", sorted(af.graph_processor.graph.nodes()))

    # 创建 trainer（跳过所有训练阶段）
    trainer = GraphEvoTrainer(
        adaptoflux_instance=af,
        enable_compression=True,
        compression_mode="symbolic",
        symbolic_compression_rules=make_rule(),
        enable_evolution=False,
        verbose=True
    )

    # ⚠️ 关键：不调用 train()，而是直接调用压缩阶段
    # 传 dummy input（仅用于边 I/O 提取，不用于数值计算）
    dummy_X = np.array([[1.0, 2.0]])

    # 直接执行符号压缩
    result = trainer._phase_symbolic_compression(af, dummy_X, np.array([0.0]))

    # 验证图结构
    G = af.graph_processor.graph
    nodes = set(G.nodes())
    print("压缩后图节点:", sorted(nodes))

    # ✅ 检查旧节点是否被删除
    assert "1_0_add_values" not in nodes, "旧 add 节点未被删除"
    assert "2_0_multiply_values" not in nodes, "旧 mul 节点未被删除"

    # ✅ 检查新节点是否存在（格式：{layer}_{index}_fused_add_mul）
    new_nodes = [n for n in nodes if "fused_add_mul" in n]
    assert len(new_nodes) == 1, f"期望1个新节点，实际{len(new_nodes)}"
    new_node = new_nodes[0]
    print("✅ 新节点:", new_node)

    # ✅ 检查输入边（root → new_node）
    in_edges = list(G.in_edges(new_node, data=True))
    assert len(in_edges) == 2, f"期望2条输入边，实际{len(in_edges)}"
    for src, _, data in in_edges:
        assert src == "root"
        assert data['data_type'] == 'scalar'
        assert data['output_index'] in {0, 1}
        assert data['data_coord'] in {0, 1}

    # ✅ 检查输出边（new_node → collapse）
    out_edges = list(G.out_edges(new_node, data=True))
    assert len(out_edges) == 1, f"期望1条输出边，实际{len(out_edges)}"
    _, dst, data = out_edges[0]
    assert dst == "collapse"
    assert data['data_type'] == 'scalar'
    assert data['output_index'] == 0
    assert data['data_coord'] == 0

    print("✅ 所有图结构验证通过！")

if __name__ == "__main__":
    test_symbolic_compression_structure()