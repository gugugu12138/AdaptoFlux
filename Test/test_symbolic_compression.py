# test_symbolic_compression_with_ports.py
import numpy as np
import networkx as nx
from ATF.ModelTrainer.GraphEvoTrainer import GraphEvoTrainer
from ATF.core.adaptoflux import AdaptoFlux
from ATF.methods.decorators import method_profile

# === 1. 定义方法 ===
@method_profile(input_types=['scalar', 'scalar'], output_types=['scalar', 'scalar'], output_count=2, group='math')
def add_values(a, b):
    return [a + b, a + b]

@method_profile(input_types=['scalar', 'scalar'], output_types=['scalar'], output_count=1, group='math')
def multiply_values(a, b):
    return [a * b]

@method_profile(input_types=['scalar', 'scalar'], output_types=['scalar'], output_count=1, group='math')
def fused_add_mul(a, b):
    return [(a + b) * (a + b)]


# === 2. 辅助：打印图结构 ===
def print_graph_structure(gp, title="Graph"):
    print(f"\n{'='*10} {title} {'='*10}")
    G = gp.graph
    print("Nodes:")
    for node in sorted(G.nodes()):
        attrs = G.nodes[node]
        if node in ("root", "collapse"):
            print(f"  {node} (interface)")
        else:
            method = attrs.get('method_name', 'N/A')
            print(f"  {node} → method_name='{method}'")
    
    print("\nEdges (src → dst):")
    for src, dst, key, data in G.edges(keys=True, data=True):
        # 格式化边属性
        attrs = ', '.join(f"{k}={v}" for k, v in data.items())
        print(f"  {src} ──[{key}]──> {dst} | {attrs}")
    print("="* (22 + len(title)) + "\n")


# === 3. 构建测试主图 ===
def build_test_graph(gp):
    graph = gp.graph
    # 清空（保留 root/collapse）
    for node in list(graph.nodes()):
        if node not in ("root", "collapse"):
            graph.remove_node(node)

    # 添加 add 节点
    add_node = "1_0_add_values"
    graph.add_node(add_node, method_name="add_values")
    graph.add_edge("root", add_node, data_type='scalar', output_index=0, input_slot=0)  # a
    graph.add_edge("root", add_node, data_type='scalar', output_index=1, input_slot=1)  # b

    # 添加 multiply 节点
    mul_node = "2_0_multiply_values"
    graph.add_node(mul_node, method_name="multiply_values")
    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=0, input_slot=0)
    graph.add_edge(add_node, mul_node, data_type='scalar', output_index=1, input_slot=1)

    # 输出到 collapse
    graph.add_edge(mul_node, "collapse", data_type='scalar', output_index=0, input_slot=0)

    gp.layer = 2
    print("=== Main Graph Built ===")


# === 4. 构造符号规则（Pattern + Replacement）===
def make_symbolic_rules():
    pattern = nx.MultiDiGraph()
    pattern.add_node("root")
    pattern.add_node("collapse")
    pattern.add_node("add", method_name="add_values")
    pattern.add_node("mul", method_name="multiply_values")
    pattern.add_edge("root", "add", port_name="input_a", input_slot=0, data_type='scalar')
    pattern.add_edge("root", "add", port_name="input_b", input_slot=1, data_type='scalar')
    pattern.add_edge("add", "mul", output_index=0, input_slot=0, data_type='scalar')
    pattern.add_edge("add", "mul", output_index=1, input_slot=1, data_type='scalar')
    pattern.add_edge("mul", "collapse", port_name="output", output_index=0, data_type='scalar')

    replacement = nx.MultiDiGraph()
    replacement.add_node("root")
    replacement.add_node("collapse")
    replacement.add_node("fused", method_name="fused_add_mul")
    replacement.add_edge("root", "fused", port_name="input_a", input_slot=0, data_type='scalar')
    replacement.add_edge("root", "fused", port_name="input_b", input_slot=1, data_type='scalar')
    replacement.add_edge("fused", "collapse", port_name="output", output_index=0, data_type='scalar')

    return [(pattern, replacement)]


# === 5. 测试函数 ===
def test_symbolic_compression_with_ports():
    X = np.array([[1.0, 2.0]])
    y = np.array([0.0])
    af = AdaptoFlux(values=X, labels=y, input_types_list=['scalar', 'scalar'])

    # 注册方法
    af.add_method("add_values", add_values, input_count=2, output_count=2, input_types=['scalar','scalar'], output_types=['scalar','scalar'], group='math', weight=1.0, vectorized=False)
    af.add_method("multiply_values", multiply_values, input_count=2, output_count=1, input_types=['scalar','scalar'], output_types=['scalar'], group='math', weight=1.0, vectorized=False)
    af.add_method("fused_add_mul", fused_add_mul, input_count=2, output_count=1, input_types=['scalar','scalar'], output_types=['scalar'], group='math', weight=1.0, vectorized=False)

    # 构建主图
    build_test_graph(af.graph_processor)
    print("原始图节点:", sorted(n for n in af.graph_processor.graph.nodes() if n not in ("root", "collapse")))
    print_graph_structure(af.graph_processor, "BEFORE Compression")

    # 创建 trainer
    trainer = GraphEvoTrainer(
        adaptoflux_instance=af,
        enable_compression=True,
        compression_mode="symbolic",
        symbolic_compression_rules=make_symbolic_rules(),
        enable_evolution=False,
        verbose=True
    )

    # 执行压缩
    result = trainer._phase_symbolic_compression(af, X, y)
    print("\n=== 压缩完成 ===")

    # 打印压缩后图
    print_graph_structure(af.graph_processor, "AFTER Compression")

    # 验证逻辑（保持不变）
    G = af.graph_processor.graph
    all_nodes = set(G.nodes())
    print("压缩后所有节点:", sorted(all_nodes))

    assert "1_0_add_values" not in all_nodes, "旧 add 节点未被删除"
    assert "2_0_multiply_values" not in all_nodes, "旧 mul 节点未被删除"

    new_nodes = [n for n in all_nodes if "fused_add_mul" in n and n not in ("root", "collapse")]
    assert len(new_nodes) == 1, f"期望1个新节点，实际 {new_nodes}"
    fused_node = new_nodes[0]
    print(f"✅ 新节点: {fused_node}")

    in_edges = list(G.in_edges(fused_node, data=True))
    assert len(in_edges) == 2, f"期望2条输入边，实际 {len(in_edges)}"
    input_slots = set()
    for src, _, data in in_edges:
        assert src == "root", f"输入边源不是 root: {src}"
        assert data['data_type'] == 'scalar'
        input_slots.add(data['input_slot'])
    assert input_slots == {0, 1}, f"输入槽位错误: {input_slots}"

    out_edges = list(G.out_edges(fused_node, data=True))
    assert len(out_edges) == 1, f"期望1条输出边，实际 {len(out_edges)}"
    _, dst, data = out_edges[0]
    assert dst == "collapse"
    assert data['output_index'] == 0
    assert data['data_type'] == 'scalar'

    print("✅ 所有验证通过！图结构符合预期。")


if __name__ == "__main__":
    test_symbolic_compression_with_ports()