import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph_hierarchy(model_path: str, root: str = "root", figsize=(12, 8)):
    """
    从指定路径加载图（支持 .gexf 或 .json），并以指定 root 节点为根进行层次化可视化。

    参数:
        model_path (str): 图文件路径，支持 .gexf 或 .json（NetworkX node-link 格式）。
        root (str): 期望的根节点名称。若不存在，自动选取度最大的节点。
        figsize (tuple): 绘图区域大小。

    注意:
        - 若图不连通，仅保留包含 root 的连通分量。
        - 对于有向图，BFS 层次计算时会转为无向图以保证连通性。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载图
    if model_path.endswith('.gexf'):
        G = nx.read_gexf(model_path)
    elif model_path.endswith('.json'):
        import json
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
    else:
        raise ValueError("仅支持 .gexf 或 .json 格式的图文件")

    # 检查并确定 root 节点
    if root not in G:
        root = max(dict(G.degree()), key=lambda x: dict(G.degree())[x])
        print(f"指定的 root 节点 '{root}' 不存在，使用度最大的节点作为根: {root}")

    # 处理不连通图：仅保留 root 所在连通分量
    try:
        if G.is_directed():
            bfs_dist = nx.shortest_path_length(G.to_undirected(), source=root)
        else:
            bfs_dist = nx.shortest_path_length(G, source=root)
    except nx.NetworkXNoPath:
        print("图不连通，仅保留包含 root 的连通分量")
        undir_G = G.to_undirected() if G.is_directed() else G
        connected_nodes = nx.node_connected_component(undir_G, root)
        G = G.subgraph(connected_nodes).copy()
        undir_G = G.to_undirected() if G.is_directed() else G
        bfs_dist = nx.shortest_path_length(undir_G, source=root)

    # 按 BFS 距离分层
    layers = {}
    for node, dist in bfs_dist.items():
        layers.setdefault(dist, []).append(node)

    # 手动布局：每层节点均匀分布在 x 轴，y 轴为层数（负值使 root 在顶部）
    pos = {}
    for layer, nodes in layers.items():
        # 可选：对每层节点排序以提升可读性（如按 ID）
        nodes_sorted = sorted(nodes, key=str)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (i - len(nodes_sorted) / 2, -layer)  # 居中对齐

    # 节点颜色映射：按层深着色
    node_colors = [bfs_dist[node] for node in G.nodes]

    # 绘图
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        cmap='viridis',
        node_size=600,
        font_size=6,
        font_color='black',
        edge_color='gray',
        arrows=True if G.is_directed() else False,
        width=1.0,
        alpha=0.9
    )

    plt.title(f"Hierarchical Layout from Root: {root}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()