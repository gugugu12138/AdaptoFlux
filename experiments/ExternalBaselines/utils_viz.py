# experiments/ExternalBaselines/utils_viz.py
import networkx as nx
import matplotlib.pyplot as plt
import os
import json

def visualize_graph_hierarchy(model_path: str, output_image_path: str, root: str = "root", figsize=(12, 8)):
    """
    从指定路径加载图，并保存可视化图像。
    
    新增参数:
        output_image_path (str): 保存图像的路径（如 .png）
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载图（AdaptoFlux 默认保存为 .json 格式）
    if model_path.endswith('.json'):
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.node_link_graph(data, edges="edges")
    else:
        raise ValueError("仅支持 .json 格式的图文件（AdaptoFlux 默认格式）")

    # 确定 root
    if root not in G:
        root = max(dict(G.degree()), key=lambda x: dict(G.degree())[x])

    # 保留 root 所在连通分量
    undir_G = G.to_undirected() if G.is_directed() else G
    if root not in undir_G:
        raise ValueError(f"Root {root} not in graph")
    connected_nodes = nx.node_connected_component(undir_G, root)
    G = G.subgraph(connected_nodes).copy()
    undir_G = G.to_undirected() if G.is_directed() else G
    bfs_dist = nx.shortest_path_length(undir_G, source=root)

    # 分层布局
    layers = {}
    for node, dist in bfs_dist.items():
        layers.setdefault(dist, []).append(node)

    pos = {}
    for layer, nodes in layers.items():
        nodes_sorted = sorted(nodes, key=str)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (i - len(nodes_sorted) / 2, -layer)

    node_colors = [bfs_dist[node] for node in G.nodes]

    # === 支持多边不重叠的绘图 ===
    plt.figure(figsize=figsize)

    # 画节点和标签
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='viridis',
                           node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black')

    # 统计每对 (u, v) 有多少条边
    from collections import defaultdict
    edge_groups = defaultdict(list)
    for u, v, key in G.edges(keys=True):
        edge_groups[(u, v)].append(key)

    # 画边：对每组平行边，分配不同弯曲度
    for (u, v), keys in edge_groups.items():
        num_edges = len(keys)
        if num_edges == 1:
            # 单边：直线
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v, keys[0])],
                edge_color='gray',
                arrows=G.is_directed(),
                arrowsize=10,
                width=1.0,
                alpha=0.9
            )
        else:
            # 多边：用弧线分开
            for i, key in enumerate(keys):
                # 弧度偏移：例如 [-0.2, -0.1, 0, 0.1, 0.2]
                rad = 0.15 * (i - (num_edges - 1) / 2)
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v, key)],  # 必须带 key！
                    edge_color='gray',
                    arrows=G.is_directed(),
                    arrowsize=10,
                    width=1.0,
                    alpha=0.9,
                    connectionstyle=f'arc3,rad={rad:.2f}'
                )
    plt.title(f"Hierarchical Layout from Root: {root}")
    # =========================
    # ✅ 强制居中：设置坐标轴范围
    # =========================
    # 获取所有节点的 x, y 坐标
    xs = [pos[node][0] for node in G.nodes]
    ys = [pos[node][1] for node in G.nodes]

    # 计算边界
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 扩展一点边界，避免节点贴边
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1

    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y

    # 设置坐标轴范围（强制居中）
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 关闭坐标轴
    plt.axis('off')
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
