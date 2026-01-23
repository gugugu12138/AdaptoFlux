import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_graph_hierarchy(
    model_path: str,
    root: str = "root",
    figsize=(12, 8),
    font_size: int = 6,
    node_size: int = 600,
    title_font_size: int = 14
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载图
    if model_path.endswith('.gexf'):
        G = nx.read_gexf(model_path)
    elif model_path.endswith('.json'):
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.node_link_graph(data, edges="edges")
    else:
        raise ValueError("仅支持 .gexf 或 .json 格式的图文件")

    # ================================
    # 检查是否使用预定义 layer 还是回退到 BFS
    # ================================
    use_predefined_layer = False
    layers = defaultdict(list)

    # 检查是否有非 root/collapse 节点包含 'layer' 属性
    for node in G.nodes():
        if node not in ("root", "collapse") and G.nodes[node].get("layer") is not None:
            use_predefined_layer = True
            break

    if use_predefined_layer:
        print("✅ 使用节点预定义的 'layer' 属性进行分层")
        for node in G.nodes():
            node_data = G.nodes[node]
            if node == "root":
                layer = 0
            elif node == "collapse":
                continue  # 稍后处理
            elif "layer" in node_data:
                layer = node_data["layer"]
            else:
                # 如果个别节点缺失 layer，可设为最大层或警告
                print(f"⚠️ 节点 '{node}' 缺少 'layer' 属性，暂放入最后一层")
                layer = 999  # 临时放最后，后续会调整
            layers[layer].append(node)

        # 处理 collapse：放在已知最大层 + 1
        if "collapse" in G:
            known_layers = [l for l in layers.keys() if l != 999]
            max_known = max(known_layers) if known_layers else -1
            # 把 layer=999 的节点移到 max_known + 1（如果需要）
            if 999 in layers:
                layers[max_known + 1].extend(layers.pop(999))
            layers[max_known + 2].append("collapse")  # collapse 在 action 之后

    else:
        print("🔄 未检测到有效的 'layer' 属性，回退到 BFS 分层")
        # === 原 BFS 逻辑 ===
        if root not in G:
            root = max(dict(G.degree()), key=lambda x: dict(G.degree())[x])
            print(f"指定的 root 节点 '{root}' 不存在，使用度最大的节点作为根: {root}")

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

        for node, dist in bfs_dist.items():
            layers[dist].append(node)

    # 排序层号
    sorted_layers = sorted(layers.items())
    pos = {}
    for layer, nodes in sorted_layers:
        nodes_sorted = sorted(nodes, key=str)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (i - len(nodes_sorted) / 2, -layer)

    # 构建 node_colors（用于着色）
    node_colors = []
    for node in G.nodes():
        if use_predefined_layer:
            if node == "root":
                color_val = 0
            elif node == "collapse":
                color_val = max(layers.keys())
            else:
                color_val = G.nodes[node].get("layer", max(layers.keys()))
        else:
            # BFS 模式
            color_val = bfs_dist[node]  # 注意：此时 bfs_dist 已定义
        node_colors.append(color_val)

    # ========================
    # 绘图（保持不变）
    # ========================
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='viridis',
                           node_size=node_size, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='black')

    if G.is_multigraph():
        edge_groups = defaultdict(list)
        for u, v, key in G.edges(keys=True):
            edge_groups[(u, v)].append(key)
        for (u, v), keys in edge_groups.items():
            n = len(keys)
            for i, key in enumerate(keys):
                rad = 0.15 * (i - (n - 1) / 2) if n > 1 else 0.0
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v, key)],
                    edge_color='gray',
                    arrows=G.is_directed(),
                    arrowsize=10,
                    width=1.0,
                    alpha=0.9,
                    connectionstyle=f'arc3,rad={rad:.2f}' if n > 1 else 'arc3'
                )
    else:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=list(G.edges()),
            edge_color='gray',
            arrows=G.is_directed(),
            arrowsize=10,
            width=1.0,
            alpha=0.9
        )

    plt.title(f"Hierarchical Layout from Root: {root}", fontsize=title_font_size)
    plt.axis('off')
    plt.tight_layout()
    plt.show()