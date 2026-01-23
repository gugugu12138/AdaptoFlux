import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from ATF.viz_adaptoflux import visualize_graph_hierarchy

def visualize_and_save_graph_hierarchy(
    model_path: str,
    output_image_path: str,
    root: str = "root",
    figsize=(12, 8),
    font_size: int = 6,          # 节点标签字体
    node_size: int = 600,
    title_font_size: int = 14    # 👈 新增：标题字体大小
):
    """
    加载图并保存层次化布局图像到指定路径。
    """
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

    # 手动布局
    pos = {}
    for layer, nodes in layers.items():
        nodes_sorted = sorted(nodes, key=str)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (i - len(nodes_sorted) / 2, -layer)

    node_colors = [bfs_dist[node] for node in G.nodes]

    # 绘图
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='viridis',
                           node_size=node_size, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='black')

    # 绘制边（兼容多重图）
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
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()  # 避免内存累积


def batch_visualize_graphs(
    base_dir: str = "experiments/ExternalBaselines/best_models",
    output_dir: str = "output_graphs",
    root: str = "root",
    figsize=(12, 8),
    font_size: int = 12,
    node_size: int = 1200,
    title_font_size: int = 14
):
    """
    批量处理 base_dir 下每个子文件夹中的 graph.json，并保存图像到 output_dir。
    
    目录结构要求：
        base_dir/
            Acceleration/
                combined_trainer_temp/final/graph.json
            Coulomb/
                combined_trainer_temp/final/graph.json
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 base_dir 下的一级子文件夹（如 Acceleration, Coulomb...）
    for task_name in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_path):
            continue  # 跳过非文件夹项

        graph_path = os.path.join(task_path, "combined_trainer_temp", "final", "graph.json")
        if not os.path.exists(graph_path):
            print(f"⚠️  跳过 {task_name}：未找到 graph.json")
            continue

        # ✅ 修改命名格式：best_{task}_collapse_prod.png
        output_image = os.path.join(output_dir, f"best_{task_name}_collapse_prod.png")
        print(f"正在处理: {task_name} -> {output_image}")

        try:
            visualize_and_save_graph_hierarchy(
                model_path=graph_path,
                output_image_path=output_image,
                root=root,
                figsize=figsize,
                font_size=font_size,
                node_size=node_size,
                title_font_size=title_font_size
            )
        except Exception as e:
            print(f"❌ 处理 {task_name} 时出错: {e}")

    print(f"\n✅ 所有图像已保存至: {os.path.abspath(output_dir)}")


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    batch_visualize_graphs(
        base_dir="experiments/ExternalBaselines/best_models",
        output_dir="experiments/ExternalBaselines/output_graphs",
        root="root",
        figsize=(14, 10),
        font_size=24,
        node_size=1600,
        title_font_size=28
    )