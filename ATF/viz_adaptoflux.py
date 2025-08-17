# 暂时放这后面可能做成类
import networkx as nx
import matplotlib.pyplot as plt

# 读取图
G = nx.read_gexf("/kaggle/working/AdaptoFlux/models/best/graph.gexf")

# 指定根节点（请根据你的数据修改 root）
# 如果不知道 root，可以选择一个度最高的节点，或手动指定
root = "root"  # 替换为你的实际 root 节点名
if root not in G:
    # 如果没有明确的 root，可以用一个中心节点（例如度最大的节点）
    root = max(dict(G.degree()), key=lambda x: dict(G.degree())[x])
    print(f"使用度最大的节点作为根: {root}")

# 使用 BFS 计算每个节点到 root 的距离（层数）
try:
    # 对于有向图，可能需要转为无向图来 BFS
    if G.is_directed():
        bfs_dist = nx.shortest_path_length(G.to_undirected(), source=root)
    else:
        bfs_dist = nx.shortest_path_length(G, source=root)
except nx.NetworkXNoPath:
    print("图不连通，只考虑包含 root 的连通分量")
    # 只保留 root 所在的连通分量
    if G.is_directed():
        connected_nodes = nx.node_connected_component(G.to_undirected(), root)
    else:
        connected_nodes = nx.node_connected_component(G, root)
    G = G.subgraph(connected_nodes)
    bfs_dist = nx.shortest_path_length(G.to_undirected() if G.is_directed() else G, source=root)

# 按距离分层
layers = {}
for node, dist in bfs_dist.items():
    layers.setdefault(dist, []).append(node)

# 手动设置布局：x 坐标在每层内均匀分布，y 坐标为层数（距离）
pos = {}
for layer, nodes in layers.items():
    pos.update({node: (i, -layer) for i, node in enumerate(nodes)})  # y = -layer 保证 root 在顶部

# 设置节点颜色（可选：根据层设置不同颜色）
node_colors = [bfs_dist[node] for node in G.nodes]

# 绘图
plt.figure(figsize=(12, 8))
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