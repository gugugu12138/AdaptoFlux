import networkx as nx
import random
import matplotlib.pyplot as plt


class LayerGraph:
    def __init__(self, methods, feature_dim=10):
        self.graph = nx.MultiDiGraph()
        self.layer = 0
        self.methods = methods
        self.graph.add_node("root", method_name="input_root")
        self.graph.add_node("collapse")
        for feature_index in range(feature_dim):
            self.graph.add_edge("root", "collapse", output_index=feature_index, data_coord=feature_index)

    def process_random_method(self, shuffle_indices=True):
        if not self.methods:
            raise ValueError("方法字典为空，无法处理！")

        collapse_edges = list(self.graph.in_edges("collapse", data=True))
        num_elements = len(collapse_edges)
        method_list = [random.choice(list(self.methods.keys())) for _ in range(num_elements)]

        index_map = {}
        valid_groups = {}
        unmatched = []

        multi_input_positions = {}
        for idx, method_name in enumerate(method_list):
            input_count = self.methods[method_name]["input_count"]
            if input_count <= 0:
                raise ValueError(f"方法 '{method_name}' 的 input_count 必须大于 0，当前值为 {input_count}")
            multi_input_positions.setdefault(method_name, []).append(idx)

        for method_name, indices in multi_input_positions.items():
            input_count = self.methods[method_name]["input_count"]
            groups, unmatch = self.group_indices_by_input_count(indices, input_count, shuffle_indices)
            valid_groups[method_name] = groups
            for group in groups:
                for idx in group:
                    index_map[idx] = {"method": method_name, "group": tuple(group)}
            unmatched.extend([idx] for idx in unmatch)

        # 处理未匹配项
        for sublist in unmatched:
            for idx in sublist:
                index_map[idx] = {"method": "unmatched", "group": tuple(sublist)}

        return {
            "index_map": index_map,
            "valid_groups": valid_groups,
            "unmatched": unmatched
        }

    def group_indices_by_input_count(self, indices, input_count, shuffle=True):
        if shuffle:
            random.shuffle(indices)
        full_groups = [indices[i:i + input_count] for i in range(0, len(indices) - len(indices) % input_count, input_count)]
        remainder = indices[len(full_groups) * input_count:]
        return full_groups, remainder

    def append_nx_layer(self, result, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        将给定的结果转换为图中的一层新节点。
        
        参数:
            result: 一个字典，包含以下键：
                - 'valid_groups': 按照方法名分组的有效数据索引列表
                - 'unmatched': 无法匹配成完整输入的剩余索引
            discard_unmatched: 如何处理未匹配项。可选值：
                'to_discard' - 创建单独节点并连接回 collapse
                'ignore' - 忽略这些未匹配项
            discard_node_method_name: 如果 discard_unmatched == 'to_discard'，则用该名称作为丢弃节点的 method_name
        """

        # 增加层数计数器
        self.layer += 1
        
        # 记录当前新增边的 data_coord 编号，保证唯一性(模型总索引不会重复)
        new_index_edge = 0

        # 初始化每个方法使用次数的计数器（用于生成唯一的节点名）
        method_counts = {method_name: 0 for method_name in self.methods}
        method_counts["unmatched"] = 0  # 单独记录 unmatched 的数量

        # 获取所有指向 "collapse" 的入边（即上一层输出的边）
        collapse_edges = list(self.graph.in_edges("collapse", data=True))

        # 设置目标节点为 collapse（统一变量名方便后续操作）
        v = "collapse"

        # 遍历所有有效的方法及其对应的分组
        for method_name, groups in result['valid_groups'].items():
            if method_name == "unmatched":
                continue  # 跳过无效的 unmatched 分组，在后面单独处理

            for group in groups:
                # 构造新的目标节点名：如 "1_0_add"
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                # 添加新节点到图中，并设置 method_name 属性
                self.graph.add_node(new_target_node, method_name=method_name)

                # 遍历之前所有的 collapse 入边
                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        # 如果这条边的数据坐标属于当前分组，则删除原边，并将它连接到新节点
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, new_target_node, **data)

                # 新增从新节点到 collapse 的输出边，数量由方法定义决定（output_count）
                for local_output_index in range(self.methods[method_name]["output_count"]):
                    self.graph.add_edge(new_target_node, v, output_index = local_output_index, data_coord=new_index_edge)
                    new_index_edge += 1

                # 更新该方法使用的计数
                method_counts[method_name] += 1

        # 处理未匹配项（如果存在）
        unmatched_groups = result.get('unmatched', [])
        if unmatched_groups and discard_unmatched == 'to_discard':
            for group in unmatched_groups:
                # 构造丢弃节点名，如 "1_2_unmatched"
                node_name = f"{self.layer}_{method_counts['unmatched']}_unmatched"
                # 添加节点并指定 method_name 为传入的 discard_node_method_name
                self.graph.add_node(node_name, method_name=discard_node_method_name)

                # 同样地，把对应 data_coord 的边从 collapse 移动到这个丢弃节点
                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, node_name, **data)

                # 从丢弃节点再连一条边回到 collapse（默认输出一条）
                for local_output_index in range(1):
                    self.graph.add_edge(node_name, v, output_index = local_output_index, data_coord=new_index_edge)
                    new_index_edge += 1

                method_counts["unmatched"] += 1

        elif unmatched_groups and discard_unmatched != 'ignore':
            # 如果有未匹配项且不是 ignore 或 to_discard，则报错
            raise ValueError(f"未知的 discard_unmatched 值：{discard_unmatched}。支持的选项为 'ignore' 或 'to_discard'")

    def remove_last_nx_layer(self):
        collapse_node = "collapse"
        incoming_edges_to_collapse = list(self.graph.in_edges(collapse_node, data=True))
        nodes_to_remove = set(u for u, v, d in incoming_edges_to_collapse)
        if not nodes_to_remove:
            print("没有可删除的层。")
            return

        input_edges_map = {node: list(self.graph.in_edges(node, data=True)) for node in nodes_to_remove}

        for node in nodes_to_remove:
            for prev_node, _, edge_data in input_edges_map[node]:
                self.graph.add_edge(prev_node, collapse_node, **edge_data)

        for node in nodes_to_remove:
            if node != "root":
                self.graph.remove_node(node)
        self.layer -= 1

    def get_collapse_inputs(self):
        return set(u for u, v, d in self.graph.in_edges("collapse", data=True))

    def print_graph_summary(self):
        edges = self.graph.in_edges("collapse", data=True)
        print("Current collapse inputs:")
        for u, _, data in edges:
            print(f"  {u} -> collapse (data_coord={data.get('data_coord')})")

    def visualize_graph(self, show_method_labels=True):
        plt.figure(figsize=(12, 8))
        node_colors = []
        node_labels = {}
        colors = {
            "root": "#FF9999",
            "input": "#66B2FF",
            "layer0": "#5CD6D6",
            "layer1": "#99FF99",
            "layer2": "#FFD700",
            "layer3": "#FFA07A",
            "default": "#DDDDDD",
            "collapse": "#FF4B4B"
        }
        for node in self.graph.nodes:
            if node == "collapse":
                color = colors["collapse"]
            elif node.startswith("input_"):
                color = colors["input"]
            elif node.startswith("0_"):
                color = colors["layer0"]
            elif node.startswith("1_"):
                color = colors["layer1"]
            elif node.startswith("2_"):
                color = colors["layer2"]
            elif node == "root":
                color = colors["root"]
            else:
                color = colors["default"]
            node_colors.append(color)
            if show_method_labels:
                method_name = self.graph.nodes[node].get('method_name', '')
                node_labels[node] = f"{node}\n({method_name})" if method_name else node

        pos = nx.spring_layout(self.graph, seed=42, k=0.5)
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels,
                node_color=node_colors, font_size=10, node_size=800,
                edge_color="gray", width=1.0, alpha=0.8, linewidths=1.0)
        plt.title(f"Graph Structure (Current Layer: {self.layer})")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_rank_for_node(node):
        if node == "root":
            return 0
        elif '_' in node:
            try:
                layer_num = int(node.split('_')[0])
                return layer_num + 1
            except ValueError:
                return 999
        elif node == "collapse":
            return 1000
        else:
            return 999


# ========================
# 测试代码开始
# ========================

def run_test_sequence(methods):
    lg = LayerGraph(methods)
    graph_states = []
    titles = []

    save_state(lg, graph_states, titles, "Initial State")

    actions = [
        ("add", {}),
        ("add", {}),
        ("remove",),
        ("remove",),
    ]

    for action in actions:
        if action[0] == "add":
            result = lg.process_random_method()
            lg.append_nx_layer(result)
            save_state(lg, graph_states, titles, f"After Layer {lg.layer}")
        elif action[0] == "remove":
            lg.remove_last_nx_layer()
            save_state(lg, graph_states, titles, f"After Remove (Layer {lg.layer})")

    visualize_all_states(graph_states, titles)


def save_state(lg, graph_states, titles, title_prefix):
    graph_states.append(lg.graph.copy())
    titles.append(f"{title_prefix}\n(Layer: {lg.layer})")


def visualize_all_states(states, titles, show_method_labels=True):
    num_plots = len(states)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()

    def get_rank_for_node(node):
        if node == "root":
            return 0
        elif '_' in node:
            try:
                layer_num = int(node.split('_')[0])
                return layer_num + 1
            except ValueError:
                return 999
        elif node == "collapse":
            return 1000
        else:
            return 999

    for idx, (graph, title) in enumerate(zip(states, titles)):
        ax = axes[idx]
        node_colors = []
        node_labels = {}

        for node in graph.nodes:
            if node == "collapse":
                color = "#FF4B4B"
            elif node.startswith("input_"):
                color = "#66B2FF"
            elif node.startswith("0_"):
                color = "#5CD6D6"
            elif node.startswith("1_"):
                color = "#99FF99"
            elif node.startswith("2_"):
                color = "#FFD700"
            elif node == "root":
                color = "#FF9999"
            else:
                color = "#DDDDDD"
            node_colors.append(color)
            if show_method_labels:
                method_name = graph.nodes[node].get('method_name', '')
                node_labels[node] = f"{node}\n({method_name})" if method_name else node

        A = nx.nx_agraph.to_agraph(graph)
        A.graph_attr.update(rankdir='TB')

        ranks = {}
        for node in A.nodes():
            rank = get_rank_for_node(node)
            ranks.setdefault(rank, []).append(node)

        for rank_level in sorted(ranks.keys()):
            A.add_subgraph(ranks[rank_level], rank='same')

        sorted_ranks = sorted(k for k in ranks if 0 < k < 1000)
        if len(sorted_ranks) > 1:
            for i in range(1, len(sorted_ranks)):
                prev_nodes = ranks.get(sorted_ranks[i - 1], [])
                curr_nodes = ranks.get(sorted_ranks[i], [])
                if prev_nodes and curr_nodes:
                    A.add_edge(prev_nodes[0], curr_nodes[0], style="invis")

        A.layout(prog="dot")
        pos = {node: list(map(float, A.get_node(node).attr["pos"].split(','))) for node in A.nodes()}

        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=600)

        edge_keys_seen = {}
        for u, v, key in graph.edges(keys=True):
            if (u, v) not in edge_keys_seen:
                edge_keys_seen[(u, v)] = 0
            else:
                edge_keys_seen[(u, v)] += 1
            offset = edge_keys_seen[(u, v)] * 0.02
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            norm = (dx ** 2 + dy ** 2) ** 0.5
            off_x = -dy / norm * offset if norm != 0 else 0
            off_y = dx / norm * offset if norm != 0 else 0
            ax.annotate(
                "",
                xy=(pos[v][0] + off_x, pos[v][1] + off_y),
                xycoords='data',
                xytext=(pos[u][0] + off_x, pos[u][1] + off_y),
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="->",
                    color="gray",
                    shrinkA=5,
                    shrinkB=5,
                    lw=1.0,
                    connectionstyle=f"arc3,rad={0.2 * edge_keys_seen[(u, v)]}"
                ),
            )

        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    for idx in range(len(states), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    methods = {
        "add": {"input_count": 2, "output_count": 1},
        "avg": {"input_count": 3, "output_count": 1},
        "single": {"input_count": 1, "output_count": 1}
    }

    run_test_sequence(methods)