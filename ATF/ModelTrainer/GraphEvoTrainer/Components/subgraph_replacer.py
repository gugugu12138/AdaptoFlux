import networkx as nx
from typing import Any

class SubgraphReplacer:
    """
    将子图替换为单个新节点。
    """
    def replace_with_node(
        self,
        graph_processor: Any,
        old_subgraph: nx.DiGraph,
        new_method_name: str
    ) -> str:
        """
        替换子图为一个新节点，并连接外部边。
        返回新节点 ID。
        """
        graph = graph_processor.graph

        # 找外部前驱和后继
        predecessors = set()
        successors = set()
        for node in old_subgraph.nodes():
            for pred in graph.predecessors(node):
                if pred not in old_subgraph.nodes():
                    predecessors.add(pred)
            for succ in graph.successors(node):
                if succ not in old_subgraph.nodes():
                    successors.add(succ)

        # 创建新节点
        new_node_id = graph_processor._generate_unique_node_id("compressed")
        graph.add_node(new_node_id, method_name=new_method_name)

        # 连接边
        for pred in predecessors:
            graph.add_edge(pred, new_node_id)
        for succ in successors:
            graph.add_edge(new_node_id, succ)

        # 删除原子图
        for node in old_subgraph.nodes():
            graph.remove_node(node)

        return new_node_id