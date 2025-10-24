import numpy as np
from typing import Dict, Tuple, Any

class SubgraphIOExtractor:
    """
    从完整图执行结果中提取子图的输入和输出。
    假设 AdaptoFlux.infer_with_graph 支持 return_all_nodes=True。
    """
    def extract(
        self,
        adaptoflux_instance: Any,
        subgraph: 'nx.MultiDiGraph',
        input_data: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        返回: (sub_inputs, sub_outputs)
        - sub_inputs: {入口节点名: 输入值}
        - sub_outputs: {出口节点名: 输出值}
        """
        # 执行主图，获取所有节点输出
        full_outputs = adaptoflux_instance.infer_with_graph(
            values=input_data,
            return_all_nodes=True  # 需确保 AdaptoFlux 支持此参数
        )

        graph = adaptoflux_instance.graph_processor.graph

        # 找入口节点（有来自子图外部的前驱）
        sub_inputs = {}
        for node in subgraph.nodes():
            external_preds = [
                u for u in graph.predecessors(node)
                if u not in subgraph.nodes()
            ]
            if external_preds:
                # 简化：取第一个前驱的输出作为输入（实际可能需拼接）
                sub_inputs[node] = full_outputs[external_preds[0]]
            # 注意：若无外部输入（如孤立子图），此处未处理，可加 fallback

        # 找出口节点（后继在子图外部）
        sub_outputs = {}
        for node in subgraph.nodes():
            external_succs = [
                v for v in graph.successors(node)
                if v not in subgraph.nodes()
            ]
            if external_succs:
                sub_outputs[node] = full_outputs[node]

        if not sub_outputs:
            # fallback: 取子图中最后一个节点
            last_node = list(subgraph.nodes())[-1]
            sub_outputs[last_node] = full_outputs[last_node]

        return sub_inputs, sub_outputs