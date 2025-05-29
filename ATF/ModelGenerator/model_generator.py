import random
import networkx as nx
from collections import Counter
import numpy as np


class ModelGenerator:
    def __init__(self, adaptoflux_instance):
        """
        模型生成器，用于生成/进化模型结构
        """
        self.adaptoflux = adaptoflux_instance

    def generate_single_model(self, max_layers=3):
        """
        生成单个初始模型图结构
        """
        temp_graph = nx.MultiDiGraph()
        temp_graph.add_node("root")
        temp_graph.add_node("collapse")

        # 添加 root -> collapse 的初始边
        for feature_index in range(self.adaptoflux.values.shape[1]):
            temp_graph.add_edge(
                "root",
                "collapse",
                output_index=0,
                data_coord=feature_index
            )

        graph_processor = self.adaptoflux._create_graph_processor(temp_graph)
        current_layer = 0
        while current_layer < max_layers:
            result = self.adaptoflux.process_random_method()
            graph_processor.append_nx_layer(result)
            current_layer += 1

        return graph_processor.graph

    def generate_initial_models(self, num_models=5, max_layers=3):
        """
        生成多个初始模型并评估准确率
        """
        model_candidates = []
        for _ in range(num_models):
            graph = self.generate_single_model(max_layers)
            accuracy = self._evaluate_model_accuracy(graph)
            model_candidates.append({
                "graph": graph,
                "accuracy": accuracy,
                "layer": max_layers
            })
        return model_candidates

    def _evaluate_model_accuracy(self, graph):
        """
        使用当前数据对模型进行一次推理并计算准确率
        """
        predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values, graph=graph)
        return float(np.mean(predictions == self.adaptoflux.labels))

    def evolve_subgraph(self, subgraph):
        """
        根据给定子图生成一个替代结构（简化版）
        """
        return nx.MultiDiGraph()  # 示例返回空图

    def extract_high_performing_subgraph(self, graph):
        """
        提取性能优异的子图（占位函数）
        """
        return None

    def add_subgraph_as_method(self, subgraph, method_name):
        """
        将高性能子图封装为新方法加入方法池
        """
        def wrapped_function(*inputs):
            return np.zeros_like(inputs[0])

        input_count = len([n for n in subgraph.nodes if subgraph.in_degree(n) == 0])
        output_count = len([n for n in subgraph.nodes if subgraph.out_degree(n) == 0])

        self.adaptoflux.add_method(method_name, wrapped_function, input_count, output_count)
        print(f"已添加新方法 {method_name}")