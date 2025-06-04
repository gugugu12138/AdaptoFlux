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
        生成一个基于图结构的初始模型，用于后续训练、评估或演化。

        该方法从 root 节点开始，逐步插入随机操作层（由 AdaptoFlux 提供），最终连接到 collapse 节点，
        构建出一个完整的图结构模型。图中每条边表示数据流动路径，并携带原始特征索引信息。

        Parameters:
        -----------
        max_layers : int, optional (default=3)
            模型中要添加的操作层数量。控制模型复杂度。

        Returns:
        --------
        nx.MultiDiGraph
            构造完成的图结构模型，节点表示操作或输入输出，边表示数据流路径。
            图中的每个节点和边都包含必要的元数据，可用于推理和训练。

        Notes:
        ------
        - 初始图仅包含 root 和 collapse 两个节点，以及所有原始特征输入边。
        - 使用 graph_processor.append_nx_layer(result) 方法逐层扩展模型结构。
        - 每次调用可能生成不同的图结构，因为操作是随机选择的。

        Example:
        --------
        >>> model_generator = ModelGenerator(adaptoflux_instance)
        >>> graph = model_generator.generate_single_model(max_layers=2)
        >>> nx.draw(graph, with_labels=True)
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
        生成指定数量的初始模型，并对每个模型进行准确率评估。

        该方法通过调用 `self.generate_single_model(max_layers)` 随机生成多个图结构模型。
        每个模型都会使用当前数据集进行推理预测，并计算其在标签上的分类准确率。
        最终返回一个包含所有模型信息（图结构、准确率、层数）的候选模型列表。

        Parameters:
        -----------
        num_models : int, optional (default=5)
            要生成的初始模型数量。控制候选模型池的大小。
            
        max_layers : int, optional (default=3)
            每个模型的最大层数。用于控制模型复杂度，传递给 `generate_single_model()`。

        Returns:
        --------
        list of dict
            返回一个候选模型列表，每个元素是一个字典，包含以下键值对：
            
            - 'graph' : networkx.MultiDiGraph
                表示模型结构的有向多重图对象。
                
            - 'accuracy' : float
                该模型在当前数据集上的分类准确率（预测值与真实标签匹配的比例）。
                
            - 'layer' : int
                该模型的层数（即 max_layers 的值，表示模型复杂度）。

        See Also:
        ---------
        generate_single_model : 用于生成单个图结构模型的方法。
        _evaluate_model_accuracy : 用于评估模型准确率的方法。

        Examples:
        ---------
        >>> model_generator = ModelGenerator(adaptoflux_instance)
        >>> candidates = model_generator.generate_initial_models(num_models=3, max_layers=2)
        >>> len(candidates)
        3
        >>> best_model = max(candidates, key=lambda x: x['accuracy'])
        >>> best_accuracy = best_model['accuracy']
        >>> print(f"最佳模型准确率为: {best_accuracy:.4f}")
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
        使用当前数据对指定图结构模型进行推理，并计算其分类准确率。

        该方法调用 `self.adaptoflux.infer_with_graph()` 对传入的图结构模型执行前向推理，
        得到预测结果后与真实标签进行比较，返回模型在当前数据集上的准确率。

        Parameters:
        -----------
        graph : networkx.MultiDiGraph
            表示模型结构的有向多重图对象。图中每个节点应包含可调用的操作函数，
            边表示数据流动路径。

        Returns:
        --------
        float
            模型在当前数据集上的分类准确率，即预测值与真实标签匹配的比例。
            返回值范围为 [0.0, 1.0]。

        Raises:
        -------
        ValueError
            如果推理结果与标签的形状不一致，或推理失败，可能抛出异常。
        
        Notes:
        ------
        - 该方法假设 `self.adaptoflux.values` 是输入特征数据（ndarray 或 DataFrame）。
        - 该方法假设 `self.adaptoflux.labels` 是真实标签数据（一维数组）。
        - 准确率是通过 `np.mean(predictions == labels)` 计算得到的。

        Example:
        --------
        >>> model_generator = ModelGenerator(adaptoflux_instance)
        >>> graph = model_generator.generate_single_model()
        >>> accuracy = model_generator._evaluate_model_accuracy(graph)
        >>> print(f"模型准确率为: {accuracy:.4f}")
        """
        predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values, graph=graph)
        return float(np.mean(predictions == self.adaptoflux.labels))

    def evolve_subgraph(self, subgraph):
        """
        根据给定子图生成一个替代结构（占位函数）
        """
        return nx.MultiDiGraph()  # 示例返回空图

    def extract_high_performing_subgraph(self, graph):
        """
        提取性能优异的子图（占位函数）
        """
        return None

    def add_subgraph_as_method(self, subgraph, method_name):
        """
        将高性能子图封装为新方法加入方法池（占位函数）
        """
        def wrapped_function(*inputs):
            return np.zeros_like(inputs[0])

        input_count = len([n for n in subgraph.nodes if subgraph.in_degree(n) == 0])
        output_count = len([n for n in subgraph.nodes if subgraph.out_degree(n) == 0])

        self.adaptoflux.add_method(method_name, wrapped_function, input_count, output_count)
        print(f"已添加新方法 {method_name}，对应地址为{wrapped_function}")