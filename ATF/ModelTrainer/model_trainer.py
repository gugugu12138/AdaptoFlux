import random
import numpy as np
from collections import Counter
from ..GraphManager.graph_processor import GraphProcessor
import networkx as nx
from ..ModelGenerator.model_generator import ModelGenerator
from abc import ABC, abstractmethod

class ModelTrainer:
    def __init__(self, adaptoflux_instance):
        """
        初始化训练器，绑定 AdaptoFlux 实例
        :param adaptoflux_instance: 已初始化的 AdaptoFlux 对象
        """
        self.adaptoflux = adaptoflux_instance
        self.model_generator = ModelGenerator(adaptoflux_instance)

    @abstractmethod
    def train(self, **kwargs):
        """所有子类必须实现 train 方法"""
        pass

    # -----------------------------
    # 模型生成相关函数
    # -----------------------------
    def generate_initial_models(self, num_models=5, max_layers=3):
        """
        生成多个初始模型，并评估其准确率
        """
        
        print("使用 ModelGenerator 生成初始模型...")
        initial_models = self.model_generator.generate_initial_models(num_models, max_layers)
        print(f"共生成 {len(initial_models)} 个初始模型")

        return initial_models 

    def select_best_model(self, models):
        best_model = max(models, key=lambda x: x["accuracy"])
        self.adaptoflux.graph = best_model["graph"]
        self.adaptoflux.layer = best_model["layer"]
        print(f"选择最佳初始模型，准确率：{best_model['accuracy']:.4f}")
        return best_model

    # -----------------------------
    # 节点优化相关函数
    # -----------------------------
    def optimize_nodes(self, rounds=10, target_accuracy=None):
        best_accuracy = self._get_current_accuracy()
        print(f"\n开始节点优化，当前准确率：{best_accuracy:.4f}")

        for round_num in range(rounds):
            print(f"\n轮次 {round_num + 1}/{rounds}")
            for layer_idx in range(1, self.adaptoflux.layer + 1):
                layer_nodes = [node for node, data in self.adaptoflux.graph.nodes(data=True) if data.get('layer', 0) == layer_idx]
                for node in layer_nodes:
                    best_accuracy = self._optimize_node(node, best_accuracy)
            if target_accuracy is not None and best_accuracy >= target_accuracy:
                print(f"提前完成训练，准确率达到目标值 {target_accuracy:.4f}")
                break
        return best_accuracy

    def _optimize_node(self, node, current_best_accuracy):
        method_name = self.adaptoflux.graph.nodes[node].get("method_name")
        input_count = self.adaptoflux.graph.nodes[node].get("input_count", 1)
        output_count = self.adaptoflux.graph.nodes[node].get("output_count", 1)

        compatible_methods = [
            name for name, method in self.adaptoflux.methods.items()
            if method["input_count"] == input_count and method["output_count"] == output_count
        ]

        if len(compatible_methods) <= 1:
            return current_best_accuracy

        best_method = method_name
        best_acc = current_best_accuracy

        for candidate_method in compatible_methods:
            if candidate_method == method_name:
                continue
            self.adaptoflux.graph.nodes[node]["method_name"] = candidate_method
            self.adaptoflux.graph.nodes[node]["function"] = self.adaptoflux.methods[candidate_method]["function"]

            predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values)
            acc = np.mean(predictions == self.adaptoflux.labels)

            if acc > best_acc:
                best_acc = acc
                best_method = candidate_method

        if best_method != method_name:
            self.adaptoflux.graph.nodes[node]["method_name"] = best_method
            self.adaptoflux.graph.nodes[node]["function"] = self.adaptoflux.methods[best_method]["function"]
            print(f"节点 {node} 方法从 {method_name} 改为 {best_method}，准确率提升到 {best_acc:.4f}")
        return best_acc

    # -----------------------------
    # 子图替换 / 模型进化
    # -----------------------------
    def evolve_subgraphs(self, num_candidates=3):
        print("\n开始模型进化...")
        best_accuracy = self._get_current_accuracy()

        for _ in range(num_candidates):
            start_layer = random.randint(1, self.adaptoflux.layer - 1)
            end_layer = min(start_layer + random.randint(1, 2), self.adaptoflux.layer)
            subgraph = nx.subgraph_view(
                self.adaptoflux.graph,
                filter_node=lambda n: start_layer <= self.adaptoflux.graph.nodes[n].get('layer', 0) <= end_layer
            )

            alternative_subgraph = self._generate_alternative_subgraph(subgraph)
            original_predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values)
            original_accuracy = np.mean(original_predictions == self.adaptoflux.labels)

            temp_graph = self.adaptoflux.graph.copy()
            # 替换子图逻辑需具体实现
            alternative_predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values)
            alternative_accuracy = np.mean(alternative_predictions == self.adaptoflux.labels)

            if alternative_accuracy > original_accuracy:
                self.adaptoflux.graph = temp_graph
                best_accuracy = alternative_accuracy
                print(f"子图替换成功，准确率提升到 {best_accuracy:.4f}")
        return best_accuracy

    def _generate_alternative_subgraph(self, subgraph):
        # TODO: 根据输入输出维度生成替代子图
        return nx.MultiDiGraph()  # 示例返回空图

    # -----------------------------
    # 方法池更新
    # -----------------------------
    def update_method_pool(self, num_modules=3):
        print("\n更新方法池...")
        for module_id in range(num_modules):
            subgraph = self._extract_high_performing_subgraph()
            if subgraph:
                new_method_name = f"auto_module_{module_id}"
                self._add_subgraph_as_method(subgraph, new_method_name)

    def _extract_high_performing_subgraph(self):
        # TODO: 提取表现优异的子图逻辑
        return None

    def _add_subgraph_as_method(self, subgraph, method_name):
        def wrapped_function(*inputs):
            # TODO: 实现子图推理逻辑
            return np.zeros_like(inputs[0])
        # 获取输入输出维度
        input_count = len([n for n in subgraph.nodes if subgraph.in_degree(n) == 0])
        output_count = len([n for n in subgraph.nodes if subgraph.out_degree(n) == 0])

        self.adaptoflux.add_method(method_name, wrapped_function, input_count, output_count)
        print(f"已添加新方法 {method_name}")

    # -----------------------------
    # 辅助函数
    # -----------------------------
    def _get_current_accuracy(self):
        predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values)
        return np.mean(predictions == self.adaptoflux.labels)
    
    def exhaustive_search_train(self, num_layers=2, warn_threshold=1000):
        """
        使用穷举法在指定层数下遍历所有可能的模型组合，寻找最优模型。
        每层的节点数由前一层输出决定（假设每个函数输出1个数据）。
        
        :param num_layers: 总层数（深度）
        :param warn_threshold: 当组合总数超过该值时发出警告
        :return: 最终准确率
        """

        print("\n=== 开始穷举搜索训练（动态节点数版本） ===")

        # 1. 获取方法池
        function_pool = list(self.adaptoflux.methods.items())
        F_size = len(function_pool)
        print(f"方法池大小：{F_size}")

        # 2. 初始化输入数据量
        n_prev = 1
        output_sizes = [n_prev]

        # 3. 计算组合数和输出维度
        total_combinations, output_sizes = self._calculate_total_combinations(
            num_layers, F_size, n_prev, output_sizes
        )

        # 4. 发出警告（如果组合数太大）
        if total_combinations > warn_threshold:
            print(f"⚠️ 警告：总组合数为 {total_combinations}，超过阈值 {warn_threshold}，训练时间可能非常长！")
        else:
            print(f"总组合数为 {total_combinations}，将在合理范围内进行穷举。")

        # 5. 生成每一层的所有函数选择
        all_function_choices = self._generate_layer_function_choices(
            num_layers, output_sizes, function_pool
        )

        # 6. 生成所有模型组合（各层函数选择的笛卡尔积）
        all_model_combinations = self._generate_all_model_combinations(all_function_choices)

        print(f"已生成所有模型组合，共 {len(all_model_combinations)} 种")

        # 7. 初始化最佳模型记录器
        best_accuracy = 0.0
        best_model_graph = None

        # 8. 遍历所有组合并评估模型
        for idx, model_combo in enumerate(all_model_combinations):
            print(f"\r处理第 {idx + 1}/{len(all_model_combinations)} 个模型...", end="")

            graph = self._build_graph_from_combo(model_combo)
            accuracy = self._evaluate_model_accuracy(graph)

            best_accuracy, best_model_graph = self._update_best_model(
                accuracy, best_accuracy, best_model_graph, graph
            )

        # 9. 完成训练并更新模型
        self.adaptoflux.graph = best_model_graph
        print("\n✅ 穷举搜索完成，已使用最佳模型更新 AdaptoFlux 实例")
        print(f"最终准确率：{best_accuracy:.4f}")
        return best_accuracy

    def _calculate_total_combinations(self, num_layers, F_size, n_prev, output_sizes):
        total_combinations = 1
        for l in range(num_layers):
            combinations_this_layer = F_size ** n_prev
            total_combinations *= combinations_this_layer
            print(f"第 {l + 1} 层组合数：{combinations_this_layer} (|F|^{n_prev})")

            n_current = n_prev * 1  # 假设每个函数输出1个数据
            output_sizes.append(n_current)
            n_prev = n_current
        return total_combinations, output_sizes

    def _generate_layer_function_choices(self, num_layers, output_sizes, function_pool):
        from itertools import product
        all_function_choices = []
        for l in range(num_layers):
            prev_data_count = output_sizes[l]
            layer_function_choices = list(product(function_pool, repeat=prev_data_count))
            all_function_choices.append(layer_function_choices)
        return all_function_choices

    def _generate_all_model_combinations(self, all_function_choices):
        from itertools import product as full_product
        return list(full_product(*all_function_choices))

    def _build_graph_from_combo(self, model_combo):
        graph = nx.MultiDiGraph()
        graph.add_node("input", layer=0, method_name="input", function=lambda x: x)
        current_nodes = ["input"]

        for layer_idx, layer_functions in enumerate(model_combo):
            layer_id = layer_idx + 1
            new_nodes = []
            for i, (func_name, func_obj) in enumerate(layer_functions):
                node_name = f"L{layer_id}_N{i}"
                func = func_obj["function"]
                input_count = func_obj.get("input_count", 1)
                output_count = func_obj.get("output_count", 1)

                graph.add_node(node_name,
                            layer=layer_id,
                            method_name=func_name,
                            function=func,
                            input_count=input_count,
                            output_count=output_count)

                for prev_node in current_nodes:
                    graph.add_edge(prev_node, node_name)
                new_nodes.append(node_name)

            current_nodes = new_nodes
        return graph

    def _evaluate_model_accuracy(self, graph):
        self.adaptoflux.graph = graph
        predictions = self.adaptoflux.infer_with_graph(self.adaptoflux.values)
        return np.mean(predictions == self.adaptoflux.labels)

    def _update_best_model(self, accuracy, best_accuracy, best_model_graph, graph):
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_graph = graph.copy()
            print(f"\n🎉 发现新最佳模型，准确率：{best_accuracy:.4f}")
        return best_accuracy, best_model_graph