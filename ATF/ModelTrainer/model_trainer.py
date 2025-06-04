import random
import numpy as np
from collections import Counter
from ..GraphManager.graph_processor import GraphProcessor
import networkx as nx
from ..ModelGenerator.model_generator import ModelGenerator

class ModelTrainer:
    def __init__(self, adaptoflux_instance):
        """
        初始化训练器，绑定 AdaptoFlux 实例
        :param adaptoflux_instance: 已初始化的 AdaptoFlux 对象
        """
        self.adaptoflux = adaptoflux_instance
        self.model_generator = ModelGenerator(adaptoflux_instance)

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

    # -----------------------------
    # 主训练流程
    # -----------------------------
    def train(self, 
              num_initial_models=5,
              max_layers=3,
              optimization_rounds=10,
              evolution_candidates=3,
              target_accuracy=None):
        """
        完整训练流程入口
        """
        print("=== 开始完整训练流程 ===")

        # 阶段一：生成初始模型
        initial_models = self.generate_initial_models(num_initial_models, max_layers)
        best_model = self.select_best_model(initial_models)

        # 阶段二：节点优化
        final_accuracy = self.optimize_nodes(rounds=optimization_rounds, target_accuracy=target_accuracy)

        # 阶段三：子图替换进化
        final_accuracy = self.evolve_subgraphs(evolution_candidates)

        # 阶段四：更新方法池
        self.update_method_pool()

        print(f"\n✅ 训练完成，最终准确率：{final_accuracy:.4f}")
        return final_accuracy