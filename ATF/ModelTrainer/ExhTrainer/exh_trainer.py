import numpy as np
import networkx as nx
from ..model_trainer import ModelTrainer

class ExhaustiveSearchEngine(ModelTrainer):
    def __init__(self, model_trainer):
        super().__init__(model_trainer.adaptoflux)
        self.model_trainer = model_trainer

    def train(self, num_layers=2, warn_threshold=1000):
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
        n_prev = self.adaptoflux.get_input_dimension()
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
            # 计算当前层函数选择组合数
            function_choice_count = F_size ** n_prev

            # 假设平均每个函数输出维度为 avg_dim_out
            # 或者根据具体函数动态计算（推荐）
            avg_dim_out = 1  # 修改为实际值或动态计算

            # 当前层总的分支扩展因子
            branch_factor = function_choice_count * (avg_dim_out ** n_prev)

            total_combinations *= branch_factor
            print(f"第 {l + 1} 层组合数：{function_choice_count} (|F|^{n_prev}), 分支扩展因子：{branch_factor}")

            # 更新下一层输入数据量
            n_current = n_prev * avg_dim_out
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