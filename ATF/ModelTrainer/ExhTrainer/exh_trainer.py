import numpy as np
import networkx as nx
from ..model_trainer import ModelTrainer
from itertools import permutations, product
from collections import defaultdict
from .ModelTreeNode import ModelTreeNode
class ExhaustiveSearchEngine(ModelTrainer):
    def __init__(self, adaptoflux_instance):
        super().__init__(adaptoflux_instance)

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

        # 6. 生成所有模型组合
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
    
    def _build_layer_info(self, layer_idx, structure, combo):
        """
        根据当前层的结构和函数组合，构造结构化信息
        
        :param layer_idx: 层索引
        :param structure: 输入结构（如 ['numerical', 'categorical']）
        :param combo: 函数组合 [(name, info), ...]
        :return: dict 包含 index_map, valid_groups, unmatched
        """
        index_map = {}
        valid_groups = defaultdict(list) 

        for idx, (_, func_name) in enumerate(combo):
            input_count = self.adaptoflux.methods[func_name]["input_count"]
            if input_count <= 0:
                raise ValueError(f"方法 '{func_name}' 的 input_count 必须大于 0")

            groups = [list(range(idx, idx + input_count))]  # 示例简单分组
            for group in groups:
                if len(group) == input_count:
                    valid_groups[func_name].append(group)
                    for i in group:
                        index_map[i] = {"method": func_name, "group": tuple(group)}

        return {
            "index_map": index_map,
            "valid_groups": dict(valid_groups),
            "unmatched": []
        }

    def generate_valid_layer_combinations(self, input_function_pools):
        """
        根据每个输入点的函数池，生成当前层所有合法的函数组合。
        支持多输入函数，自动处理函数对输入数量的需求。
        
        :param input_function_pools: list[list[tuple(method_name, method_info)]]
        :return: list of combinations，每个组合是 [(group_indices, func_name), ...]
        """
        n_inputs = len(input_function_pools)
        all_possible_groups = []

        # Step 1: 枚举所有合法分组（基于函数所需的 input_count）
        def dfs(used_indices, current_groups, start=0):
            """
            深度优先搜索函数，用于枚举当前层所有合法的输入分组方式。
            
            :param used_indices: list，已使用的输入索引列表，例如 [0, 1]
            :param current_groups: list of tuples，当前尝试的分组方式，例如 [(0,1), (2,)]
            :param start: int，尝试从哪个输入索引开始分组
            """

            # 如果所有输入点都被使用了，说明找到了一个合法的完整分组方式
            if len(used_indices) == n_inputs:
                # 将当前分组方式加入结果列表（拷贝一份，防止后续修改）
                all_possible_groups.append(current_groups.copy())
                return

            # 从 start 开始尝试每一个输入点
            for i in range(start, n_inputs):
                # 如果当前输入点已经被使用了，跳过
                if i in used_indices:
                    continue

                # 获取当前输入点上所有函数支持的 input_count（即一个函数可以处理几个输入）
                # 去重处理，避免重复尝试相同的 input_count
                possible_input_counts = set(
                    method_info["input_count"]
                    for method_name, method_info in input_function_pools[i]
                )

                # 对所有可能的 input_count 按从小到大排序，尝试每种可能性
                for input_count in sorted(possible_input_counts):
                    # 计算这个函数处理的输入范围：从 i 开始，连续 input_count 个输入点
                    end = i + input_count

                    # 如果超出了输入点范围，跳过
                    if end > n_inputs:
                        continue

                    # 构造当前函数要处理的输入组（如 i=0, input_count=2 → group = [0, 1]）
                    group = list(range(i, end))

                    # 如果这个组中已经有输入点被使用了，跳过
                    if any(x in used_indices for x in group):
                        continue

                    # 将当前组加入 current_groups（尝试这个分组）
                    current_groups.append(tuple(group))

                    # 递归调用：继续尝试从 end 开始分组
                    dfs(used_indices + group, current_groups, end)

                    # 回溯：尝试完当前分组后，将当前组从 current_groups 中弹出
                    current_groups.pop()

        dfs([], [])

        # Step 2: 对每种分组方式，枚举每组可选的函数
        valid_combinations = []

        for group_list in all_possible_groups:
            group_function_options = []

            for group in group_list:
                # 取第一个输入点的函数池作为候选（假设同一组内函数池一致或兼容）
                candidate_funcs = input_function_pools[group[0]]

                # 过滤出 input_count == len(group) 的函数
                valid_funcs = [
                    (name, info) for name, info in candidate_funcs
                    if info.get("input_count", 1) == len(group)
                ]

                if not valid_funcs:
                    valid_funcs = [("__empty__", {
                        "input_count": len(group),
                        "output_count": 1,
                        "output_types": ["None"]
                    })]

                group_function_options.append([name for name, _ in valid_funcs])

            # 枚举该分组下每个组的函数选择（笛卡尔积）
            for func_choices in product(*group_function_options):
                combination = list(zip(group_list, func_choices))
                valid_combinations.append(combination)

        return valid_combinations

    def _calculate_total_combinations(self, num_layers):
        """
        动态计算每一层的组合数，并构建树状结构来记录所有模型组合
        
        :param num_layers: 层数
        :return: total_combinations 总组合数, root 树根节点
        """

        root = ModelTreeNode(layer_idx=0, structure=self.adaptoflux.feature_types)
        current_nodes = [root]

        total_combinations = 1
        function_pool_by_input_type = self.adaptoflux.build_function_pool_by_input_type()

        for layer_idx in range(num_layers):
            print(f"\n--- 第 {layer_idx + 1} 层计算开始 ---")
            next_layer_nodes = []
            current_layer_combinations = 0
            next_layer_structures = []
            
            for node in current_nodes:
                # 为每个节点在input_function_pools配置对应的函数池
                structure = node.structure
                input_function_pools = [
                    function_pool_by_input_type.get(input_type, [])
                    for input_type in structure
                ]
            
                input_function_pools = [
                    pool if len(pool) > 0 else [("__empty__", {
                        "input_count": 1,
                        "output_count": 1,
                        "input_types": ["None"],
                        "output_types": ["None"],
                        "group": "none",
                        "weight": 0.0,
                        "vectorized": True,
                        "function": lambda x: None
                    })]
                    for pool in input_function_pools
                ]

                # 生成该结构下的所有函数组合（笛卡尔积）
                all_function_combinations = self.generate_valid_layer_combinations(input_function_pools)

                current_layer_combinations += len(all_function_combinations)

                for combo in all_function_combinations:
                    # 构造下一层结构
                    input_types_for_next_layer = []
                    for group_indices, method_name in combo:

                        method_info = self.adaptoflux.methods.get(method_name, None)
                        if method_info is None:
                            raise ValueError(f"未找到方法 '{method_name}' 的定义")
                        output_types = method_info.get("output_types", ["None"])
                        input_types_for_next_layer.extend(output_types)

                    next_layer_structures.append(input_types_for_next_layer)

                    # 构建结构化信息
                    node_info = self._build_layer_info(layer_idx + 1, structure, combo)

                    # 创建新节点
                    child_node = ModelTreeNode(
                        layer_idx=layer_idx + 1,
                        structure=input_types_for_next_layer,
                        function_combo=combo,
                        parent=node
                    )
                    child_node.node_info = node_info
                    node.children.append(child_node)
                    next_layer_nodes.append(child_node)

            # 更新状态
            total_combinations = current_layer_combinations
            current_nodes = next_layer_nodes

            print(f"第 {layer_idx + 1} 层组合数：{current_layer_combinations}")
            print(f"下一层输入结构（示例）：{next_layer_structures} 共{len(next_layer_structures)} 种")

        return total_combinations, root
    
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