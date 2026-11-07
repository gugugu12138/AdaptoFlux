import numpy as np
import networkx as nx
from ..model_trainer import ModelTrainer
from itertools import permutations, product
from collections import defaultdict
from .ModelTreeNode import ModelTreeNode
from itertools import combinations
# 未完成 有严重bug 不建议使用
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
        主流程：生成当前层所有合法的函数组合。
        支持多输入函数。
        """
        # Step 1: 枚举所有合法的输入分组方式
        all_possible_groups = self.enumerate_input_groups(input_function_pools)

        # Step 2: 对每种分组方式，生成所有可能的函数组合（笛卡尔积）
        valid_combinations = []
        for group_list in all_possible_groups:
            combinations = self.generate_combinations_for_group(input_function_pools, group_list)
            valid_combinations.extend(combinations)

        return valid_combinations
    
    def enumerate_input_groups(self, input_function_pools, allow_non_increasing_order=True):
        """
        枚举当前层所有合法的输入分组方式（允许非连续、不重复、覆盖所有输入）
        
        核心思想：
        1. 将当前层的所有输入数据点看作一个集合。
        2. 我们需要将这些数据点"打包"成若干个小组 (group)。
        3. 每个小组将被送入一个函数进行处理。
        4. 本函数的目标是找出所有合法的"打包"方案。
        
        合法性要求：
        - 覆盖：每一个输入数据点都必须且只能被使用一次。
        - 函数匹配：每个小组的大小（即包含的数据点数量）必须等于该小组所选函数的 input_count。
        - 不重复：不允许同一个数据点被分配给多个函数。
        
        注意：此版本使用 permutations，因此 [0,2] 和 [2,0] 被视为不同的分组，
              这意味着函数对输入顺序是敏感的。
              
        :param input_function_pools: 列表，每个元素是一个列表，包含了可以处理对应位置输入数据的 (函数名, 函数信息) 元组。
                                    例如: [[('funcA', {...}), ('funcB', {...})], [('funcC', {...})]]
        :return: List[List[tuple]] - 所有合法的分组方案列表。每个方案是一个元组列表，每个元组代表一个输入组。
                 例如: [[(0,), (1, 2)], [(0, 1), (2,)]]
        """
        n_inputs = len(input_function_pools)
        all_possible_groups = [] # 存储所有找到的合法分组方案

        def dfs(used_indices, current_groups):
            """
            深度优先搜索 (DFS) 辅助函数，用于递归地生成所有可能的分组方案。
            
            :param used_indices: list, 当前已使用的输入索引列表。
            :param current_groups: list of tuples, 当前正在构建的分组方案。
            """
            # 终止条件：如果所有输入都被使用了，则找到了一个完整的合法方案
            if len(used_indices) == n_inputs:
                # 将当前方案的深拷贝加入结果列表
                all_possible_groups.append(current_groups.copy())
                return # 回溯

            # 获取当前还未被使用的输入索引
            available_indices = [i for i in range(n_inputs) if i not in used_indices]

            # 遍历每一个可用的输入索引
            # 注意：这里遍历的是 available_indices，而不是从0开始的range(n_inputs)，因为我们只关心未使用的点
            for i in available_indices:
                # 双重检查，确保i确实没被用过（逻辑上应该总是True）
                if i in used_indices:
                    continue

                # 获取第i个输入点可以支持的函数的输入数量
                # 例如，如果input_function_pools[i]中有两个函数，一个input_count=1，另一个input_count=2
                # 那么 possible_input_counts = {1, 2}
                possible_input_counts = set(
                    method_info["input_count"]
                    for _, method_info in input_function_pools[i]
                )

                # 遍历所有可能的输入数量
                # sorted() 确保我们按从小到大的顺序尝试，但这不是必须的
                for input_count in sorted(possible_input_counts):
                    # 从剩余可用的索引中选择，但有一个限制：x >= i
                    # 这个限制是为了减少重复搜索。例如，当我们已经选择了 (0, 2) 作为一组时，
                    # 在后续的搜索中，就不会再以2为起点去尝试包含0的组合，因为0<i=2。
                    # 这是一种剪枝策略，但它**并不完全阻止**像 (2,0) 和 (0,2) 这样的顺序不同组合，
                    # 因为我们使用的是 permutations。
                    # 这一步保证[(0,),(1,2)]和[(1,2),(0,)]不会被重复遍历（语义上是相同的）、
                    # 如果组内顺序（groups）对结果有影响直接使用available_indices
                    # 控制是否启用剪枝逻辑
                    if allow_non_increasing_order:
                        available = [x for x in available_indices if x >= i]
                    else:
                        available = available_indices  # 允许任意顺序组合

                    # 使用 itertools.permutations 生成从 'available' 中选取 'input_count' 个索引的所有排列
                    # 例如：available=[0,1,2], input_count=2 -> (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
                    # 这意味着函数会考虑输入的顺序，[0,1] 和 [1,0] 是两种不同的输入方式。
                    for group in permutations(available, input_count):
                        # 再次检查是否有重复索引（虽然permutations保证无重复，但set长度检查是个好习惯）
                        if len(set(group)) != input_count:
                            continue  # 跳过无效组合

                        # 将这个新生成的组加入当前路径
                        current_groups.append(group)

                        # 递归调用：标记这些索引已被使用，并继续为剩下的索引寻找分组
                        # used_indices + list(group) 创建了一个新的已使用索引列表
                        dfs(used_indices + list(group), current_groups)

                        # 回溯：移除刚才添加的组，尝试其他可能性
                        current_groups.pop()

        # 启动深度优先搜索，初始状态：没有索引被使用，分组列表为空
        dfs([], [])

        print(all_possible_groups)
        # 返回所有找到的合法分组方案
        return all_possible_groups

    def get_group_functions(self, input_function_pools, group):
        """
        获取一个 group 对应的可选函数名列表。
        """
        candidate_funcs = input_function_pools[group[0]]
        valid_funcs = [
            name for name, info in candidate_funcs
            if info.get("input_count", 1) == len(group)
        ]

        if not valid_funcs:
            valid_funcs = ["__empty__"]

        return valid_funcs

    def generate_combinations_for_group(self, input_function_pools, group_list):
        """
        对一个 group_list，生成所有合法的函数组合（笛卡尔积）。
        """
        group_function_options = []
        for group in group_list:
            funcs = self.get_group_functions(input_function_pools, group)
            group_function_options.append(funcs)

        combinations = []
        for func_choices in product(*group_function_options):
            combination = list(zip(group_list, func_choices))
            combinations.append(combination)

        return combinations

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