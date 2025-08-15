import random
from copy import deepcopy
import numpy as np
from collections import Counter
import networkx as nx


class PathGenerator:
    def __init__(self, graph, methods):
        """
        初始化路径生成器。
        
        :param graph: 当前图结构（nx.MultiDiGraph）
        :param methods: 方法池字典 {method_name: {"function": ..., "input_count": int, ...}}
        """
        self.graph = graph
        self.methods = methods

    def group_indices_by_input_count(self, indices, input_count, shuffle=False):
        """
        将索引按 input_count 分组。
        """
        if shuffle:
            random.shuffle(indices)
        groups = []
        unmatched = []
        i = 0
        while i < len(indices):
            group = indices[i:i + input_count]
            if len(group) == input_count:
                groups.append(group)
            else:
                unmatched.extend([[idx] for idx in group])
            i += input_count
        return groups, unmatched

    def process_random_method(self, shuffle_indices=True):
        """
        生成一个随机的方法选择路径，并确保多输入方法的输入索引匹配。
        """
        if not self.methods:
            raise ValueError("方法字典为空，无法处理！")
        
        method_list = []
        collapse_edges = list(self.graph.in_edges("collapse", data=True))
        print(f"当前图中 'collapse' 节点的入边数量: {len(collapse_edges)}")
        num_elements = len(collapse_edges)

        for _ in range(num_elements):
            method_name = random.choice(list(self.methods.keys()))
            method_list.append(method_name)

        multi_input_positions = {}
        for idx, method_name in enumerate(method_list):
            input_count = self.methods[method_name]["input_count"]
            if input_count <= 0:
                raise ValueError(f"方法 '{method_name}' 的 input_count 必须大于 0，当前值为 {input_count}")
            multi_input_positions.setdefault(method_name, []).append(idx)

        index_map = {}
        valid_groups = {}
        unmatched = []

        for method_name, indices in multi_input_positions.items():
            input_count = self.methods[method_name]["input_count"]
            groups, unmatch = self.group_indices_by_input_count(indices, input_count, shuffle_indices)
            valid_groups[method_name] = groups
            for group in groups:
                for idx in group:
                    index_map[idx] = {"method": method_name, "group": tuple(group)}
            unmatched.extend(idx for idx in unmatch)

        # 新增：将 unmatched 中的索引也加入 index_map
        for sublist in unmatched:
            for idx in sublist:
                index_map[idx] = {
                    "method": "unmatched",
                    "group": tuple([idx])  # 每个 idx 自己成为一个 group
                }

        return {
            "index_map": index_map,
            "valid_groups": valid_groups,
            "unmatched": unmatched
        }

    def replace_random_elements(self, result, n, shuffle_indices=True):
        """
        替换部分索引对应的方法，并重新分组以符合 input_count 要求。
        """
        index_map = deepcopy(result["index_map"])
        valid_groups = deepcopy(result["valid_groups"])
        unmatched = deepcopy(result.get("unmatched", []))

        # 收集所有可替换的索引条目 (method_name, group, idx)
        all_index_entries = []
        for method_name, groups in valid_groups.items():
            for group in groups:
                for idx in group:
                    all_index_entries.append((method_name, group, idx))

        for unmatched_group in unmatched:
            for unmatched_idx in unmatched_group:
                all_index_entries.append(("unmatched", unmatched_group, unmatched_idx))

        total_indices = len(all_index_entries)
        if n > total_indices:
            raise ValueError(f"无法替换 {n} 个索引，总数只有 {total_indices}")

        # 随机选取 n 个要替换的索引
        indices_to_replace = random.sample(all_index_entries, n)

        # 删除这些索引所属的 group
        new_valid_groups = deepcopy(valid_groups)
        for method_name, old_group, idx in indices_to_replace:
            if method_name in new_valid_groups and old_group in new_valid_groups[method_name]:
                new_valid_groups[method_name].remove(old_group)

        # 收集被替换的索引
        indices_to_be_assigned = [idx for _, _, idx in indices_to_replace]

        # 分配新方法给这些索引
        new_assignments = {}
        for idx in indices_to_be_assigned:
            method_name = random.choice(list(self.methods.keys()))
            new_assignments.setdefault(method_name, []).append(idx)

        # 按照 input_count 分组
        assignments_method_group_positions = {"unmatched": []}
        for method_name, indices in new_assignments.items():
            input_count = self.methods[method_name]["input_count"]
            if shuffle_indices:
                random.shuffle(indices)
            groups, unmatch = self.group_indices_by_input_count(indices, input_count, shuffle_indices)
            assignments_method_group_positions[method_name] = groups
            assignments_method_group_positions["unmatched"].extend([[idx] for idx in unmatch])

        # 合并老的和新的 valid_groups
        updated_valid_groups = deepcopy(new_valid_groups)
        for method_name, groups in assignments_method_group_positions.items():
            if not groups or method_name == "unmatched":
                continue
            if method_name in updated_valid_groups:
                updated_valid_groups[method_name].extend(groups)
            else:
                updated_valid_groups[method_name] = groups

        # 构建新的 index_map
        new_index_map = {}
        for method_name, groups in updated_valid_groups.items():
            for group in groups:
                for idx in group:
                    new_index_map[idx] = {
                        "method": method_name,
                        "group": tuple(group)  # 保持 hashable
                    }

        # 处理未匹配项（unmatched）
        new_unmatched = assignments_method_group_positions["unmatched"]

        return {
            "index_map": new_index_map,
            "valid_groups": updated_valid_groups,
            "unmatched": new_unmatched
        }