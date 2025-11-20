import random
from copy import deepcopy
import numpy as np
from collections import Counter
import networkx as nx
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(self, graph, methods, remove_from_pool=False, optimize_same_type_inputs=False):
        """
        初始化路径生成器。
        目前正在做类型匹配机制
        
        :param graph: 当前图结构（nx.MultiDiGraph）
        :param methods: 方法池字典 {method_name: {"function": ..., "input_count": int, ...}}
        :param remove_from_pool: 是否从随机池中移出已被选择的方法索引
        :param optimize_same_type_inputs: 当方法的所有输入类型相同时，优化匹配过程
        """
        self.graph = graph
        self.methods = methods
        self.remove_from_pool = remove_from_pool
        self.optimize_same_type_inputs = optimize_same_type_inputs

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
        
        collapse_edges = list(self.graph.in_edges("collapse", data=True))
        num_elements = len(collapse_edges)

        # 建立类型对应索引位置列表的字典
        type_to_indices = {}
        for idx, (u, v, data) in enumerate(collapse_edges):
            data_type = data.get("data_type")
            if data_type is not None:
                type_to_indices.setdefault(data_type, []).append(idx)

        # 建立类型对应方法输入索引字典
        type_to_method_input_indices = {}
        for method_name in self.methods:
            if "input_types" in self.methods[method_name]:
                input_types = self.methods[method_name]["input_types"]
                for input_idx, input_type in enumerate(input_types):
                    if input_type not in type_to_method_input_indices:
                        type_to_method_input_indices[input_type] = []
                    type_to_method_input_indices[input_type].append((method_name, input_idx))

        # 为每个数据索引选择一个兼容的方法输入
        index_to_method_input = {}  # idx -> (method_name, input_idx)
        unmatched_indices = []
        available_choices = deepcopy(type_to_method_input_indices) if self.remove_from_pool else type_to_method_input_indices
        
        for idx in range(num_elements):
            _, _, data = collapse_edges[idx]
            data_type = data.get("data_type")
            
            if data_type is not None and data_type in available_choices and available_choices[data_type]:
                # 选择与当前数据类型和位置兼容的方法输入
                compatible_choices = [(m, i) for m, i in available_choices[data_type] 
                                    if "input_types" in self.methods[m] and 
                                    i < len(self.methods[m]["input_types"]) and
                                    self.methods[m]["input_types"][i] == data_type]
                
                if compatible_choices:
                    chosen_method_input = random.choice(compatible_choices)
                    index_to_method_input[idx] = chosen_method_input
                    
                    if self.remove_from_pool:
                        available_choices[data_type].remove(chosen_method_input)
                else:
                    logger.warning(f"索引 {idx} 的类型 {data_type} 没有找到兼容的方法输入")
                    unmatched_indices.append(idx)
            else:
                logger.warning(
                    f"索引 {idx} 无法匹配到支持的类型，可能是开启 remove_from_pool 或提供的方法池与数据不符导致",
                )
                unmatched_indices.append(idx)

        # 根据 method_input 信息进行分组，确保每个方法调用的输入位置正确
        method_call_groups = defaultdict(list)  # (method_name, call_instance) -> [idx1, idx2, ...]
        method_call_counters = defaultdict(int)  # 记录每个方法的调用实例ID

        # 按 (method_name, input_idx) 分桶
        method_input_buckets = defaultdict(lambda: defaultdict(list))
        for idx, (method_name, input_idx) in index_to_method_input.items():
            method_input_buckets[method_name][input_idx].append(idx)

        # 为每个方法构建完整的调用组
        for method_name, buckets in method_input_buckets.items():
            expected_input_count = len(self.methods[method_name]["input_types"])
            
            if expected_input_count == 1:
                # 单输入方法：每个索引独立成组
                for idx in buckets[0]:
                    call_key = (method_name, method_call_counters[method_name])
                    method_call_groups[call_key] = [idx]
                    method_call_counters[method_name] += 1
            else:
                # 多输入方法：需要每个输入位置都有值
                available_inputs = [buckets[i] for i in range(expected_input_count) if i in buckets]
                if len(available_inputs) == expected_input_count:
                    # 确保每个输入位置都有足够的索引
                    min_len = min(len(input_list) for input_list in available_inputs)
                    for call_i in range(min_len):
                        group = [buckets[input_pos][call_i] for input_pos in range(expected_input_count)]
                        call_key = (method_name, method_call_counters[method_name])
                        method_call_groups[call_key] = group
                        method_call_counters[method_name] += 1

        # 构建结果映射
        index_map = {}
        valid_groups = defaultdict(list)
        
        for (method_name, call_instance), group in method_call_groups.items():
            valid_groups[method_name].append(group)
            for idx in group:
                index_map[idx] = {"method": method_name, "group": tuple(group)}

        # 处理未使用的索引（在分桶后剩余的）
        used_indices = set()
        for group_list in valid_groups.values():
            for group in group_list:
                used_indices.update(group)
        
        remaining_unmatched = [idx for idx in range(num_elements) 
                            if idx not in used_indices and idx not in unmatched_indices]
        unmatched_indices.extend(remaining_unmatched)

        # 将未匹配的索引也加入 index_map
        for idx in unmatched_indices:
            index_map[idx] = {
                "method": "unmatched",
                "group": tuple([idx])
            }

        return {
            "index_map": index_map,
            "valid_groups": dict(valid_groups),
            "unmatched": [[idx] for idx in unmatched_indices]
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