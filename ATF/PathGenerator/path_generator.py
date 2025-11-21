import random
from copy import deepcopy
import numpy as np
from collections import Counter
import networkx as nx
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(self, graph, methods, remove_from_pool=False, optimize_same_type_inputs=False, type_equivalence_map=None):
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
        # optimize_same_type_inputs用于控制对于输入完全同类型的方法是否要更容易构成一组，逻辑是出现多个相同类型在方法同一位置时将其分配到其他位置，
        # 以增加成组概率，不是特别重要的功能，可能后面实现或者直接不实现
        self.optimize_same_type_inputs = optimize_same_type_inputs

        # 使用标准格式定义默认映射
        default_mapping = {
            "scalar": ["int32", "int64", "float32", "float64", "bool", "uint32", "uint64"]
        }
        
        # 使用双向映射函数构建完整的映射表
        self.type_equivalence_map = self.split_type_mapping(default_mapping)

        # 如果提供了自定义类型对照表，则合并到默认映射中
        if type_equivalence_map:
            # 使用新的双向映射函数处理自定义映射
            custom_mapping = self.split_type_mapping(type_equivalence_map)
            
            # 将自定义映射合并到现有映射中
            for key_type, equiv_types in custom_mapping.items():
                if key_type in self.type_equivalence_map:
                    self.type_equivalence_map[key_type].extend(equiv_types)
                    self.type_equivalence_map[key_type] = list(set(self.type_equivalence_map[key_type]))  # 去重
                else:
                    self.type_equivalence_map[key_type] = equiv_types

    def split_type_mapping(self, type_dict):
        """
        将键值对两边都是类型列表的字典拆分为正向和反向类型映射
        
        Args:
            type_dict (dict): 键值对两边都是类型列表的字典
                            格式: {[type1]: [type2, type3], [type4]: [type5]}
        
        Returns:
            dict: 合并了正向和反向映射的字典
                格式: {source_type: [target_types], target_type: [source_types]}
        """
        result_mapping = {}
        
        for source_types, target_types in type_dict.items():
            # 确保source_types是列表格式
            if not isinstance(source_types, (list, tuple)):
                source_types = [source_types]
            else:
                source_types = list(source_types)
            
            # 确保target_types是列表格式
            if not isinstance(target_types, (list, tuple)):
                target_types = [target_types]
            else:
                target_types = list(target_types)
            
            # 构建正向映射：从source到target
            for source in source_types:
                if source not in result_mapping:
                    result_mapping[source] = []
                for target in target_types:
                    if target not in result_mapping[source]:
                        result_mapping[source].append(target)
            
            # 构建反向映射：从target到source
            for target in target_types:
                if target not in result_mapping:
                    result_mapping[target] = []
                for source in source_types:
                    if source not in result_mapping[target]:
                        result_mapping[target].append(source)
        
        return result_mapping

    def _is_type_compatible(self, data_type, method_input_type):
        """
        检查两个类型是否兼容（包括直接匹配和通过映射表匹配）
        """
        # 直接匹配
        if data_type == method_input_type:
            return True
        
        # 通过映射表匹配
        if data_type in self.type_equivalence_map:
            equivalent_types = self.type_equivalence_map[data_type]
            if method_input_type in equivalent_types:
                return True
        
        return False

    def _build_type_to_method_input_mapping(self, available_types):
        """
        构建类型到方法输入索引的映射字典
        
        Args:
            available_types: 可用的数据类型集合或字典
        
        Returns:
            dict: {data_type: [(method_name, input_idx), ...]}
        """
        type_to_method_input_indices = {}
        for method_name in self.methods:
            if "input_types" in self.methods[method_name]:
                input_types = self.methods[method_name]["input_types"]
                for input_idx, input_type in enumerate(input_types):
                    # 遍历所有可能的类型，检查兼容性
                    for data_type in available_types:
                        if self._is_type_compatible(data_type, input_type):
                            if data_type not in type_to_method_input_indices:
                                type_to_method_input_indices[data_type] = []
                            type_to_method_input_indices[data_type].append((method_name, input_idx))
        return type_to_method_input_indices

    def _select_method_input_with_weights(self, compatible_choices):
        """
        根据权重选择方法输入
        
        Args:
            compatible_choices: 兼容的方法输入选择列表 [(method_name, input_idx), ...]
        
        Returns:
            tuple: (method_name, input_idx) 或 None
        """
        if not compatible_choices:
            return None
            
        weights = []
        for method_name, input_idx in compatible_choices:
            # 获取方法的权重，如果没有weight属性则默认为1
            weight = self.methods[method_name].get("weight", 1.0)
            weights.append(weight)
        
        return random.choices(compatible_choices, weights=weights, k=1)[0]

    def _build_index_to_method_input_mapping(self, collapse_edges, indices_to_process, type_to_method_input_indices):
        """
        为指定索引构建索引到方法输入的映射
        
        Args:
            collapse_edges: 图的边列表
            indices_to_process: 要处理的索引列表
            type_to_method_input_indices: 类型到方法输入的映射
        
        Returns:
            tuple: (index_to_method_input, unmatched_indices)
        """
        index_to_method_input = {}  # idx -> (method_name, input_idx)
        unmatched_indices = []
        available_choices = deepcopy(type_to_method_input_indices) if self.remove_from_pool else type_to_method_input_indices
        
        for idx in indices_to_process:
            if idx >= len(collapse_edges):
                logger.warning(f"索引 {idx} 超出 collapse_edges 范围")
                unmatched_indices.append(idx)
                continue
                
            _, _, data = collapse_edges[idx]
            data_type = data.get("data_type")
            
            if data_type is not None and data_type in available_choices and available_choices[data_type]:
                # 选择与当前数据类型兼容的方法输入
                compatible_choices = [(m, i) for m, i in available_choices[data_type] 
                                    if "input_types" in self.methods[m] and 
                                    i < len(self.methods[m]["input_types"]) and
                                    self._is_type_compatible(data_type, self.methods[m]["input_types"][i])]
                
                chosen_method_input = self._select_method_input_with_weights(compatible_choices)
                if chosen_method_input:
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
        
        return index_to_method_input, unmatched_indices

    def _build_method_call_groups(self, index_to_method_input):
        """
        根据方法输入信息构建方法调用组
        
        Args:
            index_to_method_input: 索引到方法输入的映射 {idx: (method_name, input_idx)}
        
        Returns:
            dict: 分组结果
        """
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
        
        return method_call_groups, method_call_counters

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

        # 构建类型到方法输入的映射
        type_to_method_input_indices = self._build_type_to_method_input_mapping(type_to_indices.keys())

        # 为每个数据索引选择一个兼容的方法输入
        indices_to_process = list(range(num_elements))
        index_to_method_input, unmatched_indices = self._build_index_to_method_input_mapping(
            collapse_edges, indices_to_process, type_to_method_input_indices
        )

        # 根据 method_input 信息进行分组，确保每个方法调用的输入位置正确
        method_call_groups, method_call_counters = self._build_method_call_groups(index_to_method_input)

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
        注:这里替换的索引数由于为是按组删除的，所以实际被替换的索引数可能大于 n。
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
        groups_to_remove = set()
        for method_name, old_group, idx in indices_to_replace:
            if method_name in new_valid_groups and old_group in new_valid_groups[method_name]:
                groups_to_remove.add((method_name, tuple(old_group)))

        # 删除整个组
        for method_name, old_group_tuple in groups_to_remove:
            new_valid_groups[method_name].remove(list(old_group_tuple))

        # 收集被替换的索引（包括被删除组中的所有索引）
        indices_to_be_assigned = []
        for method_name, old_group, idx in indices_to_replace:
            # 添加整个组的所有索引
            for group_idx in old_group:
                if group_idx not in indices_to_be_assigned:
                    indices_to_be_assigned.append(group_idx)
                    
        # 获取需要被替换的索引的类型数据
        collapse_edges = list(self.graph.in_edges("collapse", data=True))
        indices_to_be_assigned_types = {}
        for idx in indices_to_be_assigned:
            if idx < len(collapse_edges):
                _, _, data = collapse_edges[idx]
                data_type = data.get("data_type")
                indices_to_be_assigned_types[idx] = data_type

        # 构建类型到方法输入的映射
        type_to_method_input_indices = self._build_type_to_method_input_mapping(indices_to_be_assigned_types.values())

        # 为每个数据索引选择一个兼容的方法输入
        index_to_method_input, unmatched_indices = self._build_index_to_method_input_mapping(
            collapse_edges, indices_to_be_assigned, type_to_method_input_indices
        )

        # 根据 method_input 信息进行分组，确保每个方法调用的输入位置正确
        method_call_groups, method_call_counters = self._build_method_call_groups(index_to_method_input)

        # 构建结果映射
        new_assignments_map = {}
        valid_groups_new = defaultdict(list)
        
        for (method_name, call_instance), group in method_call_groups.items():
            valid_groups_new[method_name].append(group)
            for idx in group:
                new_assignments_map[idx] = {"method": method_name, "group": tuple(group)}

        # 处理未使用的索引（在分桶后剩余的）
        used_indices = set()
        for group_list in valid_groups_new.values():
            for group in group_list:
                used_indices.update(group)
        
        remaining_unmatched = [idx for idx in indices_to_be_assigned 
                            if idx not in used_indices and idx not in unmatched_indices]
        unmatched_indices.extend(remaining_unmatched)

        # 将未匹配的索引也加入 new_assignments_map
        for idx in unmatched_indices:
            new_assignments_map[idx] = {
                "method": "unmatched",
                "group": tuple([idx])
            }

        # 合并老的和新的 valid_groups
        updated_valid_groups = deepcopy(new_valid_groups)
        for method_name, groups in valid_groups_new.items():
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
        new_unmatched = [[idx] for idx in unmatched_indices]
        if "unmatched" in updated_valid_groups:
            new_unmatched.extend(updated_valid_groups["unmatched"])
        else:
            # 添加原有的unmatched项（如果有的话）
            existing_unmatched = result.get("unmatched", [])
            new_unmatched.extend(existing_unmatched)

        return {
            "index_map": new_index_map,
            "valid_groups": updated_valid_groups,
            "unmatched": new_unmatched
        }