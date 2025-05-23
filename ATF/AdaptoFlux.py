import random
from enum import Enum
import uuid
import numpy as np
import inspect
import importlib.util
import math
import traceback
from . import DynamicWeightController
from threading import Thread, Event
from collections import Counter
import os
import shutil
import networkx as nx


# 定义一个枚举表示不同的坍缩方法
class CollapseMethod(Enum):
    SUM = 1       # 求和
    AVERAGE = 2   # 平均
    VARIANCE = 3  # 方差
    PRODUCT = 4   # 相乘
    CUSTOM = 5  #用于自定义方法

class AdaptoFlux:
    def __init__(self, values, labels, methods_path, collapse_method=CollapseMethod.SUM):
        """
        初始化 AdaptoFlux 类的实例
        
        :param values: 一维数据值列表
        :param labels: 每个值对应的标签列表
        :param collapse_method: 选择的坍缩方法，默认为 SUM
        :param methods_path: 存储方法路径的文件路径，默认为 "methods.py"
        """

        # 检查 values 是二维的
        assert len(values.shape) == 2, f"values 必须是二维的 (样本数, 特征维度)，但得到 shape={values.shape}"
        
        # 检查 labels 是一维的，并且样本数一致
        assert len(labels.shape) == 1, f"labels 必须是一维的 (样本数,)，但得到 shape={labels.shape}"
        assert values.shape[0] == labels.shape[0], f"values 和 labels 样本数量不一致：{values.shape[0]} vs {labels.shape[0]}"

        # 存储输入数据
        self.values = values  # 原始数值列表
        self.labels = labels  # 对应的标签列表

        # 存储处理过程中的值
        self.last_values = values  # 记录上一次处理后的值
        self.history_values = [values]  # 记录历史处理值

        # 记录方法及其预输入信息
        self.methods = {}  # 存储方法的字典
        self.method_inputs = {}  # 存储每个方法的预输入索引
        self.history_method_inputs = []  # 记录历史每层的预输入索引
        # 意外发现即使不使用历史记录和清空不可取的网络残余（即被清空的网络依然参与预输入索引和预输入值计算）依然会出现概率上升和方差下降
        
        # 存储路径信息
        # self.paths = []  # 记录每个值对应的路径
        # self.max_probability_path = [] # 记录最高概率对应的路径
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("root") 
        self.graph.add_node('collapse')
        for feature_index in range(self.values.shape[1]):  # 遍历特征维度
            self.graph.add_edge(
                "root",
                "collapse",
                output_index=feature_index, # 函数局部索引
                data_coord=feature_index  # 当前层全局索引
            )
        self.layer = 0
        
        # 选择的坍缩方法
        self.collapse_method = collapse_method  # 默认使用 SUM 方法
        self.custom_collapse_function = None  # 预定义自定义坍缩方法，默认值为 None
        
        # 记录推理过程中的输入值
        self.method_input_values = {}  # 记录当前层的方法输入值
        self.history_method_input_values = []  # 记录历史层的方法输入值
        
        # 监控当前任务的性能指标
        self.metrics = {
            "accuracy": 0.0,  # 准确率
            "max_accuracy": 0.0,  # 最高准确率
            "entropy": 0.0,  # 路径熵值
            "redundancy_penalty": 0.0,  # 冗余惩罚
        }

        # 记录最高准确率未更新但发生回退的次数
        self.rollback_count = 0

        # 存储文件路径
        self.methods_path = methods_path  # 默认为 "methods.py"
    
    def add_collapse_method(self, collapse_function):
        """
        允许用户自定义坍缩方法，并替换现有的坍缩方法
        :param collapse_function: 传入一个函数
        """
        if callable(collapse_function):
            self.custom_collapse_function = collapse_function
            self.collapse_method = CollapseMethod.CUSTOM  # 标记当前使用的是自定义坍缩方法
        else:
            raise ValueError("提供的坍缩方法必须是一个可调用函数")

    # 添加处理方法到字典
    def add_method(self, method_name, method, input_count, output_count):
        """
        添加方法到字典
        :param method_name: 方法名称
        :param method: 方法本身
        :param input_count: 方法所需的输入值数量
        """ 
        self.methods[method_name] = {
            "function" : method,
            "input_count" : input_count,
            "output_count" : output_count,
        }
        self.method_inputs[method_name] = []
        self.method_input_values[method_name] = [] # 疑似没有必要

    def import_methods_from_file(self):
        """
        从指定文件中导入所有方法并添加到方法字典中。
        :param file_path: Python 文件路径
        """
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("module.name", self.methods_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 遍历模块中的所有成员
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):  # 检查是否为函数
                # 获取函数所需的参数数量
                input_count = len(inspect.signature(obj).parameters)
                output_count = getattr(obj, "output_count", 1)
                self.add_method(name, obj, input_count, output_count)
        # 记录初始状态
        self.history_method_inputs.append(self.method_inputs)
        self.history_method_input_values.append(self.method_input_values)
        # self.history_method_input_val_values.append(self.method_input_val_values)
        print(self.methods)

    # 根据当前选择的坍缩方法，对输入值进行计算
    def collapse(self, values):
        """
        根据设定的坍缩方法（collapse_method）对输入值进行聚合计算。

        :param values: 需要进行坍缩计算的数值列表
        :return: 计算后的坍缩值
        :raises ValueError: 如果指定的坍缩方法未知，则抛出异常
        """
        if self.collapse_method == CollapseMethod.SUM:
            return self._collapse_sum(values)
        elif self.collapse_method == CollapseMethod.AVERAGE:
            return self._collapse_average(values)
        elif self.collapse_method == CollapseMethod.VARIANCE:
            return self._collapse_variance(values)
        elif self.collapse_method == CollapseMethod.PRODUCT:
            return self._collapse_product(values)
        elif self.collapse_method == CollapseMethod.CUSTOM and self.custom_collapse_function:
            return self.custom_collapse_function(values)
        else:
            raise ValueError("未知的坍缩方法")

    # 计算输入值的总和
    def _collapse_sum(self, values):
        """
        计算输入值的总和。

        :param values: 需要计算的数值列表
        :return: values 的总和
        """
        return np.sum(values)

    # 计算输入值的平均值
    def _collapse_average(self, values):
        """
        计算输入值的平均值。

        :param values: 需要计算的数值列表
        :return: values 的均值
        """
        return np.mean(values)

    # 计算输入值的方差
    def _collapse_variance(self, values):
        """
        计算输入值的方差。

        :param values: 需要计算的数值列表
        :return: values 的方差
        """
        return np.var(values)

    # 计算输入值的乘积
    def _collapse_product(self, values):
        """
        计算输入值的乘积。

        :param values: 需要计算的数值列表
        :return: values 的乘积
        """
        return np.prod(values)
    
    def group_indices_by_input_count(self, indices, input_count, shuffle=False):
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

    # 生成单次路径选择
    def process_random_method(self, shuffle_indices=True):
        """
        生成一个随机的方法选择路径，并确保多输入方法的输入索引匹配。

        :param shuffle_indices: 是否打乱多输入方法的索引，默认为 True
        :return: method_list，包含为每个元素随机分配的方法（可能是单输入或多输入方法）
        :raises ValueError: 如果方法字典为空或值列表为空，则抛出异常
        """
        if not self.methods:
            raise ValueError("方法字典为空，无法处理！")
        if self.last_values.size == 0:  # 处理 NumPy 数组的情况
            raise ValueError("值列表为空，无法处理！")

        method_list = []
        collapse_edges = list(self.graph.in_edges("collapse", data=True))
        num_elements = len(collapse_edges)

        # 1. 初始为全部正值方法
        for _ in range(num_elements):
            method_name = random.choice(list(self.methods.keys()))
            method_list.append(method_name)

        # 2. 收集方法的位置
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
            # 未分组的数据每一个单独视为一个组
            unmatched.extend(idx for idx in unmatch)

        # 新增：将 unmatched 中的索引也加入 index_map
        for sublist in unmatched:
            for idx in sublist:
                print(idx)
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
        
        :param result: 来自 process_random_method 的输出结构
        :param n: 要随机替换的索引数量
        :param shuffle_indices: 是否打乱新方法分配时的索引顺序
        :return: 新的 result 结构，包含更新后的 index_map、valid_groups 和 unmatched
        """
        from copy import deepcopy

        # 提取原始信息
        index_map = deepcopy(result["index_map"])
        valid_groups = deepcopy(result["valid_groups"])
        unmatched = deepcopy(result.get("unmatched", []))

        # 收集所有可替换的索引条目 (method_name, group, idx)
        all_index_entries = []
        for method_name, groups in valid_groups.items():
            for group in groups:
                for idx in group:
                    all_index_entries.append((method_name, group, idx))

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

    def append_nx_layer(self, result, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        将给定的结果转换为图中的一层新节点。
        
        参数:
            result: 一个字典，包含以下键：
                - 'valid_groups': 按照方法名分组的有效数据索引列表
                - 'unmatched': 无法匹配成完整输入的剩余索引
            discard_unmatched: 如何处理未匹配项。可选值：
                'to_discard' - 创建单独节点并连接回 collapse
                'ignore' - 忽略这些未匹配项
            discard_node_method_name: 如果 discard_unmatched == 'to_discard'，则用该名称作为丢弃节点的 method_name
        """

        # 增加层数计数器
        self.layer += 1
        
        # 记录当前新增边的 data_coord 编号，保证唯一性(模型总索引不会重复)
        new_index_edge = 0

        # 初始化每个方法使用次数的计数器（用于生成唯一的节点名）
        method_counts = {method_name: 0 for method_name in self.methods}
        method_counts["unmatched"] = 0  # 单独记录 unmatched 的数量

        # 获取所有指向 "collapse" 的入边（即上一层输出的边）
        collapse_edges = list(self.graph.in_edges("collapse", data=True))

        # 设置目标节点为 collapse（统一变量名方便后续操作）
        v = "collapse"

        # 遍历所有有效的方法及其对应的分组
        for method_name, groups in result['valid_groups'].items():
            if method_name == "unmatched":
                continue  # 跳过无效的 unmatched 分组，在后面单独处理

            for group in groups:
                # 构造新的目标节点名：如 "1_0_add"
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                # 添加新节点到图中，并设置 method_name 属性
                self.graph.add_node(new_target_node, method_name=method_name)

                # 遍历之前所有的 collapse 入边
                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        # 如果这条边的数据坐标属于当前分组，则删除原边，并将它连接到新节点
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, new_target_node, **data)

                # 新增从新节点到 collapse 的输出边，数量由方法定义决定（output_count）
                for local_output_index in range(self.methods[method_name]["output_count"]):
                    self.graph.add_edge(new_target_node, v, output_index = local_output_index, data_coord=new_index_edge)
                    new_index_edge += 1

                # 更新该方法使用的计数
                method_counts[method_name] += 1

        # 处理未匹配项（如果存在）
        unmatched_groups = result.get('unmatched', [])
        if unmatched_groups and discard_unmatched == 'to_discard':
            for group in unmatched_groups:
                # 构造丢弃节点名，如 "1_2_unmatched"
                node_name = f"{self.layer}_{method_counts['unmatched']}_unmatched"
                # 添加节点并指定 method_name 为传入的 discard_node_method_name
                self.graph.add_node(node_name, method_name=discard_node_method_name)

                # 同样地，把对应 data_coord 的边从 collapse 移动到这个丢弃节点
                for u, _, data in collapse_edges:
                    if data.get('data_coord') in group:
                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(u, node_name, **data)

                # 从丢弃节点再连一条边回到 collapse（默认输出一条）
                for local_output_index in range(1):
                    self.graph.add_edge(node_name, v, output_index = local_output_index, data_coord=new_index_edge)
                    new_index_edge += 1

                method_counts["unmatched"] += 1

        elif unmatched_groups and discard_unmatched != 'ignore':
            # 如果有未匹配项且不是 ignore 或 to_discard，则报错
            raise ValueError(f"未知的 discard_unmatched 值：{discard_unmatched}。支持的选项为 'ignore' 或 'to_discard'")

    def remove_last_nx_layer(self):
        """
        删除图中最后一层节点，并将它们的输入直接连接回 collapse，
        实现“撤回”一层的操作。
        """

        collapse_node = "collapse"

        # 获取所有连接到 collapse 的入边（即最后一层的所有输出边）
        incoming_edges_to_collapse = list(self.graph.in_edges(collapse_node, data=True))

        # 提取这些边的源节点，即要删除的那一层的节点
        nodes_to_remove = set(u for u, v, d in incoming_edges_to_collapse)

        # 如果没有节点需要删除，就提示用户
        if not nodes_to_remove:
            print("没有可删除的层。")
            return

        # 收集要删除节点的所有输入边（即倒数第二层 -> 最后一层）
        input_edges_map = {
            node: list(self.graph.in_edges(node, data=True)) for node in nodes_to_remove
        }

        # 开始重建连接：将每个被删除节点的输入边直接连接回 collapse
        for node in nodes_to_remove:
            for prev_node, _, edge_data in input_edges_map[node]:
                # 添加边，保留原来的 data 属性（如 data_coord）
                self.graph.add_edge(prev_node, collapse_node, **edge_data)

        # 删除这些旧节点（但跳过 root 节点，因为它是根节点，不能删）
        for node in nodes_to_remove:
            if node != "root":
                self.graph.remove_node(node)
            else:
                print(f"跳过删除节点：{node}（这是保留的根节点）")

        # 层数减一
        self.layer -= 1

    def infer_with_graph(self, values):
        """
        使用图结构对输入 values 进行推理计算，支持多输入/多输出操作。
        
        参数:
            values (np.ndarray): 输入数据，形状为 [样本数, 特征维度]
        
        返回:
            np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        # 存储每个节点的输出结果，格式为 {节点名: 输出数组}
        node_outputs = {}
        
        # 存储多输入方法的待处理缓存，格式为 {节点名: 输入数据列表}
        input_buffers = {}

        # 获取所有非 root/collapse 的节点并按拓扑排序
        # 确保节点按依赖顺序处理（前驱节点先于后继节点）
        nodes_in_order = list(nx.topological_sort(self.graph))
        nodes_in_order.remove("collapse")  # 移除特殊节点
        if "root" in nodes_in_order:
            nodes_in_order.remove("root")  # 移除特殊节点

        # 初始化 root 节点的输出（输入数据直接作为 root 的输出）
        root_output = values.copy()
        node_outputs["root"] = root_output

        # 遍历图中的所有节点（按拓扑顺序）
        for node in nodes_in_order:
            # 获取当前节点的属性和方法名
            node_data = self.graph.nodes[node]
            method_name = node_data.get("method_name", None)

            # 处理特殊 "null" 方法节点（数据透传）
            if method_name == "null":
                # 获取当前节点的所有前驱节点
                predecessors = list(self.graph.predecessors(node))
                
                # 校验：null 方法节点只能有一个前驱节点
                if len(predecessors) > 1:
                    raise ValueError(
                        f"节点 {node} 使用了 'null' 方法，但有多个前驱节点 {predecessors}。"
                        f"'null' 方法不支持多输入，请检查图结构，确保该节点只有一个前驱节点，"
                        f"或者自定义方法来处理多输入逻辑（如拼接、加权等）。"
                    )
                
                # 如果有前驱节点，直接透传其输出
                if predecessors:
                    node_outputs[node] = node_outputs.get(predecessors[0], np.array([]))
                
                continue  # 跳过后续处理

            # ========== 普通方法节点处理 ==========
            
            # 1. 获取方法信息
            if method_name not in self.methods:
                raise ValueError(f"未知方法名 {method_name}，请确保已正确加载方法")
            
            # 从预定义方法中获取函数对象和输入输出参数配置
            method_info = self.methods[method_name]
            func = method_info["function"]         # 实际调用的函数
            input_count = method_info["input_count"]  # 方法需要的输入数量
            output_count = method_info["output_count"]  # 方法产生的输出数量

            # 2. 收集前驱节点的输出作为输入
            # 收集当前节点的所有入边
            in_edges = list(self.graph.in_edges(node, data=True))

            if not in_edges:
                raise ValueError(f"节点 {node} 没有入边")

            inputs = []
            for src, dst, edge_data in in_edges:
                if src not in node_outputs:
                    raise ValueError(f"前驱节点 {src} 没有输出")

                output_index = edge_data.get("output_index")  # 获取边上指定的节点局部索引
                if output_index is not None:
                    # 取出前驱输出中的指定索引值
                    samples = node_outputs[src]

                    try:
                        # 提取每个样本中指定列的数据，并包装成二维列表
                        selected_column = [[sample[output_index]] for sample in samples]

                    except IndexError:
                        raise ValueError(f"前驱节点 {src} 的输出长度不足，无法获取 output_index={output_index}")
                    
                    inputs.append(selected_column)
                else:
                    # 如果没有指定 output_index，则将前驱输出全部展开作为输入
                    inputs.extend(node_outputs[src])

            # ====== 新增：合并所有列 ======
            if not inputs:
                raise ValueError("没有可合并的输入列")

            # 检查所有列样本数量一致
            num_samples = len(inputs[0])
            for col in inputs:
                if len(col) != num_samples:
                    raise ValueError("所有输入列的样本数必须一致")

            # 合并列
            final_inputs = []
            for i in range(num_samples):
                merged = []
                for col in inputs:
                    merged.extend(col[i])
                final_inputs.append(merged)

            # 最终返回的就是每个样本一行的标准结构
            inputs = final_inputs

            # 3. 合并前驱节点的输出为一个二维数组
            # 形状为 (样本数, 总特征数)，所有输入按列拼接
            flat_inputs = np.array(inputs)
            num_samples = flat_inputs.shape[0]  # 获取样本数量

            # 4. 执行方法函数
            results = []
            for row in flat_inputs:
                result = func(*row)  # 解包参数并调用函数
                
                # 统一返回值格式（标量->列表，数组->列表）
                if isinstance(result, (int, float)):
                    result = [result]
                elif isinstance(result, np.ndarray):
                    result = result.tolist()
                results.append(result)

            # 5. 构造输出数组
            # 形状为 (样本数, 输出数量*结果数)
            new_output = np.zeros((num_samples, output_count))
            
            # 将结果填充到输出数组中
            for i, res in enumerate(results):
                for j, val in enumerate(res):
                    new_output[i, j] = val  # 按行填充
            
            # 存储当前节点的输出
            node_outputs[node] = new_output

        # ========== 最终结果收集 ==========

        # 收集 (global_coord, column_data) 对
        temp_result = []

        for u, v, data in self.graph.in_edges("collapse", data=True):
            if v == "collapse":
                local_index = data.get('output_index')
                global_coord = data.get('data_coord')

                if local_index is None or global_coord is None:
                    raise ValueError("缺少 output_index 或 data_coord 属性")

                print("Nodes in topological order:", nodes_in_order)
                output_array = node_outputs[u]


                try:
                    column_data = output_array[:, local_index]
                    temp_result.append((global_coord, column_data))
                except IndexError:
                    raise ValueError(f"节点 {u} 输出维度不足，无法提取 output_index={local_index}")

        # 按照 data_coord 排序，保证输出顺序一致
        temp_result.sort(key=lambda x: x[0])

        # 只保留数据部分
        collapse_result = [col for _, col in temp_result]

        # 将所有输出列拼接为一个二维数组
        raw_output = np.column_stack(collapse_result)

        # 应用 collapse 方法，按行进行聚合（例如每行变成一个值）
        collapsed_output = np.apply_along_axis(self.collapse, axis=1, arr=raw_output)
        
        # 将所有输出列拼接为最终结果
        return collapsed_output

    def infer_with_graph_single(self, sample):
        """
        使用图结构对单个样本进行推理计算。
        
        参数:
            sample (np.ndarray or list): 单个样本，形状为 [特征维度]
        
        返回:
            float or np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        # 确保输入是一维数组
        sample = np.array(sample)
        assert len(sample.shape) == 1, "输入必须是一维数组"

        # 扩展为二维 (1, 特征维度)
        values = sample.reshape(1, -1)

        # 调用批量推理方法
        result = self.infer_with_graph(values)

        # 返回单个样本的结果
        return result[0]

    def save_model(self, folder="models"):
        """
        保存模型的路径数据、图结构（.gexf）和相关文件。
        
        1. 确保目标文件夹存在，如果不存在则创建。
        2. 将模型的路径信息写入到 output.txt。
        3. 将图结构保存为 graph.gexf（可读性强）。
        4. 复制指定的 methods_path 文件到目标文件夹。
        
        参数:
            folder (str): 用于保存模型的文件夹路径，默认为 "models"。
        """
        import json
        from networkx.readwrite import json_graph

        # 确保文件夹存在，如果不存在则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 保存图结构到 graph.gexf（可读性强）
        gexf_file_path = os.path.join(folder, "graph.gexf")
        nx.write_gexf(self.graph, gexf_file_path)

        json_file_path = os.path.join(folder, "graph.json")
        try:
            data = json_graph.node_link_data(self.graph)  # 转换为可序列化的字典
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存 JSON 文件时出错: {e}")

        # 复制 methods_path 文件到保存的文件夹
        if os.path.exists(self.methods_path):  # 确保源文件存在
            shutil.copy(self.methods_path, folder)  # 复制文件到目标文件夹

    def get_path_entropy(self, paths):
        """
        该方法已废弃
        计算路径熵

        :param paths: 需要计算的二维列表路径数据
        :return: 计算得到的路径熵值
        """
        try:
            method_counter = Counter()

            # 统计每种方法的出现次数
            for node in self.graph.nodes:
                data = self.graph.nodes[node]
                method_name = data.get("method_name")
                if method_name and method_name != "null":  # 忽略 null 节点
                    method_counter[method_name] += 1

            if not method_counter:
                return 0.0

            # 计算概率分布
            total = sum(method_counter.values())
            probabilities = [count / total for count in method_counter.values()]

            # 计算香农熵
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

            return entropy
        except ValueError as ve:
            print(f"值错误: {ve}")
            return None
        except Exception as e:
            print(f"计算路径熵时发生错误: {e}")
            print("路径数据:", paths)  # 打印路径数据以便定位问题
            return None
        
    def get_graph_entropy(self):
        """
        计算图结构的熵值，基于节点和方法类型的分布。
        示例计算方法，可根据实际需求替换。
        :return: 计算得到的图结构熵值
        """
        method_counter = Counter()

        # 统计每种方法的出现次数
        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            method_name = data.get("method_name")
            if method_name and method_name != "null":  # 忽略 null 节点
                method_counter[method_name] += 1

        if not method_counter:
            return 0.0

        # 计算概率分布
        total = sum(method_counter.values())
        probabilities = [count / total for count in method_counter.values()]

        # 计算香农熵
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        return entropy
    
    def get_method_counter(self):
        """
        统计图中各 method_name 的出现次数
        """
        from collections import Counter
        method_counter = Counter()

        for node in self.graph.nodes:
            data = self.graph.nodes[node]
            method_name = data.get("method_name")
            if method_name and method_name != "null":  # 忽略 null 节点
                method_counter[method_name] += 1

        return method_counter
        
    def RMSE(self, y_true, y_pred):
        """
        计算均方根误差 (RMSE)

        Args:
            y_true (array-like): 真实值
            y_pred (array-like): 预测值

        Returns:
            float: RMSE 值
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)  # 计算均方误差 (MSE)
        return np.sqrt(mse)  # 计算 RMSE
                    
    # 训练方法,epochs决定最终训练出来的模型层数,step用于控制重随机时每次增加几个重随机的指数上升速度 # 第一轮训练如果直接失败会出现错误，待解决
    def training(self, epochs=20, depth_interval=1, depth_reverse=1, step=2, target_accuracy=None):
        """
        训练方法，用于训练模型，执行指定的轮次并在每一轮中根据训练集和验证集的表现进行调整。
        
        - 该方法会根据给定的 `epochs` 参数迭代指定次数，逐步调整模型的权重和输入方法。
        - 在每一轮迭代中，会计算当前模型的训练集准确率、方差以及与前一轮相比的变化。
        - 训练过程中可能会根据预设的条件重新调整方法的输入，尝试不同的路径来改进模型表现。
        - 如果本轮训练的方差较小或者准确率较高，则将当前模型保存到历史记录中；否则，会进行调整，重新计算适合的路径。
        
        参数：
            epochs (int): 训练的轮次，决定了模型训练的次数，默认为 10000。
            depth_interval (int): 控制深度的增量，默认为 1。
            depth_reverse (int): 控制深度的反向调整，默认为 1。
            step (int): 控制重随机时每次增加的指数上升速度，默认为 2。
        
        返回值：
            None: 该方法不返回任何值，而是直接修改类的状态并进行模型训练。
        """
        try:
            # 清空路径列表
            self.paths = []
            # 创建动态权重控制器
            dynamicWeightController = DynamicWeightController.DynamicWeightController(epochs)

            # 训练循环
            for i in range(1, epochs + 1):
                print("epoch:", i)
                last_method = self.process_random_method()  # 获取当前的随机方法
                new_last_values = self.process_array_with_list(last_method)  # 根据随机方法处理数据

                # 计算训练集方差和准确率
                last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
                new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
                last_vs_train = last_collapse_values == self.labels  # 计算训练集的相等情况
                new_vs_train = new_collapse_values == self.labels  # 计算新训练集的相等情况
                
                # 计算准确率和损失
                last_accuracy_trian = np.mean(last_vs_train)  # 计算上一轮训练集准确率
                new_accuracy_trian = np.mean(new_vs_train)  # 计算本轮训练集准确率
                last_loss_value = self.RMSE(self.labels, last_collapse_values)
                new_loss_value = self.RMSE(self.labels, new_collapse_values)

                print("上一轮训练集相等概率:" + str(last_accuracy_trian))
                print("本轮训练集相等概率：" + str(new_accuracy_trian))

                # 计算前后路径熵和从动态权重控制器获取权值
                last_path_entropy = self.get_path_entropy(self.paths)
                new_path_entropy = self.get_path_entropy(self.paths+ [last_method])


                last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
                new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

                # 计算指导值（暂未编写冗余部分）
                last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
                new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value

                print("上一轮训练集指导值:" + str(last_guiding_value))
                print("本轮训练集指导值：" + str(new_guiding_value))
                
                # 判断是否要更新网络路径
                if (last_guiding_value < new_guiding_value) and new_last_values.size != 0:
                    self.paths.append(last_method)
                    self.history_values.append(new_last_values)
                    self.last_values = new_last_values
                    self.history_method_inputs.append(self.method_inputs)
                    self.history_method_input_values.append(self.method_input_values)
                    self.metrics["accuracy"] = new_accuracy_trian
                    self.metrics["entropy"] = new_path_entropy

                    if self.metrics["accuracy"] > self.metrics["max_accuracy"]:
                        self.metrics["max_accuracy"] = self.metrics["accuracy"]

                    self.rollback_count = 0

                    if target_accuracy is not None and new_accuracy_trian >= target_accuracy:
                        print(f"训练提前停止，准确率达到 {new_accuracy_trian}，大于等于目标 {target_accuracy}")
                        # 保存模型或处理
                        self.save_model()  # 假设 save_model 方法已经定义
                        return
                else:
                    # 如果本轮训练不符合要求，重随机重新寻找合适的路径
                    j = 1
                    while j <= len(last_method):
                        print(f"在当层重新寻找适合的路径：当前重随机数{j}")
                        if self.history_method_inputs:  # 检查是否有历史输入
                            self.method_inputs = self.history_method_inputs[-1]

                        if self.history_method_input_values:  # 检查是否有历史输入值
                            self.method_input_values = self.history_method_input_values[-1]

                        last_method = self.replace_random_elements(last_method, j)  # 替换方法中的随机元素
                        j *= step  # 增加重随机的步长
                        
                        new_last_values = self.process_array_with_list(last_method)  # 处理新的数据

                        # 计算训练集方差和准确率
                        last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
                        new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
                        last_vs_train = last_collapse_values == self.labels  # 计算训练集的相等情况
                        new_vs_train = new_collapse_values == self.labels  # 计算新训练集的相等情况
                        
                        # 计算准确率和损失
                        last_accuracy_trian = np.mean(last_vs_train)  # 计算上一轮训练集准确率
                        new_accuracy_trian = np.mean(new_vs_train)  # 计算本轮训练集准确率
                        last_loss_value = self.RMSE(self.labels, last_collapse_values)
                        new_loss_value = self.RMSE(self.labels, new_collapse_values)

                        print("上一轮训练集相等概率:" + str(last_accuracy_trian))
                        print("本轮训练集相等概率：" + str(new_accuracy_trian))

                        # 计算前后路径熵和从动态权重控制器获取权值
                        last_path_entropy = self.get_path_entropy(self.paths)
                        new_path_entropy = self.get_path_entropy(self.paths+ [last_method])

                        last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
                        new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

                        # 计算指导值（暂未编写冗余部分）
                        last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
                        new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value

                        print("上一轮训练集指导值:" + str(last_guiding_value))
                        print("本轮训练集指导值：" + str(new_guiding_value))

                        # 判断是否需要更新路径
                        if (last_guiding_value < new_guiding_value) and new_last_values.size != 0:
                            self.paths.append(last_method)
                            self.history_values.append(new_last_values)
                            self.last_values = new_last_values
                            self.history_method_inputs.append(self.method_inputs)
                            self.history_method_input_values.append(self.method_input_values)
                            self.metrics["accuracy"] = new_accuracy_trian
                            self.metrics["entropy"] = new_path_entropy
                            
                            if self.metrics["accuracy"] > self.metrics["max_accuracy"]:
                                self.metrics["max_accuracy"] = self.metrics["accuracy"]
                            
                            self.rollback_count = 0

                            if target_accuracy is not None and new_accuracy_trian >= target_accuracy:
                                print(f"训练提前停止，准确率达到 {new_accuracy_trian}，大于等于目标 {target_accuracy}")
                                # 保存模型或处理
                                self.save_model()  # 假设 save_model 方法已经定义
                                return  
                            break

                    else:
                        # 记录最高准确率未更新但发生回退的次数
                        self.rollback_count += 1
                        # 避免过度回退
                        self.rollback_count = min(self.rollback_count, 5)
                        # 如果找不到合适的路径，则清除上一层网络并重新寻找
                        print(f'清除上{self.rollback_count}层网络')
                        for k in range(self.rollback_count):
                            if self.history_method_inputs:  # 检查是否有历史输入
                                self.history_method_inputs.pop()
                            if self.history_method_input_values:  # 检查是否有历史输入值
                                self.history_method_input_values.pop()
                                
                            if len(self.history_values) > 1:
                                self.history_values.pop()
                                self.paths.pop()
                            self.last_values = self.history_values[-1]

            # 打开文件并写入路径数据
            self.save_model()

        except KeyboardInterrupt:
            print("\n检测到中断，正在导出路径数据...")
            self.save_model()
            print(f"已导出 {len(self.paths)} 层路径到 output.txt。训练结束。")
        except Exception as e:
            exception_details = traceback.format_exc()
            print(f"\n发生异常: {str(e)}，正在导出路径数据...")
            with open("error_log.txt", "w", encoding="utf-8") as error_file:
                error_file.write(exception_details)
            self.save_model()
            print(f"已导出 {len(self.paths)} 层路径到 output.txt。训练结束。")


    def load_paths_from_file(self, file_path="output.txt"):
        """
        从指定的文件中加载路径数据到 self.paths 中。

        参数:
            file_path (str): 包含路径数据的文件路径，默认是 "output.txt"。
        返回:
            None
        """
        self.paths = [] # 清空self.paths
        try:
            with open(file_path, "r") as f:
                for line in f:
                    # 去除每行末尾的换行符并转为适当的数据类型
                    item = line.strip()
                    if item:  # 确保该行不为空
                        self.paths.append(eval(item))  # 使用 eval 转换字符串回原数据类型
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")
            raise
        except Exception as e:
            print(f"读取文件时发生错误: {str(e)}")

    # 评估函数
    def evaluate(self, inputs, targets):
        """
        评估模型在给定输入和目标上的准确性。

        该函数根据随机方法处理输入数据，计算模型预测值与目标值的匹配情况，并返回准确率。

        参数:
            inputs (np.ndarray): 输入数据，形状通常为 (样本数, 特征数)。
            targets (np.ndarray): 真实目标值，与输入数据一一对应。

        返回:
            float: 模型在给定输入数据上的准确率。
        """
        for path in self.paths:
            inputs = self.process_array_with_list(path, inputs)  # 根据随机方法处理数据

        collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=inputs)
        # 计算预测值与真实值匹配的情况
        prediction_matches = collapse_values == targets  
        # 计算准确率
        train_accuracy = np.mean(prediction_matches) 
        print(f"准确率：{train_accuracy}")
        return train_accuracy

    def inference(self, data):
        """
        执行推理过程，将输入数据依次通过多个路径进行处理，最终合并成一个推理结果。

        参数：
        data : 任意类型
            初始输入数据，将依次通过 `self.paths` 进行处理。

        过程：
        1. 遍历 `self.paths` 列表中的每个路径，并使用 `process_single_value_with_list` 处理数据。
        2. 经过所有路径处理后，使用 `collapse` 方法合并最终数据。
        3. 打印推理结果。

        输出：
        无直接返回值，但会在控制台打印最终的推理结果。
        """
        for path in self.paths:
            data = self.process_single_value_with_list(path, data)
        result = self.collapse(data)
        print(f"推理值：{result}")

