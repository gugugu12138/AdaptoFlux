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
        self.graph = nx.DiGraph()
        self.graph.add_node("root") 
        self.graph.add_node('collapse')
        for feature_index in range(self.values.shape[1]):  # 遍历特征维度
            self.graph.add_edge(
                "root",
                "collapse",
                data_coord=feature_index  # 或者用字符串名代替索引也可以
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

    def append_nx_layer(self, method_group_positions, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        :param method_group_positions: 由 process_random_method 返回的方法分组信息
        :param discard_unmatched: 如何处理未匹配的节点。可选值：
                                'ignore' - 忽略不处理；
                                'to_discard' - 将其链接到指定的节点。
        :param discard_node_name: 当 discard_unmatched == 'to_discard' 时，使用的节点函数名
        """
        self.layer += 1
        new_index_edge = 0
        method_counts = {method_name: 0 for method_name in self.methods}
        method_counts["unmatched"] = 0

        # 提前获取所有与 'collapse' 相连的边
        collapse_edges = list(self.graph.in_edges("collapse", data=True))

        # 遍历每个方法的分组
        for method_name, groups in method_group_positions.items():

            if method_name == "unmatched":
                continue  # 跳过废弃组，在后面统一处理

            for group in groups:
                # 创建新节点并添加属性
                new_target_node = f"{self.layer}_{method_counts[method_name]}_{method_name}"
                self.graph.add_node(new_target_node, method_name=method_name)

                # 遍历 collapse 相关的边，避免重复获取
                for u, v, data in collapse_edges:
                    if data.get('data_coord') in group:
                        # 删除原有的边
                        self.graph.remove_edge(u, v)
                        # 添加新的边，目标节点改为新的目标节点
                        self.graph.add_edge(u, new_target_node, **data)

                        # 添加从新节点到 v 的边
                        for i in range(self.methods[method_name]["output_count"]):
                            self.graph.add_edge(new_target_node, v, data_coord=new_index_edge)
                            new_index_edge += 1

                method_counts[method_name] += 1

        # 处理 unmatched 组（如果存在）
        unmatched_indices = method_group_positions.get("unmatched", [])
        if unmatched_indices:
            if discard_unmatched == 'to_discard':
                # 为每个废弃索引创建独立节点，统一使用 unmatched_method_name
                for groups in unmatched_indices:
                    for group in groups:
                        node_name = f"{self.layer}_{method_counts["unmatched"]}_unmatched"
                        self.graph.add_node(node_name, method_name=discard_node_method_name)

                        # 遍历 collapse 边，将 data_coord 匹配的边连接到这个新节点
                        for u, v, data in collapse_edges:
                            if data.get('data_coord') in group:
                                self.graph.remove_edge(u, v)
                                self.graph.add_edge(u, node_name, **data)

                                # 从这个节点再连接回 collapse（可以控制输出数量）
                                for _ in range(1):  # 默认输出1条边
                                    self.graph.add_edge(node_name, v, data_coord=new_index_edge)
                                    new_index_edge += 1

                    method_counts["unmatched"] += 1

            elif discard_unmatched == 'ignore':
                # 不做任何处理，直接跳过
                pass

            else:
                raise ValueError(f"未知的 discard_unmatched 值：{discard_unmatched}。支持的选项为 'ignore' 或 'to_discard'")



        


        




    # 待修改
    # 根据输入的列表更改数组,处于神经待处理队列状态的值采用暂时不处理方案, 还有一种方案是先将这些值置于最后
    # 使用该方法会导致数据收敛到一定数量级时可能出现所有数据都在待处理队列中导致数组全为空的情况发生
    # 这种方法坍缩时不会将待处理数据加入计算
    def process_array_with_list(self, method_list, values=None, max_values_multipie=5):
        """
        根据输入的 method_list 处理数组 values。
        - 处于待处理队列的值采用暂时不处理方案（跳过计算）。
        - 另一种方案是将这些值暂存，并在满足输入要求后一起计算。
        - 该方法可能导致数据收敛到一定数量级时所有数据都在待处理队列中，导致数组全为空。
        
        参数：
            method_list (list): 处理方法的列表，部分方法可能以 '-' 开头表示跳过计算。
            max_values_multipie (int): 预估最大扩展倍数，用于预分配存储空间。
        
        返回值：
            np.ndarray: 处理后的新数组。
        """
        if values is None:
            values = self.last_values  # 使用默认值

        try:
            # 预分配一个较大的数组用于存储计算结果，避免动态扩展带来的性能开销
            new_values = np.empty((values.shape[0], max_values_multipie * len(method_list)))
            
            for j in range(values.shape[0]):  # 遍历每一行数据
                new_number = 0  # 记录新数组的当前存储位置
                
                for i in range(len(method_list)):
                    method_name = method_list[i]
                    
                    if method_name.startswith('-'):
                        # 以 '-' 开头的方法不处理，直接存入待处理队列
                        self.method_input_values[method_name.lstrip('-')].append(values[j, i])
                        continue
                    
                    method_info = self.methods[method_name]
                    
                    if method_info['input_count'] == 1:
                        # 处理单输入的函数，直接调用并存储输出
                        for k in method_info['function'](values[j, i]):
                            new_values[j, new_number] = k
                            new_number += 1
                    else:
                        # 处理多输入的函数，先存入待处理队列，达到输入要求后再计算
                        self.method_input_values[method_name].append(values[j, i])
                        
                        if len(self.method_input_values[method_name]) == method_info['input_count']:
                            # 当待处理数据量满足要求时，调用函数进行计算
                            for k in method_info['function'](*self.method_input_values[method_name]):
                                new_values[j, new_number] = k
                                new_number += 1
                            
                            # 清空已使用的输入值
                            self.method_input_values[method_name] = []
                
            # 截取有效数据，去除未使用的预分配部分
            new_values = new_values[:, :new_number]  
            
            return new_values
        
        except Exception as e:
            print("出现错误，返回原数组：", str(e))
            traceback.print_exc()
            print(self.last_values.shape)
            return self.last_values

    def process_single_value_with_list(self, method_list, value):
        """
        根据输入的 method_list 处理单个值（仅一个数据元素）。
        - 处于待处理队列的值采用暂时不处理方案（跳过计算）。
        - 另一种方案是将这些值暂存，并在满足输入要求后一起计算。

        参数：
            method_list (list): 处理方法的列表，部分方法可能以 '-' 开头表示跳过计算。
            value (单一值): 需要处理的单个数据值。

        返回值：
            处理后的计算结果。
        """
        # 初始化存储计算结果
        result = []

        for method_name in method_list:
            if method_name.startswith('-'):
                # 以 '-' 开头的方法不处理，跳过
                self.method_input_values[method_name.lstrip('-')].append(value)
                continue

            method_info = self.methods[method_name]

            if method_info['input_count'] == 1:
                # 处理单输入的函数，直接调用并存储输出
                for k in method_info['function'](value):
                    result.append(k)
            else:
                # 处理多输入的函数，将当前值暂存，等待满足输入要求后再计算
                self.method_input_values[method_name].append(value)

                if len(self.method_input_values[method_name]) == method_info['input_count']:
                    # 当待处理数据量满足要求时，调用函数进行计算
                    for k in method_info['function'](*self.method_input_values[method_name]):
                        result.append(k)

                    # 清空已使用的输入值
                    self.method_input_values[method_name] = []

        # 返回计算结果
        return result

    def save_model(self, folder="models"):
        """
        保存模型的路径数据和相关文件。

        1. 检查并确保目标文件夹存在。如果文件夹不存在，则创建文件夹。
        2. 将模型的路径信息写入到指定的文件中（默认为 output.txt）。
        3. 复制指定的 `methods_path` 文件到目标文件夹，确保源文件存在。

        参数:
            folder (str): 用于保存模型的文件夹路径，默认为 "models"。
        """
        # 确保文件夹存在，如果不存在则创建
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 构建文件路径
        file_path = os.path.join(folder, "output.txt")

        # 打开文件并写入路径数据
        with open(file_path, "w") as f:
            for item in self.paths:
                f.write(str(item) + "\n")

        # 复制 methods_path 文件到保存的文件夹
        if os.path.exists(self.methods_path):  # 确保源文件存在
            shutil.copy(self.methods_path, folder)  # 复制文件到目标文件夹


    def get_path_entropy(self, paths):
        """
        计算路径熵

        :param paths: 需要计算的二维列表路径数据
        :return: 计算得到的路径熵值
        """
        try:
            # 如果 paths 为空，或者所有子列表都为空，直接返回 0
            if not paths or all(len(row) == 0 for row in paths):
                return 0
            
            if not paths or not isinstance(paths, list) or not all(isinstance(row, list) for row in paths):
                raise ValueError("输入的 paths 必须是一个二维列表")
            
            # 将所有元素转换为整数并展平为一维列表
            data = [abs(int(x)) for row in paths for x in row]

            if not data:  # 确保数据不为空
                return 0

            # 计算概率分布
            counts = Counter(data)
            total = len(data)
            probabilities = [count / total for count in counts.values()]

            # 计算熵
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)
        except ValueError as ve:
            print(f"值错误: {ve}")
            print("输入的路径数据:", paths)  # 打印输入的数据，帮助调试
            return None
        except Exception as e:
            print(f"计算路径熵时发生错误: {e}")
            print("路径数据:", paths)  # 打印路径数据以便定位问题
            return None
        
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

