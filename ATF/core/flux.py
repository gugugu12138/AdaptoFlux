import random
from enum import Enum
import uuid
import numpy as np
import inspect
import importlib.util
import math
import traceback
from threading import Thread, Event
from collections import Counter
import os
import shutil
import networkx as nx
import pandas as pd
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod
from ..GraphManager.graph_processor import GraphProcessor
from ..PathGenerator.path_generator import PathGenerator
from ..ModelTrainer.model_trainer import ModelTrainer


class AdaptoFlux:
    def __init__(self, values = None, labels = None, methods_path = None, collapse_method=CollapseMethod.SUM):
        """
        初始化 AdaptoFlux 类的实例
        
        :param values: 一维数据值列表
        :param labels: 每个值对应的标签列表
        :param collapse_method: 选择的坍缩方法，默认为 SUM
        :param methods_path: 存储方法路径的文件路径，默认为 "methods.py"
        """
        if values is not None and labels is not None:
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

        # 获取每个特征维度的数据类型
        if isinstance(self.values, pd.DataFrame):
            self.feature_types = [str(dtype) for dtype in self.values.dtypes]
        elif self.values.dtype.names is not None:  # structured array
            self.feature_types = [self.values.dtype.fields[name][0].name for name in self.values.dtype.names]
        else:
            self.feature_types = [str(self.values.dtype)] * self.values.shape[1]

        if self.values is not None:
            # 添加 root -> collapse 边
            for feature_index, feature_type in enumerate(self.feature_types):
                self.graph.add_edge(
                    "root",
                    "collapse",
                    output_index=feature_index,
                    data_coord=feature_index,
                    data_type=feature_type
                )
            self.layer = 0
        
        # # 选择的坍缩方法
        # self.collapse_method = collapse_method  # 默认使用 SUM 方法
        # self.custom_collapse_function = None  # 预定义自定义坍缩方法，默认值为 None
        
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

        # 初始化图处理器
        self.graph_processor = GraphProcessor(
            graph=self.graph,
            methods=self.methods,
            collapse_method=collapse_method
        )

        # 自动导入方法
        if self.methods_path is not None:
            self.import_methods_from_file(self.methods_path)
        
        self.model_trainer = ModelTrainer(adaptoflux_instance=self)

    def get_input_dimension(self):
        """
        获取输入维度（特征数量）
        """
        if self.values is not None:
            return self.values.shape[1]
        
        # 如果没有 values，尝试从图结构中获取
        if "root" in self.graph:
            return len(list(self.graph.out_edges("root")))
        
        return 0  # 默认值

    
    def add_collapse_method(self, collapse_function):
        """
        允许用户自定义坍缩方法，并替换现有的坍缩方法
        :param collapse_function: 传入一个函数
        """
        self.graph_processor.collapse_manager.set_custom_collapse(collapse_function)

    # 添加处理方法到字典
    def add_method(self, method_name, method, input_count, output_count, input_types, output_types, group, weight, vectorized):
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
            "input_types" : input_types,
            "output_types" : output_types,
            "group" : group,
            "weight" : weight,
            "vectorized": vectorized
        }
        self.method_inputs[method_name] = []
        self.method_input_values[method_name] = [] # 疑似没有必要

    def import_methods_from_file(self, path=None):
        """
        从指定文件中导入所有方法并添加到方法字典中。
        :param file_path: Python 文件路径
        """

        import_path = path or self.methods_path
        if import_path is None:
            print("没有指定 methods 文件路径，跳过导入")
            return

        # 动态加载模块
        spec = importlib.util.spec_from_file_location("module.name", self.methods_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 遍历模块中的所有成员
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):  # 检查是否为函数
                if getattr(obj, "is_decorator", False):
                    continue
                # 获取函数所需的参数数量
                input_count = len(inspect.signature(obj).parameters)
                output_count = getattr(obj, "output_count", 1)
                input_types = getattr(obj, "input_types", None)
                output_types = getattr(obj, "output_types", None)
                group = getattr(obj, "group", "default")
                weight = getattr(obj, "weight", 1.0)
                vectorized = getattr(obj, "vectorized", False)
                self.add_method(name, obj, input_count, output_count, input_types, output_types, group, weight, vectorized)
        # 记录初始状态
        self.history_method_inputs.append(self.method_inputs)
        self.history_method_input_values.append(self.method_input_values)
        # self.history_method_input_val_values.append(self.method_input_val_values)
        print(self.methods)

    def build_function_pool_by_input_type(self):
        """
        根据 self.methods 中各方法的 input_types 字段，
        构建一个字典：输入类型 -> 可用方法列表（包含完整方法信息）
        """
        function_pool_map = {}

        for method_name, method_info in self.methods.items():
            input_types = method_info.get("input_types", [])

            # 如果没有指定 input_types，默认不归类（或可设为通用类型）
            if not input_types:
                continue

            for input_type in input_types:
                if input_type not in function_pool_map:
                    function_pool_map[input_type] = []

                # 将该方法加入对应输入类型的函数池中
                function_pool_map[input_type].append((method_name,method_info))

        return function_pool_map

    # 根据当前选择的坍缩方法，对输入值进行计算
    def collapse(self, values):
        """
        根据设定的坍缩方法（collapse_method）对输入值进行聚合计算。

        :param values: 需要进行坍缩计算的数值列表
        :return: 计算后的坍缩值
        :raises ValueError: 如果指定的坍缩方法未知，则抛出异常
        """
        return self.graph_processor.collapse_manager.collapse(values)

    # 生成单次路径选择
    def process_random_method(self, shuffle_indices=True):
        """
        生成一个随机的方法选择路径，并确保多输入方法的输入索引匹配。

        :param shuffle_indices: 是否打乱多输入方法的索引，默认为 True
        :return: method_list，包含为每个元素随机分配的方法（可能是单输入或多输入方法）
        :raises ValueError: 如果方法字典为空或值列表为空，则抛出异常
        """
        self.path_generator = PathGenerator(
            graph=self.graph,
            methods=self.methods,
        )
        return self.path_generator.process_random_method(shuffle_indices=shuffle_indices)

    
    def replace_random_elements(self, result, n, shuffle_indices=True):
        """
        替换部分索引对应的方法，并重新分组以符合 input_count 要求。
        
        :param result: 来自 process_random_method 的输出结构
        :param n: 要随机替换的索引数量
        :param shuffle_indices: 是否打乱新方法分配时的索引顺序
        :return: 新的 result 结构，包含更新后的 index_map、valid_groups 和 unmatched
        """
        self.path_generator = PathGenerator(
            graph=self.graph,
            methods=self.methods,
        )
        return self.path_generator.replace_random_elements(result, n, shuffle_indices=shuffle_indices)

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

        self.graph_processor.append_nx_layer(result, discard_unmatched, discard_node_method_name)
        self.graph = self.graph_processor.graph  # 同步图结构
        self.layer = self.graph_processor.layer  # 同步层数

    def remove_last_nx_layer(self):
        """
        删除图中最后一层节点，并将它们的输入直接连接回 collapse，
        实现“撤回”一层的操作。
        """

        self.graph_processor.remove_last_nx_layer()
        self.graph = self.graph_processor.graph  # 同步图结构
        self.layer = self.graph_processor.layer  # 同步层数

    def infer_with_graph(self, values):
        """
        使用图结构对输入 values 进行推理计算，支持多输入/多输出操作。
        
        参数:
            values (np.ndarray): 输入数据，形状为 [样本数, 特征维度]
        
        返回:
            np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        return self.graph_processor.infer_with_graph(values)

    def infer_with_graph_single(self, sample):
        """
        使用图结构对单个样本进行推理计算。
        
        参数:
            sample (np.ndarray or list): 单个样本，形状为 [特征维度]
        
        返回:
            float or np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        return self.graph_processor.infer_with_graph_single(sample)

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
            data = json_graph.node_link_data(self.graph, edges="edges")  # 转换为可序列化的字典
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
    
    def generate_models(self, num_models=5, max_layers=3):
        """
        生成多个随机图模型，并计算它们的准确率。

        参数:
            num_models (int): 要生成的模型数量
            max_layers (int): 每个模型的最大层数

        返回:
            List[Dict]: 包含模型及其准确率的字典列表，格式为 [{"model": graph, "accuracy": acc}, ...]
        """
        model_candidates = []

        for _ in range(num_models):
            # 创建一个新的临时图结构
            temp_graph = nx.MultiDiGraph()
            temp_graph.add_node("root")
            temp_graph.add_node("collapse")

            # 添加初始边（从 root 到 collapse）
            for feature_index in range(self.values.shape[1]):
                temp_graph.add_edge(
                    "root",
                    "collapse",
                    output_index=0,
                    data_coord=feature_index
                )

            layer = 0  # 局部层数计数器

            # 添加多层随机节点
            for _ in range(max_layers):
                result = self.process_random_method()
                self.append_nx_layer_to_graph(temp_graph, result, layer)
                layer += 1

            # 推理并计算准确率
            predictions = self.infer_with_graph_custom(temp_graph, self.values)
            accuracy = np.mean(predictions == self.labels)

            # 存储模型及准确率
            model_candidates.append({
                "model": temp_graph,
                "accuracy": accuracy
            })

        return model_candidates

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

    def _create_graph_processor(self, graph):
        """
        工厂方法：创建一个图处理器（GraphProcessor）实例。

        该方法根据传入的图结构和当前 AdaptoFlux 的配置（如可用方法池、聚合方式），
        返回一个已初始化的 GraphProcessor 对象，用于后续图结构的扩展和推理操作。

        Parameters:
        -----------
        graph : nx.MultiDiGraph
            初始图结构，通常是一个包含 "root" 和 "collapse" 节点的基础图

        Returns:
        --------
        GraphProcessor
            一个已经配置好的图处理器对象，可用于：
            - 向图中添加新节点/层（append_nx_layer）
            - 使用图结构进行推理（infer_with_graph）
            - 图结构的优化与分析等操作
        """
        return GraphProcessor(
            graph=graph,
            methods=self.methods,              # 可用的方法集合（如加法、乘法、激活函数等）
            collapse_method=self.collapse_method  # 输出聚合方式（如 sum、mean 等）
        )

