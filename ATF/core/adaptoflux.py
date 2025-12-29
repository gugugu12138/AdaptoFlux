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
import copy
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod
from ..GraphProcessor.graph_processor import GraphProcessor
from ..PathGenerator.path_generator import PathGenerator
from ..ModelTrainer.model_trainer import ModelTrainer


class AdaptoFlux:
    def __init__(
        self,
        values=None,
        labels=None,
        methods_path=None,
        collapse_method=CollapseMethod.SUM,
        type_equivalence_map=None,
        input_types_list=None  # ← 新增参数
    ):
        """
        初始化 AdaptoFlux 类的实例
        
        :param values: 一维数据值列表
        :param labels: 每个值对应的标签列表
        :param collapse_method: 选择的坍缩方法，默认为 SUM
        :param methods_path: 存储方法路径的文件路径，默认为 "methods.py"
        """
        if values is not None and labels is not None:
            # 1. labels 必须是一维
            if hasattr(labels, 'shape'):
                assert len(labels.shape) == 1, "labels 必须是一维"
            else:
                labels = np.asarray(labels)
                assert labels.ndim == 1, "labels 转为 array 后必须是一维"

            # 2. values 必须支持“按样本索引”（即 values[i] 有效）
            if hasattr(values, '__getitem__') and hasattr(values, '__len__'):
                num_samples = len(values)
                assert num_samples == len(labels), "样本数量不一致"
                # ✅ 允许 values 是 list[dict], list[np.ndarray], Dataset 等
            else:
                raise ValueError("values 必须是可索引的批量容器（如 list, np.ndarray, Dataset）")

        # 3. 如果是 array-like，检查是否至少一维
        if hasattr(values, 'shape'):
            assert len(values.shape) >= 1, "values 至少应有一维（样本维）"

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

        # 存储路径信息
        # self.paths = []  # 记录每个值对应的路径
        # self.max_probability_path = [] # 记录最高概率对应的路径

        # 获取每个特征维度的数据类型
        if input_types_list is not None:
            # 用户显式提供了类型列表
            if self.values is not None:
                # 校验长度一致性
                expected_dim = self.values.shape[1] if hasattr(self.values, 'shape') else len(self.values[0])
                assert len(input_types_list) == expected_dim, \
                    f"input_types_list 长度 ({len(input_types_list)}) 与特征维度 ({expected_dim}) 不匹配"
            feature_types = list(input_types_list)
        else:
            # 原有自动推断逻辑（保持兼容）
            feature_types = []
            if self.values is not None:
                if isinstance(self.values, pd.DataFrame):
                    feature_types = [str(dtype) for dtype in self.values.dtypes]
                elif self.values.dtype.names is not None:
                    feature_types = [self.values.dtype.fields[name][0].name for name in self.values.dtype.names]
                else:
                    if len(self.values.shape) != 2:
                        raise ValueError(...)
                    if self.values.shape[1] == 0:
                        raise ValueError(...)
                    feature_types = [str(self.values.dtype)] * self.values.shape[1]

            if len(feature_types) == 0:
                raise ValueError("无法推断特征类型，且未提供 input_types_list")

        # 构建初始图结构
        graph = nx.MultiDiGraph()
        graph.add_node("root") 
        graph.add_node('collapse')

        # 添加 root -> collapse 边
        for feature_index, feature_type in enumerate(feature_types):
            graph.add_edge(
                "root",
                "collapse",
                output_index=feature_index,
                data_coord=feature_index,
                data_type=feature_type,
                input_slot=feature_index
            )

        # 自动导入方法
        if self.methods_path is not None:
            self.import_methods_from_file(self.methods_path)

        # 初始化图处理器
        self.graph_processor = GraphProcessor(
            graph=graph,
            methods=self.methods,
            collapse_method=collapse_method
        )
        
        self.path_generator = PathGenerator(
            graph=self.graph_processor.graph,
            methods=self.methods,
            type_equivalence_map=type_equivalence_map
        )

    @property
    def graph(self):
        """提供图的访问接口"""
        return self.graph_processor.graph

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

    
    def set_custom_collapse(self, collapse_function):
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
                if getattr(obj, "is_internal_decorator", False):
                    continue
                # 获取函数所需的参数数量
                input_count = len(inspect.signature(obj).parameters)
                if input_count <= 0:
                    raise ValueError(f"方法 {name} 的输入参数数量无效: {input_count}")
                output_count = getattr(obj, "output_count", 1)
                input_types = getattr(obj, "input_types", None)
                if len(input_types) == 0:
                    raise ValueError(f"方法 {name} 的 input_types 不能为空")
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
        self.path_generator.graph = self.graph_processor.graph

        return self.path_generator.process_random_method(shuffle_indices=shuffle_indices)

    
    def replace_random_elements(self, result, n, shuffle_indices=True):
        """
        替换部分索引对应的方法，并重新分组以符合 input_count 要求。
        
        :param result: 来自 process_random_method 的输出结构
        :param n: 要随机替换的索引数量
        :param shuffle_indices: 是否打乱新方法分配时的索引顺序
        :return: 新的 result 结构，包含更新后的 index_map、valid_groups 和 unmatched
        """
        self.path_generator.graph = self.graph_processor.graph

        return self.path_generator.replace_random_elements(result, n, shuffle_indices=shuffle_indices)

    def append_nx_layer(self, result, discard_unmatched='to_discard', discard_node_method_name="null"):
        """
        将给定的结果转换为图中的一层新节点。
        
        参数:
            result: 一个字典，包含以下键：
                - 'valid_groups': 按照方法名分组的有效数据索引列表
                - 'unmatched': 无法匹配成完整输入的剩余索引
            discard_unmatched: 如何处理未匹配项。可选值：
                'to_discard' - 创建单独节点并连接回 collapse，数据仍然保留在图中参与运算
                'ignore' - 忽略这些未匹配项，不会进入 collapse
            discard_node_method_name: 如果 discard_unmatched == 'to_discard'，则用该名称作为丢弃节点的 method_name
        """
        
        self.graph_processor.append_nx_layer(result, discard_unmatched, discard_node_method_name)

    def remove_last_nx_layer(self):
        """
        删除图中最后一层节点，并将它们的输入直接连接回 collapse，
        实现“撤回”一层的操作。
        """

        self.graph_processor.remove_last_nx_layer()

    def infer_with_graph(self, values):
        """
        使用图结构对输入 values 进行推理计算，支持多输入/多输出操作。
        
        参数:
            values (np.ndarray): 输入数据，形状为 [样本数, 特征维度]
        
        返回:
            np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        return self.graph_processor.infer_with_graph(values)

    def infer_with_graph_single(self, sample, use_pipeline=False, num_workers=4):
        """
        使用图结构对单个样本进行推理计算。
        
        参数:
            sample (np.ndarray or list): 单个样本，形状为 [特征维度]
        
        返回:
            float or np.ndarray: 经过图结构处理后的结果（通过 collapse 输出）
        """
        return self.graph_processor.infer_with_graph_single(sample, use_pipeline=False, num_workers=4)

    def infer_with_task_parallel(self, values, num_workers=4):

        return self.graph_processor.infer_with_task_parallel(values, num_workers=num_workers)


    def save_model(self, folder="models"):
        """
        保存模型的路径数据、图结构（.gexf）和相关文件。
        
        1. 确保目标文件夹存在，如果不存在则创建。
        3. 将图结构保存为 graph.gexf（可读性强）。
        4. 复制指定的 methods_path 文件到目标文件夹。
        
        参数:
            folder (str): 用于保存模型的文件夹路径，默认为 "models"。
        """
        import json
        from networkx.readwrite import json_graph

        # 确保文件夹存在，如果不存在则创建
        if self.methods_path and os.path.exists(self.methods_path):
            os.makedirs(folder)

        # 保存图结构到 graph.gexf（可读性强）
        gexf_file_path = os.path.join(folder, "graph.gexf")
        nx.write_gexf(self.graph_processor.graph, gexf_file_path)

        json_file_path = os.path.join(folder, "graph.json")
        try:
            data = json_graph.node_link_data(self.graph_processor.graph, edges="edges")  # 转换为可序列化的字典
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存 JSON 文件时出错: {e}")

        # 复制 methods_path 文件到保存的文件夹
        if os.path.exists(self.methods_path):  # 确保源文件存在
            shutil.copy(self.methods_path, folder)  # 复制文件到目标文件夹

    def load_model(self, folder="models"):
        """
        从指定文件夹加载保存的模型：图结构（.gexf 或 .json）文件。

        1. 检查目标文件夹是否存在。
        2. 优先从 graph.gexf 加载图结构，若不存在则尝试 graph.json。
        
        参数:
            folder (str): 包含模型文件的文件夹路径，默认为 "models"。
        """
        import os
        import json
        import networkx as nx
        from networkx.readwrite import json_graph

        if not os.path.exists(folder):
            raise FileNotFoundError(f"模型文件夹不存在: {folder}")

        self.graph_processor.graph = None

        # 尝试优先加载 .gexf 文件
        gexf_file_path = os.path.join(folder, "graph.gexf")
        json_file_path = os.path.join(folder, "graph.json")

        if os.path.exists(gexf_file_path):
            try:
                self.graph_processor.set_graph(nx.read_gexf(gexf_file_path))
                print("成功从 graph.gexf 加载图结构。")
            except Exception as e:
                print(f"读取 graph.gexf 失败: {e}")
        elif os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.graph_processor.set_graph(json_graph.node_link_graph(data, edges="edges"))
                print("成功从 graph.json 加载图结构。")
            except Exception as e:
                print(f"读取 graph.json 失败: {e}")
        else:
            raise FileNotFoundError("未找到 graph.gexf 或 graph.json 文件。")


        # 确保图被成功加载
        if self.graph_processor.graph is None:
            raise RuntimeError("图结构加载失败。")

        print(f"模型已成功从 '{folder}' 加载。")
        
    def get_graph_entropy(self):
        """
        计算图结构的熵值，基于节点和方法类型的分布。
        示例计算方法，可根据实际需求替换。
        :return: 计算得到的图结构熵值
        """

        return self.graph_processor.get_graph_entropy()
    
    def get_method_counter(self):
        """
        统计图中各 method_name 的出现次数
        """

        return self.graph_processor.get_method_counter()
    

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
        # --- 修改开始 ---
        # 获取旧的 graph_processor 的 collapse_manager
        old_collapse_manager = self.graph_processor.collapse_manager
        
        if old_collapse_manager.collapse_method == CollapseMethod.CUSTOM:
            # 创建新的 GraphProcessor 实例
            gp = GraphProcessor(
                graph=graph,
                methods=self.methods,              # 可用的方法集合（如加法、乘法、激活函数等）
            )
            
            gp.collapse_manager.set_custom_collapse(old_collapse_manager.custom_function)
        else:
            gp = GraphProcessor(
                graph=graph,
                methods=self.methods,              # 可用的方法集合（如加法、乘法、激活函数等）
                collapse_method=old_collapse_manager.collapse_method  # 继承旧的坍缩方法
            )
            
        return gp

    def clone(self):
        """
        该方法未作测试，谨慎使用！
        创建当前 AdaptoFlux 实例的深拷贝副本。
        
        该方法确保：
        - 图结构（NetworkX MultiDiGraph）被完整复制
        - 方法字典（self.methods）中的函数引用被保留（函数本身不可 deep copy，但引用安全）
        - 所有数据（values, labels）、历史记录、配置参数均被深拷贝
        - graph_processor 和 path_generator 等内部组件被重新初始化以绑定新图和方法池
        
        返回:
            AdaptoFlux: 一个与当前实例完全独立的新实例
        """
        # 创建新实例（不调用 __init__，避免重复初始化）
        cloned = AdaptoFlux.__new__(AdaptoFlux)

        # 深拷贝所有基础属性（除函数和图处理器外）
        for attr, value in self.__dict__.items():
            if attr in {'graph_processor', 'path_generator', 'model_trainer'}:
                continue  # 跳过，后面单独处理
            if attr == 'methods':
                # methods 中的 'function' 是函数对象，不能 deepcopy，但可以浅拷贝引用（安全）
                cloned.methods = {}
                for name, info in value.items():
                    cloned.methods[name] = copy.copy(info)  # 浅拷贝 dict，保留 function 引用
            elif attr == 'graph':
                # NetworkX 图必须用 copy.deepcopy 才能完全独立
                cloned.graph = copy.deepcopy(value)
            else:
                try:
                    setattr(cloned, attr, copy.deepcopy(value))
                except Exception as e:
                    # 对无法 deepcopy 的对象（如某些闭包），尝试浅拷贝
                    setattr(cloned, attr, copy.copy(value))

        # 重新初始化 graph_processor，绑定新图和新方法池
        cloned.graph_processor = GraphProcessor(
            graph=cloned.graph,
            methods=cloned.methods,
            collapse_method=cloned.graph_processor.collapse_manager.collapse_method
        )

        # 可选：重置 path_generator 和 model_trainer（它们通常是临时对象）
        cloned.path_generator = None
        cloned.model_trainer = ModelTrainer(adaptoflux_instance=cloned)

        return cloned

    def set_methods(self, new_methods: dict):
        """
        安全地更新方法池，并同步到所有子组件。
        """
        self.methods = new_methods
        # ✅ 关键：同步更新子组件
        if hasattr(self, 'graph_processor'):
            self.graph_processor.methods = new_methods
        if hasattr(self, 'path_generator'):
            self.path_generator.methods = new_methods