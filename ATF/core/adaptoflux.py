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
from ..GraphProcessor.execution_plan import (
    compile_execution_plan,
    effects_for_method,
    export_execution_plan,
    restore_execution_plan,
)
from ..PathGenerator.path_generator import PathGenerator
from ..ModelTrainer.model_trainer import ModelTrainer

# value 预期添加动态接口
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
                        raise ValueError("无法推断特征类型")
                    if self.values.shape[1] == 0:
                        raise ValueError("无法推断特征类型")
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
    def add_method(self, method_name, method, input_count, output_count, input_types, output_types, group, weight, vectorized, effects=None):
        """
        添加方法到字典
        :param method_name: 方法名称
        :param method: 方法本身
        :param input_count: 方法所需的输入值数量
        :param effects: 可选的显式副作用声明。优先使用这里的声明；否则从函数的
            method_profile 元数据读取。旧函数没有声明时登记为 unknown，运行器
            会采取保守顺序，不把未声明函数视为可自由并行的纯计算。

        增量注册用于建图或训练准备阶段，不应与正在执行的推理并发调用。
        替换完整方法池时使用 set_methods，由运行器加锁验证后再提交。
        """ 
        self.methods[method_name] = {
            "function" : method,
            "input_count" : input_count,
            "output_count" : output_count,
            "input_types" : input_types,
            "output_types" : output_types,
            "group" : group,
            "weight" : weight,
            "vectorized": vectorized,
            "effects": effects_for_method(
                {"function": method, **({"effects": effects} if effects is not None else {})}
            ),
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
        spec = importlib.util.spec_from_file_location("module.name", import_path)
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
        保存数据图和已经冻结的执行计划，保留状态访问的语义顺序。

        graph.json 内嵌完整计划，可以独立恢复；execution_plan.json 为伴随文件，
        用于 GEXF 恢复，因为 GEXF 不保留任意图级元数据。数据依赖和控制依赖
        分别保存，控制依赖不会被转换为方法的数据参数。

        本方法不序列化函数闭包、外部环境状态或自定义 collapse 函数。载入者仍需
        提供相同的方法实现、聚合配置和初始外部状态，才能复现相同运行结果。
        """
        import json
        from networkx.readwrite import json_graph

        self.graph_processor.prepare_execution_plan()
        plan = export_execution_plan(self.graph, self.methods)
        data = json_graph.node_link_data(self.graph, edges="edges")
        data["execution_plan"] = plan
        # 在真正写文件前先完成计划校验和 JSON 编码。类型/效果元数据不可序列化
        # 时直接报错，不像旧实现那样打印错误后继续，留下看似成功的不完整模型。
        graph_json = json.dumps(data, indent=2)
        plan_json = json.dumps(plan, indent=2)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "graph.json"), "w", encoding="utf-8") as f:
            f.write(graph_json)
        with open(os.path.join(folder, "execution_plan.json"), "w", encoding="utf-8") as f:
            f.write(plan_json)
        nx.write_gexf(self.graph, os.path.join(folder, "graph.gexf"))

        if self.methods_path and os.path.isfile(self.methods_path):
            destination = os.path.join(folder, os.path.basename(self.methods_path))
            if os.path.abspath(self.methods_path) != os.path.abspath(destination):
                shutil.copy(self.methods_path, destination)

    def load_model(self, path="models"):
        """
        从指定路径加载保存的模型：图结构（.gexf 或 .json）文件。

        支持两种用法：
        - 传入文件夹路径（如 "models"）：优先读取 graph.json，其次 graph.gexf。
        - 传入具体文件路径（如 "models/custom.gexf"）：直接加载该文件。

        参数:
            path (str): 文件夹路径或具体模型文件路径（.gexf 或 .json），默认为 "models"。

        当前方法库提供实际函数；方法签名及副作用声明必须与保存的计划一致。
        文件解析、端口检查、效果检查和无环检查全部成功后才替换当前模型。
        没有保存计划的旧模型在此次载入边界生成一次保守顺序，后续执行沿用它。
        """
        import os
        import json
        import networkx as nx
        from networkx.readwrite import json_graph

        if os.path.isdir(path):
            # JSON 是完整模型的首选来源。若它损坏或其计划不合法，必须报错；
            # 不能静默回退到可能有另一种节点顺序的 GEXF 后继续运行。
            candidates = [os.path.join(path, "graph.json"), os.path.join(path, "graph.gexf")]
            file_path = next((p for p in candidates if os.path.isfile(p)), None)
            if file_path is None:
                raise FileNotFoundError("在文件夹中未找到 graph.json 或 graph.gexf 文件。")
        elif os.path.isfile(path):
            file_path = os.fspath(path)
        else:
            raise FileNotFoundError(f"指定的路径既不是有效文件也不是目录: {path}")
        extension = os.path.splitext(file_path)[1].lower()
        embedded_plan = None
        if extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            embedded_plan = data.get("execution_plan")
            if "execution_plan" in data and not isinstance(embedded_plan, dict):
                raise ValueError("Invalid embedded execution plan.")
            candidate_graph = json_graph.node_link_graph(data, edges="edges")
        elif extension == ".gexf":
            candidate_graph = nx.MultiDiGraph(nx.read_gexf(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {extension}。仅支持 .gexf 或 .json。")

        sidecar_path = os.path.join(os.path.dirname(file_path), "execution_plan.json")
        sidecar_plan = None
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar_plan = json.load(f)
            if not isinstance(sidecar_plan, dict):
                raise ValueError("Invalid execution_plan.json.")
        if embedded_plan is not None and sidecar_plan is not None and embedded_plan != sidecar_plan:
            raise ValueError("Embedded execution plan does not match execution_plan.json.")
        # 两份计划都存在时要求完全一致，防止把旧侧车文件误配到新图。恢复函数
        # 还会核对稳定节点身份、数据端口、方法签名、效果声明及控制依赖。
        saved_plan = sidecar_plan if sidecar_plan is not None else embedded_plan
        if saved_plan is not None:
            restore_execution_plan(candidate_graph, self.methods, saved_plan)
        else:
            if any("_atf_uid" in d for _, d in candidate_graph.nodes(data=True)):
                raise ValueError("A model with persistent node UIDs requires a saved execution plan (embedded JSON or execution_plan.json).")
            # 旧文件第一次载入时选择并冻结保守顺序；这不是每次推理时根据当前
            # 节点插入顺序重新决定语义。JSON 和 GEXF 都采用相同规则：带新 UID
            # 却没有保存计划时直接拒绝，不能把损坏的新模型误认作旧模型重建顺序。
            compile_execution_plan(candidate_graph, self.methods)

        candidate_processor = self._create_graph_processor(candidate_graph)
        candidate_processor.prepare_execution_plan()
        # 先在独立候选图/运行器上完成所有校验，失败时原模型仍可继续使用。
        # 正式替换后同步 PathGenerator 的图引用，避免下一次训练增长旧图。
        self.graph_processor = candidate_processor
        if getattr(self, "path_generator", None) is None:
            self.path_generator = PathGenerator(graph=self.graph, methods=self.methods)
        else:
            self.path_generator.graph = self.graph
            self.path_generator.methods = self.methods
        print(f"模型已成功从 '{path}' 加载。")
        
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
        创建当前 AdaptoFlux 实例的深拷贝副本。
        
        该方法确保：
        - 图结构（NetworkX MultiDiGraph）被完整复制
        - 方法字典（self.methods）中的函数引用被保留（函数本身不可 deep copy，但引用安全）
        - 所有数据（values, labels）、历史记录、配置参数和执行计划均被深拷贝
        - graph_processor 和 path_generator 等内部组件被重新初始化以绑定新图和方法池
        - 方法函数引用保留；函数闭包引用的外部状态不会因此产生独立环境
        
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
                # 深拷贝效果声明等列表/字典；Python 函数仍保留同一个引用。
                # 所以模型副本有独立的图和计划，但不会自动获得独立的闭包环境。
                cloned.methods = copy.deepcopy(value)
            else:
                try:
                    setattr(cloned, attr, copy.deepcopy(value))
                except Exception as e:
                    # 对无法 deepcopy 的对象（如某些闭包），尝试浅拷贝
                    setattr(cloned, attr, copy.copy(value))

        # 重新初始化 graph_processor，绑定新图和新方法池
        self.graph_processor.prepare_execution_plan()
        cloned.graph_processor = GraphProcessor(
            graph=copy.deepcopy(self.graph),
            methods=cloned.methods,
            collapse_method=self.graph_processor.collapse_manager.collapse_method,
        )
        cloned.graph_processor.collapse_manager = copy.deepcopy(self.graph_processor.collapse_manager)
        cloned.graph_processor.prepare_execution_plan()

        if getattr(self, "path_generator", None) is not None:
            cloned.path_generator = copy.copy(self.path_generator)
            for name, value in self.path_generator.__dict__.items():
                if name not in {"graph", "methods"}:
                    setattr(cloned.path_generator, name, copy.deepcopy(value))
            cloned.path_generator.graph = cloned.graph
            cloned.path_generator.methods = cloned.methods
        else:
            cloned.path_generator = PathGenerator(graph=cloned.graph, methods=cloned.methods)
        # 训练会话的优化器状态不属于模型副本，后续需要训练时应重新绑定训练器。
        cloned.model_trainer = None

        return cloned

    def set_methods(self, new_methods: dict):
        """
        安全地更新方法池，并同步到所有子组件。
        """
        # 先规范化全部声明，任何条目非法时都不更改当前方法池；通过后再同步。
        # 手工构造的方法字典、装饰器元数据和旧的无声明方法使用同一解释规则。
        normalized = {name: {**info, "effects": effects_for_method(info)}
                      for name, info in new_methods.items()}
        # 先由运行器持有实例锁，在候选图上验证新方法池。若签名或效果声明不合法，
        # 此处直接抛错，self.methods 和 PathGenerator 仍保留已接受的旧方法池。
        # 不能直接赋值 graph_processor.methods，否则会绕过它的锁和验证边界。
        if hasattr(self, 'graph_processor'):
            self.graph_processor.set_methods(normalized)
        self.methods = normalized
        if getattr(self, 'path_generator', None) is not None:
            self.path_generator.methods = normalized
