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
from ..CollapseManager.collapse_functions import CollapseFunctionManager, CollapseMethod
from ..GraphManager.graph_processor import GraphProcessor
from ..PathGenerator.path_generator import PathGenerator


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
        if self.values is not None:
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

        # 初始化图处理器
        self.graph_processor = GraphProcessor(
            graph=self.graph,
            methods=self.methods,
            collapse_method=collapse_method
        )

    
    def add_collapse_method(self, collapse_function):
        """
        允许用户自定义坍缩方法，并替换现有的坍缩方法
        :param collapse_function: 传入一个函数
        """
        self.graph_processor.collapse_manager.set_custom_collapse(collapse_function)

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
        return self.collapse_manager.collapse(values)

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

                    
    # # 训练方法,epochs决定最终训练出来的模型层数,step用于控制重随机时每次增加几个重随机的指数上升速度 # 第一轮训练如果直接失败会出现错误，待解决
    # def training(self, epochs=20, depth_interval=1, depth_reverse=1, step=2, target_accuracy=None):
    #     """
    #     训练方法，用于训练模型，执行指定的轮次并在每一轮中根据训练集和验证集的表现进行调整。
        
    #     - 该方法会根据给定的 `epochs` 参数迭代指定次数，逐步调整模型的权重和输入方法。
    #     - 在每一轮迭代中，会计算当前模型的训练集准确率、方差以及与前一轮相比的变化。
    #     - 训练过程中可能会根据预设的条件重新调整方法的输入，尝试不同的路径来改进模型表现。
    #     - 如果本轮训练的方差较小或者准确率较高，则将当前模型保存到历史记录中；否则，会进行调整，重新计算适合的路径。
        
    #     参数：
    #         epochs (int): 训练的轮次，决定了模型训练的次数，默认为 10000。
    #         depth_interval (int): 控制深度的增量，默认为 1。
    #         depth_reverse (int): 控制深度的反向调整，默认为 1。
    #         step (int): 控制重随机时每次增加的指数上升速度，默认为 2。
        
    #     返回值：
    #         None: 该方法不返回任何值，而是直接修改类的状态并进行模型训练。
    #     """
    #     try:
    #         # 清空路径列表
    #         self.paths = []
    #         # 创建动态权重控制器
    #         dynamicWeightController = DynamicWeightController.DynamicWeightController(epochs)

    #         # 训练循环
    #         for i in range(1, epochs + 1):
    #             print("epoch:", i)
    #             last_method = self.process_random_method()  # 获取当前的随机方法
    #             new_last_values = self.process_array_with_list(last_method)  # 根据随机方法处理数据

    #             # 计算训练集方差和准确率
    #             last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
    #             new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
    #             last_vs_train = last_collapse_values == self.labels  # 计算训练集的相等情况
    #             new_vs_train = new_collapse_values == self.labels  # 计算新训练集的相等情况
                
    #             # 计算准确率和损失
    #             last_accuracy_trian = np.mean(last_vs_train)  # 计算上一轮训练集准确率
    #             new_accuracy_trian = np.mean(new_vs_train)  # 计算本轮训练集准确率
    #             last_loss_value = self.RMSE(self.labels, last_collapse_values)
    #             new_loss_value = self.RMSE(self.labels, new_collapse_values)

    #             print("上一轮训练集相等概率:" + str(last_accuracy_trian))
    #             print("本轮训练集相等概率：" + str(new_accuracy_trian))

    #             # 计算前后路径熵和从动态权重控制器获取权值
    #             last_path_entropy = self.get_path_entropy(self.paths)
    #             new_path_entropy = self.get_path_entropy(self.paths+ [last_method])


    #             last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
    #             new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

    #             # 计算指导值（暂未编写冗余部分）
    #             last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
    #             new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value

    #             print("上一轮训练集指导值:" + str(last_guiding_value))
    #             print("本轮训练集指导值：" + str(new_guiding_value))
                
    #             # 判断是否要更新网络路径
    #             if (last_guiding_value < new_guiding_value) and new_last_values.size != 0:
    #                 self.paths.append(last_method)
    #                 self.history_values.append(new_last_values)
    #                 self.last_values = new_last_values
    #                 self.history_method_inputs.append(self.method_inputs)
    #                 self.history_method_input_values.append(self.method_input_values)
    #                 self.metrics["accuracy"] = new_accuracy_trian
    #                 self.metrics["entropy"] = new_path_entropy

    #                 if self.metrics["accuracy"] > self.metrics["max_accuracy"]:
    #                     self.metrics["max_accuracy"] = self.metrics["accuracy"]

    #                 self.rollback_count = 0

    #                 if target_accuracy is not None and new_accuracy_trian >= target_accuracy:
    #                     print(f"训练提前停止，准确率达到 {new_accuracy_trian}，大于等于目标 {target_accuracy}")
    #                     # 保存模型或处理
    #                     self.save_model()  # 假设 save_model 方法已经定义
    #                     return
    #             else:
    #                 # 如果本轮训练不符合要求，重随机重新寻找合适的路径
    #                 j = 1
    #                 while j <= len(last_method):
    #                     print(f"在当层重新寻找适合的路径：当前重随机数{j}")
    #                     if self.history_method_inputs:  # 检查是否有历史输入
    #                         self.method_inputs = self.history_method_inputs[-1]

    #                     if self.history_method_input_values:  # 检查是否有历史输入值
    #                         self.method_input_values = self.history_method_input_values[-1]

    #                     last_method = self.replace_random_elements(last_method, j)  # 替换方法中的随机元素
    #                     j *= step  # 增加重随机的步长
                        
    #                     new_last_values = self.process_array_with_list(last_method)  # 处理新的数据

    #                     # 计算训练集方差和准确率
    #                     last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
    #                     new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
    #                     last_vs_train = last_collapse_values == self.labels  # 计算训练集的相等情况
    #                     new_vs_train = new_collapse_values == self.labels  # 计算新训练集的相等情况
                        
    #                     # 计算准确率和损失
    #                     last_accuracy_trian = np.mean(last_vs_train)  # 计算上一轮训练集准确率
    #                     new_accuracy_trian = np.mean(new_vs_train)  # 计算本轮训练集准确率
    #                     last_loss_value = self.RMSE(self.labels, last_collapse_values)
    #                     new_loss_value = self.RMSE(self.labels, new_collapse_values)

    #                     print("上一轮训练集相等概率:" + str(last_accuracy_trian))
    #                     print("本轮训练集相等概率：" + str(new_accuracy_trian))

    #                     # 计算前后路径熵和从动态权重控制器获取权值
    #                     last_path_entropy = self.get_path_entropy(self.paths)
    #                     new_path_entropy = self.get_path_entropy(self.paths+ [last_method])

    #                     last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
    #                     new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

    #                     # 计算指导值（暂未编写冗余部分）
    #                     last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
    #                     new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value

    #                     print("上一轮训练集指导值:" + str(last_guiding_value))
    #                     print("本轮训练集指导值：" + str(new_guiding_value))

    #                     # 判断是否需要更新路径
    #                     if (last_guiding_value < new_guiding_value) and new_last_values.size != 0:
    #                         self.paths.append(last_method)
    #                         self.history_values.append(new_last_values)
    #                         self.last_values = new_last_values
    #                         self.history_method_inputs.append(self.method_inputs)
    #                         self.history_method_input_values.append(self.method_input_values)
    #                         self.metrics["accuracy"] = new_accuracy_trian
    #                         self.metrics["entropy"] = new_path_entropy
                            
    #                         if self.metrics["accuracy"] > self.metrics["max_accuracy"]:
    #                             self.metrics["max_accuracy"] = self.metrics["accuracy"]
                            
    #                         self.rollback_count = 0

    #                         if target_accuracy is not None and new_accuracy_trian >= target_accuracy:
    #                             print(f"训练提前停止，准确率达到 {new_accuracy_trian}，大于等于目标 {target_accuracy}")
    #                             # 保存模型或处理
    #                             self.save_model()  # 假设 save_model 方法已经定义
    #                             return  
    #                         break

    #                 else:
    #                     # 记录最高准确率未更新但发生回退的次数
    #                     self.rollback_count += 1
    #                     # 避免过度回退
    #                     self.rollback_count = min(self.rollback_count, 5)
    #                     # 如果找不到合适的路径，则清除上一层网络并重新寻找
    #                     print(f'清除上{self.rollback_count}层网络')
    #                     for k in range(self.rollback_count):
    #                         if self.history_method_inputs:  # 检查是否有历史输入
    #                             self.history_method_inputs.pop()
    #                         if self.history_method_input_values:  # 检查是否有历史输入值
    #                             self.history_method_input_values.pop()
                                
    #                         if len(self.history_values) > 1:
    #                             self.history_values.pop()
    #                             self.paths.pop()
    #                         self.last_values = self.history_values[-1]

    #         # 打开文件并写入路径数据
    #         self.save_model()

    #     except KeyboardInterrupt:
    #         print("\n检测到中断，正在导出路径数据...")
    #         self.save_model()
    #         print(f"已导出 {len(self.paths)} 层路径到 output.txt。训练结束。")
    #     except Exception as e:
    #         exception_details = traceback.format_exc()
    #         print(f"\n发生异常: {str(e)}，正在导出路径数据...")
    #         with open("error_log.txt", "w", encoding="utf-8") as error_file:
    #             error_file.write(exception_details)
    #         self.save_model()
    #         print(f"已导出 {len(self.paths)} 层路径到 output.txt。训练结束。")


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

