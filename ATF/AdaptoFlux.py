import random
from enum import Enum
import uuid
import numpy as np
import inspect
import importlib.util
import math
import traceback
import DynamicWeightController
from threading import Thread, Event
from collections import Counter
import os
import shutil
from tqdm import tqdm
import json

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
        self.paths = []  # 记录每个值对应的路径
        self.max_probability_path = [] # 记录最高概率对应的路径

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
    def add_method(self, method_name, method, input_count=1):
        """
        添加方法到字典
        :param method_name: 方法名称
        :param method: 方法本身
        :param input_count: 方法所需的输入值数量
        """
        if not hasattr(self, '_method_counter'):
            self._method_counter = 1  # 初始化计数器
        method_id = str(self._method_counter)  # 为方法分配从1开始的整数 ID
        self.methods[method_id] = {
            "name": method_name,
            "function": method,
            "input_count": input_count,
        }
        # 初始化方法的预输入字典，初始值为空列表
        self.method_inputs[method_id] = []
        self.method_input_values[method_id] = []
        # self.method_input_val_values[method_id] = []
        self._method_counter += 1  # 增加计数器
    
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
                self.add_method(name, obj, input_count)
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

    def clear_method_inputs(self):
        """
        清空 self.method_inputs 字典中的旧索引数据，以确保后续计算的正确性。
        
        - 该方法不会改变字典的键（方法 ID），也不会修改列表的长度。
        - 仅将每个方法的输入索引列表中的元素置为 None，以避免上一轮计算的残留数据索引影响下一轮计算。
        - 确保在后续匹配输入索引时，方法能够正确获取新的输入数据，而不会因为旧数据干扰导致错误匹配。
        """
        for key, value_list in self.method_inputs.items():
            for i in range(len(value_list)):
                value_list[i] = None

    # # 生成单次路径选择
    # def process_random_method(self):
    #     """
    #     生成一个随机的方法选择路径，并确保多输入方法的输入索引匹配。

    #     :return: method_list，包含为每个元素随机分配的方法（可能是单输入或多输入方法）
    #     :raises ValueError: 如果方法字典为空或值列表为空，则抛出异常
    #     """
    #     if not self.methods:
    #         raise ValueError("方法字典为空，无法处理！")
    #     if self.last_values.size == 0:  # 处理 NumPy 数组的情况
    #         raise ValueError("值列表为空，无法处理！")

    #     # 存储每个元素对应的方法（单输入方法直接存储方法 ID，多输入方法存储占位符 "-method_id"）
    #     method_list = []

    #     # 获取 values 列数，即每个元素的数量
    #     num_elements = self.last_values.shape[1]

    #     # 随机选择方法并处理单输入和多输入方法
    #     for i in range(num_elements):
    #         method_id = random.choice(list(self.methods.keys()))  # 随机选择一个方法 ID
    #         method_data = self.methods[method_id]
    #         input_count = method_data["input_count"]  # 获取该方法需要的输入值数量

    #         if input_count == 1:
    #             # 如果方法只需要一个输入值，则直接存储方法 ID
    #             method_list.append(method_id)
    #         else:
    #             # 多输入方法使用占位符，以便后续匹配输入索引
    #             method_list.append(f"-{method_id}")

    #     # 处理多输入方法的输入索引匹配
    #     for index, i in enumerate(method_list):
    #         if i.startswith('-'):  # 识别多输入方法
    #             method_id = i.lstrip('-')  # 去除占位符前缀，获取真正的方法 ID
    #             method_data = self.methods[method_id]
    #             input_count = method_data["input_count"]  # 获取该方法所需的输入值数量

    #             # 记录方法所需的输入索引
    #             self.method_inputs[method_id].append(index)

    #             # 当方法的输入索引数达到所需数量时，进行替换
    #             if len(self.method_inputs[method_id]) == input_count:
    #                 for j in range(len(self.method_inputs[method_id])):
    #                     if self.method_inputs[method_id][j] is None:
    #                         # 忽略无效索引
    #                         continue
    #                     # 替换 method_list 中的占位符 # 这一行出现过报错，但是无法复现，很奇怪定位不了问题在哪
    #                     method_list[self.method_inputs[method_id][j]] = method_id
    #                 # 清空已处理的方法输入索引
    #                 self.method_inputs[method_id] = []
        
    #     self.clear_method_inputs()

    #     return method_list


    # 生成单次路径选择
    def process_random_method(self):
        """
        生成一个随机的方法选择路径，并确保多输入方法的输入索引匹配。

        :return: method_list，包含为每个元素随机分配的方法（可能是单输入或多输入方法）
        :raises ValueError: 如果方法字典为空或值列表为空，则抛出异常
        """
        if not self.methods:
            raise ValueError("方法字典为空，无法处理！")
        if self.last_values.size == 0:  # 处理 NumPy 数组的情况
            raise ValueError("值列表为空，无法处理！")

        method_list = []
        num_elements = self.last_values.shape[1]

        # 1. 初始为全部正值方法
        for _ in range(num_elements):
            method_id = random.choice(list(self.methods.keys()))
            method_list.append(method_id)

        # 2. 收集多输入方法的位置
        multi_input_positions = {}
        for idx, method_id in enumerate(method_list):
            if self.methods[method_id]["input_count"] > 1:
                multi_input_positions.setdefault(method_id, []).append(idx)
        
        # 3. 随机标记为负号占位
        for method_id, indices in multi_input_positions.items():
            input_count = self.methods[method_id]["input_count"]
            total_count = len(indices)
            remaining_count = total_count % input_count

            if remaining_count == 0:
                continue  # 刚好满足则跳过

            # 从索引中随机选出可组成完整组的部分
            selected_indices = random.sample(indices, remaining_count)
            for idx in selected_indices:
                method_list[idx] = f"-{method_id}"  # 替换为负号占位符

        return method_list


    # # 待修改
    # # 接受导向列表，返回有n个元素不同的新列表
    # def replace_random_elements(self, method_list, n):
    #     """
    #     替换列表中的 n 个元素为随机方法，并确保多输入方法的输入索引匹配。

    #     :param method_list: 需要修改的方法列表
    #     :param n: 要替换的元素数量
    #     :return: new_list，包含修改后的方法列表
    #     :raises ValueError: 如果 n 大于原列表长度，则抛出异常
    #     """
    #     if n > len(method_list):
    #         raise ValueError("n 不能大于原列表的长度")
        
    #     # 随机选择 n 个不重复的索引（此处的 n 变量未被正确使用，代码中原先只取了一个随机索引）
    #     indices_to_replace = random.sample(range(len(method_list)), n)
        
    #     # 复制原列表，以免修改原始数据
    #     new_list = method_list[:]
        
    #     # 遍历选中的索引，并用随机方法替换原有元素
    #     for idx in indices_to_replace:
    #         method_id = random.choice(list(self.methods.keys()))  # 随机选择一个方法 ID
    #         method_data = self.methods[method_id]  # 获取该方法对应的数据
    #         input_count = method_data["input_count"]  # 该方法需要的输入数量
            
    #         if input_count == 1:
    #             # 方法只需要一个输入值，直接记录方法 ID
    #             new_list[idx] = method_id
    #         else:
    #             # 方法需要多个输入值，在方法 ID 前添加 `-` 作为标记
    #             new_list[idx] = f"-{method_id}"
        
    #     # 遍历新列表，确保所有多输入方法（标记为 `-` 的方法）正确格式化
    #     for index, i in enumerate(new_list):
    #         if i.startswith('-'):
    #             continue  # 如果已经是 `-` 开头，则不修改
            
    #         input_count = self.methods[i]["input_count"]  # 获取输入数目
    #         if input_count > 1:
    #             new_list[index] = f"-{new_list[index]}"  # 标记多输入方法
        
    #     # 遍历列表，处理多输入方法的输入索引
    #     for index, i in enumerate(new_list):
    #         if i.startswith('-'):
    #             method_id = i.lstrip('-')  # 移除 `-` 以获取真正的 method_id
    #             method_data = self.methods[method_id]  # 获取方法数据
    #             input_count = method_data["input_count"]  # 获取输入数目
                
    #             # 在方法输入字典中添加索引
    #             self.method_inputs[method_id].append(index)
                
    #             # 如果该方法的输入已收集完毕（达到 `input_count` 数量）
    #             if len(self.method_inputs[method_id]) == input_count:
    #                 for j in range(len(self.method_inputs[method_id])):
    #                     if self.method_inputs[method_id][j] is None:
    #                         continue  # 跳过 None 值（原代码此处有误，之前的比较是错误的if self.method_inputs[method_id][j] >= len(new_list):）运行起来可以正常收敛
                        
    #                     # 将 `new_list` 中的占位符替换为真正的 method_id
    #                     new_list[self.method_inputs[method_id][j]] = method_id
                    
    #                 # 清空 method_inputs 以便下一轮使用
    #                 self.method_inputs[method_id] = []
        
    #     # 清理 method_inputs 以重置状态
    #     self.clear_method_inputs()
        
    #     return new_list

    def replace_random_elements(self, method_list, n):
        """
        替换列表中的 n 个元素为随机方法，并确保多输入方法的输入索引匹配。
        
        :param method_list: 原始方法列表
        :param n: 要替换的元素数量
        :return: new_list，包含修改后的方法列表
        :raises ValueError: 如果 n 大于原列表长度
        """
        if n > len(method_list):
            raise ValueError("n 不能大于原列表的长度")

        indices_to_replace = random.sample(range(len(method_list)), n)
        new_list = method_list[:]

        # 随机分配新方法
        for idx in indices_to_replace:
            method_id = random.choice(list(self.methods.keys()))
            new_list[idx] = method_id  # 先不加负号，统一在后面处理

        # 收集所有多输入方法的位置
        multi_input_positions = {}
        for idx, method_id in enumerate(new_list):
            if method_id.startswith('-'):
                method_id = method_id[1:]
                new_list[idx] = method_id
                # new_list[idx] = method_id.lstrip('-')
            input_count = self.methods[method_id]["input_count"]
            if input_count > 1:
                multi_input_positions.setdefault(method_id, []).append(idx)

        # 对每种多输入方法，标记多余的为负号占位符（确保只保留 input_count 的倍数）
        for method_id, indices in multi_input_positions.items():
            input_count = self.methods[method_id]["input_count"]
            total = len(indices)
            excess_count = total % input_count

            if excess_count == 0:
                continue  # 刚好满足则跳过

            excess_indices = random.sample(indices, excess_count)
            for idx in excess_indices:
                new_list[idx] = f"-{method_id}"  # 标记为占位

        return new_list


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
            # dynamicWeightController = DynamicWeightController.DynamicWeightController(epochs)

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


                # last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
                # new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

                # # 计算指导值（暂未编写冗余部分）
                # last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
                # new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value
                last_guiding_value = last_loss_value
                new_guiding_value = new_loss_value

                print("上一轮训练集指导值:" + str(last_guiding_value))
                print("本轮训练集指导值：" + str(new_guiding_value))
                
                # 判断是否要更新网络路径
                if (last_guiding_value > new_guiding_value) and new_last_values.size != 0:
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
                        # last_path_entropy = self.get_path_entropy(self.paths)
                        # new_path_entropy = self.get_path_entropy(self.paths+ [last_method])

                        # last_alpha,last_beta,last_gamma,last_delta = dynamicWeightController.get_weights(i - 1, last_path_entropy, last_loss_value)
                        # new_alpha,new_beta,new_gamma,new_delta = dynamicWeightController.get_weights(i, new_path_entropy, new_loss_value)

                        # 计算指导值（暂未编写冗余部分）
                        # last_guiding_value = last_alpha * last_accuracy_trian + last_beta * last_path_entropy - last_delta * last_loss_value
                        # new_guiding_value = new_alpha * new_accuracy_trian + new_beta * new_path_entropy - new_delta * new_loss_value
                        last_guiding_value = last_loss_value
                        new_guiding_value = new_loss_value

                        print("上一轮训练集指导值:" + str(last_guiding_value))
                        print("本轮训练集指导值：" + str(new_guiding_value))

                        # 判断是否需要更新路径
                        if (last_guiding_value > new_guiding_value) and new_last_values.size != 0:
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
    def evaluate(self, inputs, targets, save_errors_to="errors.json"):
        """
        评估模型在给定输入和目标上的准确性，并返回准确率和方差损失。
        同时可选地将预测错误的样本保存到 JSON 文件中。

        参数:
            inputs (np.ndarray): 输入数据，形状为 (样本数, 特征数)。
            targets (np.ndarray): 真实目标值。
            save_errors_to (str): 如果不为 None，则将预测错误的样本保存到该文件路径。

        返回:
            accuracy (float): 模型在给定输入数据上的准确率。
            loss (float): 预测值与目标值之间的方差（Variance），即损失。
        """
        # 处理路径
        for path in tqdm(self.paths, desc="Processing Paths", leave=False):
            inputs = self.process_array_with_list(path, inputs)

        # 对每一行应用 collapse 函数
        collapse_values = []
        for row in tqdm(inputs, desc="Collapsing Rows", total=inputs.shape[0]):
            collapse_values.append(self.collapse(row))
        collapse_values = np.array(collapse_values)

        # 计算准确率
        prediction_matches = collapse_values == targets
        accuracy = np.mean(prediction_matches)

        # 计算损失：使用预测值与目标值之间的方差
        loss = np.var(collapse_values - targets)

        print(f"准确率：{accuracy}")
        print(f"损失（方差）：{loss}")

        # 保存预测错误的数据（如果指定了路径）
        if save_errors_to is not None:
            errors = []
            for i in range(len(inputs)):
                if prediction_matches[i]:
                    continue  # 跳过预测正确的样本

                errors.append({
                    "input": inputs[i].tolist(),          # 转成 list 方便 JSON 序列化
                    "prediction": int(collapse_values[i]),  # 确保是 Python 原生类型
                    "target": int(targets[i])
                })

            with open(save_errors_to, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            print(f"预测错误的样本已保存至: {save_errors_to}")

        return accuracy, loss

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

