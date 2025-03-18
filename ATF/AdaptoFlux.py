import random
from enum import Enum
import uuid
import numpy as np
import inspect
import importlib.util
import math
import traceback
import DynamicWeightController

# 定义一个枚举表示不同的坍缩方法
class CollapseMethod(Enum):
    SUM = 1       # 求和
    AVERAGE = 2   # 平均
    VARIANCE = 3  # 方差
    PRODUCT = 4   # 相乘
    CUSTOM = 5  #用于自定义方法

class AdaptoFlux:
    def __init__(self, values, labels, collapse_method=CollapseMethod.SUM):
        """
        初始化 AdaptoFlux 类的实例
        
        :param values: 一维数据值列表
        :param labels: 每个值对应的标签列表
        :param collapse_method: 选择的坍缩方法，默认为 SUM
        """
        
        # 存储输入数据
        self.values = values  # 原始数值列表
        self.labels = labels  # 对应的标签列表

        # 存储处理过程中的值
        self.last_values = values  # 记录上一次处理后的值
        self.histroy_values = [values]  # 记录历史处理值

        # 记录方法及其预输入信息
        self.methods = {}  # 存储方法的字典
        self.method_inputs = {}  # 存储每个方法的预输入索引
        self.histroy_method_inputs = []  # 记录历史每层的预输入索引
        # 意外发现即使不使用历史记录和清空不可取的网络残余（即被清空的网络依然参与预输入索引和预输入值计算）依然会出现概率上升和方差下降
        
        # 存储路径信息
        self.paths = []  # 记录每个值对应的路径

        # 选择的坍缩方法
        self.collapse_method = collapse_method  # 默认使用 SUM 方法
        self.custom_collapse_function = None  # 预定义自定义坍缩方法，默认值为 None
        
        # 记录推理过程中的输入值
        self.method_input_values = {}  # 记录当前层的方法输入值
        self.histroy_method_input_values = []  # 记录历史层的方法输入值
        
        # 监控当前任务的性能指标
        self.metrics = {
            "accuracy": 0.0,  # 准确率
            "entropy": 0.0,  # 路径熵值
            "redundancy_penalty": 0.0,  # 冗余惩罚
        }
    
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
    
    def import_methods_from_file(self, file_path):
        """
        从指定文件中导入所有方法并添加到方法字典中。
        :param file_path: Python 文件路径
        """
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("module.name", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 遍历模块中的所有成员
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):  # 检查是否为函数
                # 获取函数所需的参数数量
                input_count = len(inspect.signature(obj).parameters)
                self.add_method(name, obj, input_count)
        # 记录初始状态
        self.histroy_method_inputs.append(self.method_inputs)
        self.histroy_method_input_values.append(self.method_input_values)
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

        # 存储每个元素对应的方法（单输入方法直接存储方法 ID，多输入方法存储占位符 "-method_id"）
        method_list = []

        # 获取 values 列数，即每个元素的数量
        num_elements = self.last_values.shape[1]

        # 随机选择方法并处理单输入和多输入方法
        for i in range(num_elements):
            method_id = random.choice(list(self.methods.keys()))  # 随机选择一个方法 ID
            method_data = self.methods[method_id]
            input_count = method_data["input_count"]  # 获取该方法需要的输入值数量

            if input_count == 1:
                # 如果方法只需要一个输入值，则直接存储方法 ID
                method_list.append(method_id)
            else:
                # 多输入方法使用占位符，以便后续匹配输入索引
                method_list.append(f"-{method_id}")

        # 处理多输入方法的输入索引匹配
        for index, i in enumerate(method_list):
            if i.startswith('-'):  # 识别多输入方法
                method_id = i.lstrip('-')  # 去除占位符前缀，获取真正的方法 ID
                method_data = self.methods[method_id]
                input_count = method_data["input_count"]  # 获取该方法所需的输入值数量

                # 记录方法所需的输入索引
                self.method_inputs[method_id].append(index)

                # 当方法的输入索引数达到所需数量时，进行替换
                if len(self.method_inputs[method_id]) == input_count:
                    for j in range(len(self.method_inputs[method_id])):
                        if self.method_inputs[method_id][j] is None:
                            # 忽略无效索引
                            continue
                        # 替换 method_list 中的占位符 # 这一行出现过报错，但是无法复现，很奇怪定位不了问题在哪
                        method_list[self.method_inputs[method_id][j]] = method_id
                    # 清空已处理的方法输入索引
                    self.method_inputs[method_id] = []
        
        self.clear_method_inputs()

        return method_list


    # 接受导向列表，返回有n个元素不同的新列表
    def replace_random_elements(self, method_list, n):
        if n > len(method_list):
            raise ValueError("n 不能大于原列表的长度")
        # 随机选择 n 个不重复的索引
        indices_to_replace = random.sample(range(len(method_list)), 1)
        new_list = method_list[:]  # 复制原列表
        for idx in indices_to_replace:
            method_id = random.choice(list(self.methods.keys()))
            method_data = self.methods[method_id]
            input_count = method_data["input_count"]
            if input_count == 1:
                # 方法只需要一个输入值，直接记录方法
                new_list[idx] = method_id
            else:
                new_list[idx] = (f"-{method_id}")
        
        for index, i in enumerate(new_list):
            if i.startswith('-'):
                    continue
            input_count = self.methods[i]["input_count"]
            if input_count == 1:
                continue
            else:
                new_list[index] = (f"-{new_list[index]}")
            
        for index, i in enumerate(new_list):
            if i.startswith('-'):
                method_id = i.lstrip('-')
                method_data = self.methods[method_id] # 在未编写本行代码时也能体现概率的逐渐上升
                input_count = method_data["input_count"]
                self.method_inputs[method_id].append(index)  # 向字典中添加索引
                if len(self.method_inputs[method_id]) == input_count:
                    for j in range(len(self.method_inputs[method_id])):
                        if self.method_inputs[method_id][j] == None:#这里一开始使用的 if self.method_inputs[method_id][j] >= len(new_list):显然有问题，但是运行起来可以正常收敛
                            continue
                        new_list[self.method_inputs[method_id][j]] = method_id
                    self.method_inputs[method_id] = []

        self.clear_method_inputs()

        return new_list


    
    # 根据输入的列表更改数组,处于神经待处理队列状态的值采用暂时不处理方案, 还有一种方案是先将这些值置于最后
    # 使用该方法会导致数据收敛到一定数量级时可能出现所有数据都在待处理队列中导致数组全为空的情况发生
    # 这种方法坍缩时不会将待处理数据加入计算
    def process_array_with_list(self, method_list, max_values_multipie = 5):
        try:
            new_last_values = np.empty((self.last_values.shape[0],max_values_multipie*len(method_list))) #用于存储新计算出的数组
            for j in range(self.last_values.shape[0]):
                new_number = 0 #用于计数新值的位置
                for i in range(len(method_list)):
                    if method_list[i].startswith('-'):
                        self.method_input_values[method_list[i].lstrip('-')].append(self.last_values[j,i])
                        continue
                    else:
                        if self.methods[method_list[i]]['input_count'] == 1:
                            for k in self.methods[method_list[i]]['function'](self.last_values[j, i]):
                                new_last_values[j, new_number] = k
                                new_number += 1
                        else:
                            self.method_input_values[method_list[i]].append(self.last_values[j,i])
                            if len(self.method_input_values[method_list[i]]) == self.methods[method_list[i]]['input_count']:
                                for k in self.methods[method_list[i]]['function'](*[value for value in self.method_input_values[method_list[i]]]):
                                    new_last_values[j, new_number] = k
                                    new_number += 1
                                self.method_input_values[method_list[i]] = []
                            else:
                                continue
            # 对数组处理后获取还剩余值的数组
            new_last_values = new_last_values[:, :new_number]  # 根据 new_number 更新形状
            # #这一部分如果使用的函数集出现同一函数有不同数量输出的话会出现问题，从设计上是不应该出现的错误，需要修改#问题出在如果使用不同的函数输出验证集和训练集得到的method_list长度应该是不同的，但是在实际使用中我使用了相同的method_list导致出现问题，考虑将验证集分离出来
            # new_last_val_values = np.empty((self.last_val_values.shape[0],len(method_list))) #同上，但用于验证集
            # for j in range(self.last_val_values.shape[0]):
            #     new_number = 0 #用于计数新值的位置
            #     for i in range(len(method_list)):
            #         if method_list[i].startswith('-'):
            #             continue
            #         else:
            #             if self.methods[method_list[i]]['input_count'] == 1:
            #                 for k in self.methods[method_list[i]]['function'](self.last_val_values[j, i]):
            #                     new_last_val_values[j, new_number] = k
            #                     new_number += 1
            #             else:
            #                 self.method_input_val_values[method_list[i]].append(self.last_val_values[j,i])
            #                 if len(self.method_input_val_values[method_list[i]]) == self.methods[method_list[i]]['input_count']:
            #                     for k in self.methods[method_list[i]]['function'](*[value for value in self.method_input_val_values[method_list[i]]]):
            #                         new_last_val_values[j, new_number] = k
            #                         new_number += 1
            #                     self.method_input_val_values[method_list[i]] = []
            #                 else:
            #                     continue

            # new_last_val_values = new_last_val_values[:, :new_number]  # 根据 new_number 更新形状

            #return new_last_values,new_last_val_values
            return new_last_values
        except Exception as e:
            print("出现错误，返回原数组：", str(e) )
            traceback.print_exc()
            # print(self.last_val_values.shape,self.last_values.shape)
            # return self.last_values,self.last_val_values
            print(self.last_values.shape)
            return self.last_values

                

    # 训练方法,epochs决定最终训练出来的模型层数,step用于控制重随机时每次增加几个重随机的指数上升速度
    def training(self, epochs=10000, depth_interval=1,depth_reverse=1, step = 2):
        # 清空路径列表
        self.paths = []
        # 创建动态权重控制器
        dynamicWeightController = DynamicWeightController.DynamicWeightController(epochs)

        for i in range(epochs):
            print("epoch:",i)
            last_method = self.process_random_method()
            # new_last_values,new_last_val_values = self.process_array_with_list(last_method)
            new_last_values = self.process_array_with_list(last_method)
            #计算训练集方差和正确率
            last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
            new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
            last_vs_train = last_collapse_values == self.labels  # 返回布尔数组，表示每个元素是否相等
            new_vs_train = new_collapse_values == self.labels
            #计算前后方差
            last_difference_trian = last_collapse_values - self.labels
            last_variance_trian = np.var(last_difference_trian)
            new_difference_trian = new_collapse_values - self.labels
            new_variance_trian = np.var(new_difference_trian)
            #计算准确率
            last_accuracy_trian = np.mean(last_vs_train)  # 相等的比例，准确率
            new_accuracy_trian = np.mean(new_vs_train)  # 相等的比例，准确率
            print("上一轮训练集相等概率:" + str(last_accuracy_trian))
            print("本轮训练集相等概率：" + str(new_accuracy_trian))
            print("上一轮训练集方差:" + str(last_variance_trian))
            print("本轮训练集方差：" + str(new_variance_trian))

            # #计算验证集方差和正确率
            # last_val_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_val_values)
            # new_val_collpase_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_val_values)
            # last_vs_val = last_val_collapse_values == y_val
            # new_vs_val = new_val_collpase_values == y_val
            # #计算前后方差
            # last_difference_val = last_val_collapse_values - y_val
            # last_variance_val = np.var(last_difference_val)
            # new_difference_val = new_val_collpase_values - y_val
            # new_variance_val = np.var(new_difference_val)
            # #计算准确率
            # last_accuracy_val = np.mean(last_vs_val)  # 相等的比例，准确率
            # new_accuracy_val = np.mean(new_vs_val)  # 相等的比例，准确率
            # print("上一轮验证集相等概率:" + str(last_accuracy_val))
            # print("本轮验证集相等概率：" + str(new_accuracy_val))
            # print("上一轮验证集方差:" + str(last_variance_val))
            # print("本轮验证集方差：" + str(new_variance_val))

            # if (((new_variance_trian < last_variance_trian and 
            #      new_variance_val < last_variance_val) or 
            #     (last_accuracy_trian < new_accuracy_trian and
            #      last_accuracy_val < new_accuracy_val)) and
            #     (new_last_values.size != 0 and 
            #      new_last_val_values.size != 0)):
            #     self.paths.append(last_method)
            #     self.histroy_values.append(new_last_values)
            #     self.histroy_val_values.append(new_last_val_values)
            #     self.last_values = new_last_values
            #     self.last_val_values = new_last_val_values
            #     # 将添加的网络加入历史中
            #     self.histroy_method_inputs.append(self.method_inputs)
            #     self.histroy_method_input_values.append(self.method_input_values)
            #     self.history_method_input_val_values.append(self.method_input_val_values)
            # if ((new_variance_trian < last_variance_trian or 
            #     last_accuracy_trian < new_accuracy_trian) and
            #     new_last_values.size != 0):
            if (last_accuracy_trian < new_accuracy_trian or
                (last_accuracy_trian == new_accuracy_trian and new_variance_trian < last_variance_trian)) and \
                new_last_values.size != 0:


                self.paths.append(last_method)
                self.histroy_values.append(new_last_values)
                # self.histroy_val_values.append(new_last_val_values)
                self.last_values = new_last_values
                # self.last_val_values = new_last_val_values
                # 将添加的网络加入历史中
                self.histroy_method_inputs.append(self.method_inputs)
                self.histroy_method_input_values.append(self.method_input_values)
                # self.history_method_input_val_values.append(self.method_input_val_values)

            else:
                i = 1
                while i <= len(last_method):
                    print(f"在当层重新寻找适合的路径：当前重随机数{i}")
                    self.method_inputs = self.histroy_method_inputs[-1]
                    self.method_input_values = self.histroy_method_input_values[-1]
                    # self.method_input_val_values = self.history_method_input_val_values[-1]
                    last_method = self.replace_random_elements(last_method, i)
                    i *= step
                    # new_last_values,new_last_val_values = self.process_array_with_list(last_method)
                    new_last_values = self.process_array_with_list(last_method)

                    #计算训练集方差和正确率
                    last_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_values)
                    new_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_values)
                    last_vs_train = last_collapse_values == self.labels  # 返回布尔数组，表示每个元素是否相等
                    new_vs_train = new_collapse_values == self.labels
                    #计算前后方差
                    last_difference_trian = last_collapse_values - self.labels
                    last_variance_trian = np.var(last_difference_trian)
                    new_difference_trian = new_collapse_values - self.labels
                    new_variance_trian = np.var(new_difference_trian)
                    #计算准确率
                    last_accuracy_trian = np.mean(last_vs_train)  # 相等的比例，准确率
                    new_accuracy_trian = np.mean(new_vs_train)  # 相等的比例，准确率
                    print("上一轮训练集相等概率:" + str(last_accuracy_trian))
                    print("本轮训练集相等概率：" + str(new_accuracy_trian))
                    print("上一轮训练集方差:" + str(last_variance_trian))
                    print("本轮训练集方差：" + str(new_variance_trian))

                    # #计算验证集方差和正确率
                    # last_val_collapse_values = np.apply_along_axis(self.collapse, axis=1, arr=self.last_val_values)
                    # new_val_collpase_values = np.apply_along_axis(self.collapse, axis=1, arr=new_last_val_values)
                    # last_vs_val = last_val_collapse_values == y_val
                    # new_vs_val = new_val_collpase_values == y_val
                    # #计算前后方差
                    # last_difference_val = last_val_collapse_values - y_val
                    # last_variance_val = np.var(last_difference_val)
                    # new_difference_val = new_val_collpase_values - y_val
                    # new_variance_val = np.var(new_difference_val)
                    # #计算准确率
                    # last_accuracy_val = np.mean(last_vs_val)  # 相等的比例，准确率
                    # new_accuracy_val = np.mean(new_vs_val)  # 相等的比例，准确率
                    # print("上一轮验证集相等概率:" + str(last_accuracy_val))
                    # print("本轮验证集相等概率：" + str(new_accuracy_val))
                    # print("上一轮验证集方差:" + str(last_variance_val))
                    # print("本轮验证集方差：" + str(new_variance_val))
                    # if (((new_variance_trian < last_variance_trian and 
                    #     new_variance_val < last_variance_val) or 
                    #     (last_accuracy_trian < new_accuracy_trian and
                    #     last_accuracy_val < new_accuracy_val)) and
                    #     (new_last_values.size != 0 and 
                    #     new_last_val_values.size != 0)):
                    #     self.paths.append(last_method)
                    #     self.histroy_values.append(new_last_values)
                    #     self.histroy_val_values.append(new_last_val_values)
                    #     self.last_values = new_last_values
                    #     self.last_val_values = new_last_val_values
                    #     # 将添加的网络加入历史中
                    #     self.histroy_method_inputs.append(self.method_inputs)
                    #     self.histroy_method_input_values.append(self.method_input_values)
                    #     self.history_method_input_val_values.append(self.method_input_val_values)
                    #     break
                    # if ((new_variance_trian < last_variance_trian or 
                    #     last_accuracy_trian < new_accuracy_trian) and
                    #     new_last_values.size != 0):
                    if (last_accuracy_trian < new_accuracy_trian or
                        (last_accuracy_trian == new_accuracy_trian and new_variance_trian < last_variance_trian)) and \
                        new_last_values.size != 0:
                        self.paths.append(last_method)
                        self.histroy_values.append(new_last_values)
                        # self.histroy_val_values.append(new_last_val_values)
                        self.last_values = new_last_values
                        # self.last_val_values = new_last_val_values
                        # 将添加的网络加入历史中
                        self.histroy_method_inputs.append(self.method_inputs)
                        self.histroy_method_input_values.append(self.method_input_values)
                        # self.history_method_input_val_values.append(self.method_input_val_values)
                        break    
                else:
                    print('清除上一层网络')
                    # 清除上一层网络重新寻找
                    self.histroy_method_inputs.pop()
                    self.histroy_method_input_values.pop()
                    # self.history_method_input_val_values.pop()
                    self.paths.pop()
                    if len(self.histroy_values) > 1:
                        self.histroy_values.pop()
                    # if len(self.histroy_val_values) > 1:
                    #     self.histroy_val_values.pop()
                    self.last_values = self.histroy_values[-1]
                    # self.last_val_values = self.histroy_val_values[-1]

                    

                
        # 打开文件并写入
        with open("output.txt", "w") as f:
            for item in self.paths:
                f.write(str(item) + "\n")


    def evaluate(self,inputs, targets):
        return
        #for i in range(len(self.paths)):
        
