def method_profile(
    output_count: int | None = None,  # 改为可选，默认 None
    input_types=None,
    output_types=None,
    group: str = "default",
    weight: float = 1.0,
    is_internal_decorator: bool = False,
    vectorized: bool = False
):
    """
    装饰器：标注方法的输入输出结构、所属组、选择权重和是否支持向量化处理
    c++这种强类型语言更适合实现这套系统，等后面可能迁移过去
    可能后面会添加不同类型函数的分类，但实际上这个功能和逻辑分组区别不大所以暂时不添加
    
    参数说明：
        output_count (int, optional): 输出的数量。如果未提供，则使用 len(output_types)。
        input_types (list of str): 输入类型的列表，如 ['scalar', 'vector']
        output_types (list of str): 输出类型的列表
        group (str): 方法所属的功能组，用于逻辑分组
        weight (float): 在组内的选择概率权重，越大越可能被选中，该部分也可通过遗传算法优化
        is_internal_decorator (bool): 是否为内部装饰器（用于系统内部逻辑）
        vectorized (bool): 是否为向量化函数（可接受批量输入）
    """
    def decorator(func):
        # 确定 output_types 列表
        resolved_output_types = output_types if output_types is not None else []
        
        # 如果 output_count 未提供，使用 output_types 的长度
        if output_count is None:
            resolved_output_count = len(resolved_output_types)
        else:
            resolved_output_count = output_count

        # 附加属性到函数
        func.output_count = resolved_output_count
        func.input_types = input_types if input_types is not None else []
        func.output_types = resolved_output_types
        func.group = group
        func.weight = weight
        func.is_internal_decorator = is_internal_decorator
        func.vectorized = vectorized
        return func
    return decorator

method_profile.is_internal_decorator = True