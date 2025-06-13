def method_profile(
    output_count: int,
    input_types=None,
    output_types=None,
    group: str = "default",
    weight: float = 1.0,
    vectorized: bool = False  # 新增参数，默认不是向量化方法
):
    """
    装饰器：标注方法的输入输出结构、所属组、选择权重和是否支持向量化处理
    
    参数说明：
        output_count (int): 输出的数量
        input_types (list of str): 输入类型的列表，如 ['scalar', 'vector']
        output_types (list of str): 输出类型的列表
        group (str): 方法所属的功能组，用于逻辑分组
        weight (float): 在组内的选择概率权重，越大越可能被选中
        vectorized (bool): 是否为向量化函数（可接受批量输入）
    """
    def decorator(func):
        func.output_count = output_count
        func.input_types = input_types if input_types is not None else []
        func.output_types = output_types if output_types is not None else []
        func.group = group
        func.weight = weight
        func.vectorized = vectorized  # 添加新属性
        return func
    return decorator