def method_profile(
    output_count: int,
    input_types=None,
    output_types=None,
    group: str = "default",
    weight: float = 1.0
):
    """
    装饰器：标注方法的输入输出结构、所属组和选择权重。
    
    参数说明：
        output_count (int): 输出的数量
        input_types (list of str): 输入类型的列表，如 ['scalar', 'vector']
        output_types (list of str): 输出类型的列表
        group (str): 方法所属的功能组，用于逻辑分组
        weight (float): 在组内的选择概率权重，越大越可能被选中
    """
    def decorator(func):
        func.output_count = output_count
        func.input_types = input_types if input_types is not None else []
        func.output_types = output_types if output_types is not None else []
        func.group = group
        func.weight = weight
        return func
    return decorator