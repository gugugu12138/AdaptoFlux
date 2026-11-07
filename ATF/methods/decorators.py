def method_profile(
    output_count: int | None = None,  # 改为可选，默认 None
    input_types=None,
    output_types=None,
    group: str = "default",
    weight: float = 1.0,
    is_internal_decorator: bool = False,
    vectorized: bool = False,
    pure: bool | None = None,
    reads=None,
    writes=None,
):
    """
    装饰器：标注方法的输入输出结构、所属组、选择权重和是否支持向量化处理
    c++这种强类型语言更适合实现这套系统，等后面可能迁移过去
    可能后面会添加不同类型函数的分类，但实际上这个功能和逻辑分组区别不大所以暂时不添加
    以及这一块其实可以升级一下，添加对方法使用次数的统计或者内存占用什么的来支持之后的算法，但是现在还没有用到就先不考虑
    
    参数说明：
        output_count (int, optional): 输出的数量。如果未提供，则使用 len(output_types)。
        input_types (list of str): 输入类型的列表，如 ['scalar', 'vector']
        output_types (list of str): 输出类型的列表
        group (str): 方法所属的功能组，用于逻辑分组
        weight (float): 在组内的选择概率权重，越大越可能被选中，该部分也可通过遗传算法优化
        is_internal_decorator (bool): 是否为内部装饰器（用于系统内部逻辑）
        vectorized (bool): 是否为向量化函数（可接受批量输入），当前版本非常不建议启动该选项，几乎必然出现问题
        pure (bool | None): True 表示方法不访问图外可变状态。未声明时按 unknown
            保守处理，不能因为输入输出都是数值就把旧函数认作纯函数。
        reads, writes: 方法读取、写入的外部资源名称，例如 {"environment"}。
            这些声明独立于输入输出类型，只用于决定哪些节点必须按顺序执行。
            声明由方法作者负责，系统不会自动分析任意 Python 函数的真实副作用。
    """
    from ..GraphProcessor.execution_plan import normalize_effects

    # 注册时就检查矛盾声明（例如 pure=True 却声明写状态），同时把资源名称
    # 排序、去重，得到可以进入模型文件且不依赖集合迭代顺序的统一表示。
    effects = normalize_effects(pure=pure, reads=reads, writes=writes)

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
        # 每个函数持有自己的声明副本，避免同一个装饰器被复用时共享可变列表。
        func.effects = {"kind": effects["kind"],
                        "reads": list(effects["reads"]),
                        "writes": list(effects["writes"])}
        return func
    return decorator

method_profile.is_internal_decorator = True
