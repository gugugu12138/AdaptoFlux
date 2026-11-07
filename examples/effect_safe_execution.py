"""隐式状态的确定执行示例：数据边不变，控制依赖由运行器单独维护。

从仓库根目录运行：python -B -m examples.effect_safe_execution
示例不写文件，分别展示串行、并行以及显式改变程序顺序的结果。
"""
import numpy as np

from ATF.core.adaptoflux import AdaptoFlux
from ATF.methods.decorators import method_profile


STATE = {"counter": 1}


@method_profile(input_types=["scalar"], output_types=["scalar"],
                reads={"demo.counter"}, writes={"demo.counter"})
def increment(dummy):
    """保持旧式隐式状态接口；dummy 是数据参数，函数没有令牌参数。"""
    STATE["counter"] += 1
    return STATE["counter"]


@method_profile(input_types=["scalar"], output_types=["scalar"],
                reads={"demo.counter"}, writes={"demo.counter"})
def double(dummy):
    """资源名必须与increment相同，才能表示访问的是同一个计数器。"""
    STATE["counter"] *= 2
    return STATE["counter"]


@method_profile(input_types=["scalar"], output_types=["scalar"], reads={"demo.counter"})
def read_counter(dummy):
    """读取也属于副作用依赖分析；它不能越过前面的计数器更新。"""
    return STATE["counter"]


def main():
    # 这里直接注册三个函数，避免把示例的main等辅助函数当作方法池成员。
    model = AdaptoFlux(values=np.array([[0.0]]), input_types_list=["scalar"])
    for function in [increment, double, read_counter]:
        model.add_method(function.__name__, function, 1, function.output_count,
                         function.input_types, function.output_types,
                         function.group, function.weight, function.vectorized)
    graph = model.graph
    graph.remove_edges_from(list(graph.edges(keys=True)))
    for name in ["increment", "double", "read_counter"]:
        graph.add_node(name, method_name=name, layer=1, is_passthrough=False)
        graph.add_edge("root", name, output_index=0, input_slot=0, data_type="scalar")
    graph.add_edge("read_counter", "collapse", output_index=0, data_coord=0, data_type="scalar")

    # 此时数据图没有increment→double→read的连线；准备计划后，调度器会
    # 单独维护这两条控制依赖。它们不会计入函数参数数量或污染类型选路。
    processor = model.graph_processor
    plan = processor.prepare_execution_plan()
    print("冻结的程序顺序：", plan.order)
    print("独立的控制依赖：", plan.control_edges)
    STATE["counter"] = 1
    serial = model.infer_with_graph([[0]])[0]
    STATE["counter"] = 1
    parallel = model.infer_with_task_parallel([[0]], num_workers=4)[0]
    assert serial == parallel == 4
    print(f"相同程序：串行={serial}，4线程={parallel}")

    # 顺序变化是显式的程序变异。训练中需要重新评价，不可把旧分数沿用到
    # 这个新程序上。现在先乘2再加1，结果应为3，而不是原程序的4。
    processor.set_execution_order(["double", "increment", "read_counter"])
    STATE["counter"] = 1
    changed = model.infer_with_task_parallel([[0]], num_workers=4)[0]
    assert changed == 3
    print(f"显式改序后的新程序：{changed}")

    # 也可以在构造 AdaptoFlux 时传 enable_effect_ordering=False。
    # 这里演示已有模型的关闭/重新开启接口。关闭后没有额外控制边，但数据
    # 前驱仍须完成；不使用一次偶然的线程运行结果宣称共享状态仍然确定。
    model.set_effect_ordering(False)
    assert not model.enable_effect_ordering
    assert processor.prepare_execution_plan().control_edges == ()
    print("关闭副作用排序：只按数据依赖调度")
    model.set_effect_ordering(True)
    assert model.enable_effect_ordering
    print("重新开启：", processor.prepare_execution_plan().order)


if __name__ == "__main__":
    main()
