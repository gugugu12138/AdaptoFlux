"""精确核验旧 GraphEvo 任务的深度可达性；只读取源码，不导入旧实验。

示例：
    python -B experiments/typed_dag_comparison_20261108/reachability_audit.py \
        --output <报告路径.json>

这里枚举的是旧实验实际使用的单输入、单输出仿射方法链，不是任意 DAG。
每个方法都写成 a*x+b，并以 Fraction 计算；0.5 等源码常数按十进制文本
转换，避免浮点近似决定“可达/不可达”。报告同时保留语法候选数、不同函数
数和最短见证，不能把不同方法序列误算成不同数学函数。
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_SOURCE = ROOT / "experiments/GraphEvo_exp3/exp3_main.py"
TASK_SOURCE = ROOT / "experiments/GraphEvo_exp3/math_tasks.py"
IDENTITY = (Fraction(1), Fraction(0))
EXPECTED_TARGET = (Fraction(1), Fraction(5, 2))
MAX_DEPTH = 4


def compose(outer, inner):
    """计算 outer(inner(x))；序列中的后一个方法作用于前一个的输出。"""
    a, b = outer
    c, d = inner
    return a * c, a * d + b


def affine_expression(node, variable, named_functions=None, expanding=()):
    """静态解释源码里受支持的仿射表达式，不执行任何 Python 源码。

    只接受常数、一个形参、加减、与常数相乘/相除，以及 math_tasks 中
    已声明的单参数 lambda 调用。源码将来若变成非仿射形式则明确报错，
    不悄悄继续用本脚本中硬编码的旧任务。
    """
    functions = named_functions or {}
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id == variable:
        return IDENTITY
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = affine_expression(node.operand, variable, functions, expanding)
        return value if isinstance(node.op, ast.UAdd) else (-value[0], -value[1])
    if isinstance(node, ast.BinOp):
        left = affine_expression(node.left, variable, functions, expanding)
        right = affine_expression(node.right, variable, functions, expanding)
        if isinstance(node.op, ast.Add):
            return left[0] + right[0], left[1] + right[1]
        if isinstance(node.op, ast.Sub):
            return left[0] - right[0], left[1] - right[1]
        if isinstance(node.op, ast.Mult):
            if left[0] and right[0]:
                raise ValueError("源码包含非仿射乘法，不能使用仿射可达性审计")
            return left[0] * right[1] + right[0] * left[1], left[1] * right[1]
        if isinstance(node.op, ast.Div) and right[0] == 0 and right[1] != 0:
            return left[0] / right[1], left[1] / right[1]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in functions and len(node.args) == 1 and not node.keywords):
        name = node.func.id
        if name in expanding:
            raise ValueError("任务 lambda 递归引用，不能静态展开")
        function = functions[name]
        if len(function.args.args) != 1:
            raise ValueError("任务必须是单参数 lambda")
        outer = affine_expression(function.body, function.args.args[0].arg,
                                  functions, expanding + (name,))
        inner = affine_expression(node.args[0], variable, functions, expanding)
        return compose(outer, inner)
    raise ValueError(f"不支持的源码表达式：{ast.dump(node, include_attributes=False)}")


def assigned_value(statements, name):
    """只查当前语句层级，避免把其他 variant 分支的赋值混进当前方法池。"""
    matches = [statement.value for statement in statements
               if isinstance(statement, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name
                       for target in statement.targets)]
    if len(matches) != 1:
        raise ValueError(f"需要唯一的 {name} 赋值，实际找到 {len(matches)} 个")
    return matches[0]


def parse_method_list(expression, named_lists):
    """从旧工厂函数的列表字面量读取方法名、lambda、输入输出数量。"""
    if isinstance(expression, ast.Name) and expression.id in named_lists:
        return list(named_lists[expression.id])
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return (parse_method_list(expression.left, named_lists)
                + parse_method_list(expression.right, named_lists))
    if not isinstance(expression, ast.List):
        raise ValueError("方法池不再是可静态核验的列表拼接")
    result = []
    for item in expression.elts:
        if not isinstance(item, ast.Tuple) or len(item.elts) != 4:
            raise ValueError("方法记录必须包含名称、lambda、输入数、输出数")
        name, function, inputs, outputs = item.elts
        if not isinstance(name, ast.Constant) or type(name.value) is not str:
            raise ValueError("方法名必须是字符串字面量")
        if ast.literal_eval(inputs) != 1 or ast.literal_eval(outputs) != 1:
            raise ValueError("该审计仅适用于单输入、单输出方法")
        if not isinstance(function, ast.Lambda) or len(function.args.args) != 1:
            raise ValueError("方法必须是单参数 lambda")
        if not isinstance(function.body, ast.List) or len(function.body.elts) != 1:
            raise ValueError("方法 lambda 必须按原接口返回一个输出")
        pair = affine_expression(function.body.elts[0], function.args.args[0].arg)
        result.append((name.value, pair))
    return result


def read_original_protocol():
    """核验真实旧源码及任务3的层数上限；不会触发旧脚本的日志文件创建。"""
    experiment_text = EXPERIMENT_SOURCE.read_text(encoding="utf-8-sig")
    task_text = TASK_SOURCE.read_text(encoding="utf-8-sig")
    tree, tasks = ast.parse(experiment_text), ast.parse(task_text)
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "create_adaptoflux_variant")
    base = parse_method_list(assigned_value(factory.body, "base_methods"), {})
    pools = {}
    # ast.walk 会列出 if/elif 的每个分支；只提取其直接 body 中的 methods 赋值。
    for node in ast.walk(factory):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "variant" and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)):
            variant = test.comparators[0].value
            if variant in pools:
                raise ValueError(f"重复 variant 分支：{variant}")
            pools[variant] = parse_method_list(assigned_value(node.body, "methods"),
                                              {"base_methods": base})
    if set(pools) != {"minimal", "extended", "oracle"}:
        raise ValueError(f"旧实验方法池发生变化：{sorted(pools)}")

    # 不依赖注释或变量名猜测层数：直接读取 trainer3 = GraphEvoTrainer(...)。
    runner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "run_experiment_group")
    trainer = assigned_value(runner.body, "trainer3")
    if not (isinstance(trainer, ast.Call) and isinstance(trainer.func, ast.Name)
            and trainer.func.id == "GraphEvoTrainer"):
        raise ValueError("trainer3 不再直接构造 GraphEvoTrainer")
    depth_nodes = [keyword.value for keyword in trainer.keywords
                   if keyword.arg == "max_init_layers"]
    if len(depth_nodes) != 1 or ast.literal_eval(depth_nodes[0]) != 3:
        raise ValueError("任务3的原始 max_init_layers 不再等于3，需重新审查协议")

    functions = {target.id: node.value for node in tasks.body
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda)
                 for target in node.targets if isinstance(target, ast.Name)}
    target = affine_expression(functions["f3"].body, functions["f3"].args.args[0].arg,
                               functions, ("f3",))
    if target != EXPECTED_TARGET:
        raise ValueError(f"旧任务3发生变化：{target}")
    audit = {
        "experiment_source": EXPERIMENT_SOURCE.relative_to(ROOT).as_posix(),
        "experiment_sha256": hashlib.sha256(EXPERIMENT_SOURCE.read_bytes()).hexdigest(),
        "task_source": TASK_SOURCE.relative_to(ROOT).as_posix(),
        "task_sha256": hashlib.sha256(TASK_SOURCE.read_bytes()).hexdigest(),
        "inspection": "AST only; no import, eval, exec, or execution of experiment code",
        "factory": factory.name,
        "task3_trainer_function": runner.name,
        "task3_max_init_layers": 3,
        "task3_max_init_layers_source_line": depth_nodes[0].lineno,
        "task3_target": pair_record(target),
    }
    return pools, target, audit


def pair_record(pair):
    """JSON中以精确分数字符串保存系数，避免写入有损浮点数。"""
    return {"a": str(pair[0]), "b": str(pair[1])}


def enumerate_pool(methods, target):
    """逐深度穷举所有语法序列，同时汇总相同(a,b)的语义等价类。"""
    shortest = {}
    depths = []
    cumulative_candidates = 0
    for depth in range(MAX_DEPTH + 1):
        counts, first_at_depth, target_witnesses = Counter(), {}, []
        for sequence in product(methods, repeat=depth):
            value = IDENTITY
            names = []
            for name, operation in sequence:
                value = compose(operation, value)
                names.append(name)
            counts[value] += 1
            first_at_depth.setdefault(value, names)
            shortest.setdefault(value, {"depth": depth, "methods": names})
            if value == target:
                target_witnesses.append(names)
        candidate_count = sum(counts.values())
        cumulative_candidates += candidate_count
        depths.append({
            "depth": depth,
            "candidate_count": candidate_count,
            "distinct_semantics_count": len(counts),
            "cumulative_candidate_count": cumulative_candidates,
            "cumulative_distinct_semantics_count": len(shortest),
            "target_reachable_at_exact_depth": bool(target_witnesses),
            "target_reachable_within_depth": target in shortest,
            "target_witnesses_at_exact_depth": target_witnesses,
            "semantics": [{**pair_record(value), "sequence_count": counts[value],
                           "first_witness_at_exact_depth": first_at_depth[value]}
                          for value in sorted(counts)],
        })
    return {
        "operators": [{"name": name, **pair_record(pair)} for name, pair in methods],
        "operator_count": len(methods),
        "depths": depths,
        "depth3_reachable": depths[3]["target_reachable_within_depth"],
        "depth4_reachable": depths[4]["target_reachable_within_depth"],
        "depth4_witnesses": depths[4]["target_witnesses_at_exact_depth"],
        "target_shortest_witness": shortest.get(target),
        "shortest_witnesses": [{**pair_record(value), **shortest[value]}
                               for value in sorted(shortest)],
    }


def build_report():
    pools, target, audit = read_original_protocol()
    results = {name: enumerate_pool(pools[name], target)
               for name in ("minimal", "extended", "oracle")}
    # 这些断言是对审计结论的检查；若源码改变，宁可失败，也不发布旧结论。
    for name in ("minimal", "extended"):
        if results[name]["depth3_reachable"] or not results[name]["depth4_reachable"]:
            raise AssertionError(f"{name} 的原语深度可达性与待核验协议不同")
    if not results["oracle"]["depth3_reachable"]:
        raise AssertionError("oracle 方法池应能在三步内表示原任务3")
    return {
        "experiment": "exact affine-chain reachability audit of original GraphEvo exp3",
        "arithmetic": "fractions.Fraction; affine maps a*x+b; exact semantic equality",
        "scope": "unary affine chains with the operator pools extracted from original source",
        "max_enumerated_depth": MAX_DEPTH,
        "depth_definition": "number of top-level registered method calls; identity also counts",
        "witness_order": "execution order from input to output",
        "source_audit": audit,
        "pools": results,
        "interpretation": [
            "The minimal and extended primitive pools cannot express f3 within the original three-call limit.",
            "Both primitive pools can express f3 with four calls; the oracle pool can express it with three.",
            "The original zero-success controls therefore include a representation-capacity restriction.",
            "This is a reachability result, not a search-success, runtime, or general typed-DAG comparison.",
            "For a learned-composite comparison, separately count fully expanded primitive nodes; a composite call is not one primitive.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON报告的明确输出路径；脚本只写这个报告")
    args = parser.parse_args()
    report = build_report()
    # 在全部源码核验和枚举成功后才写报告，不留下半份看似成功的结果。
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pools": {
        name: {"depth3_reachable": result["depth3_reachable"],
               "depth4_reachable": result["depth4_reachable"],
               "shortest_witness": result["target_shortest_witness"]}
        for name, result in report["pools"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
