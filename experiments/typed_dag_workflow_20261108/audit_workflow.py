"""独立核验工作流对照的候选算术、模块依赖、落盘规格与汇总计数。

本文件不导入任何实验运行器、ATF 或 PyTensor，不调用被测执行器。
它用手写 NumPy 算术递归求值，把记录的候选、模块和磁盘快照当作数据。
枚举次序按照公开协议独立重建，因而也检查遗漏候选、提前停止和准入污染。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRIMITIVES = ["increment", "multiply_by_2", "add_3", "multiply_by_05", "identity"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate_pytensor_sources():
    """仅定位源目录；find_spec 查询顶层包时不会执行它的 __init__。"""
    locations = [
        os.environ.get("ATF_WORKFLOW_DEPS"),
        str(Path(tempfile.gettempdir()) / "atf_pytensor_workflow_20261108/deps"),
        str(Path.home() / "AppData/Local/Temp/atf_pytensor_workflow_20261108/deps"),
    ]
    for location in locations:
        if location and (Path(location) / "pytensor/__init__.py").is_file():
            return Path(location) / "pytensor"
    spec = importlib.util.find_spec("pytensor")
    if spec is None or spec.origin is None:
        raise RuntimeError("The frozen PyTensor source package is unavailable for hashing")
    return Path(spec.origin).resolve().parent


def verify_sources(frozen):
    """检查冻结实验源码及两个原生包的完整 Python 源文件集合和内容。"""
    for name, expected in frozen["experiment_source_hashes"].items():
        assert sha(HERE / name) == expected, ("experiment_source_changed", name)
    roots = {
        "ATF_V0": REPO / "experiments/rebuttal_20261106/runtime_v0/ATF",
        "PyTensor": locate_pytensor_sources(),
    }
    current = {}
    for label, root in roots.items():
        for path in sorted(root.rglob("*.py")):
            current[f"{label}/{path.relative_to(root).as_posix()}"] = sha(path)
    assert current == frozen["native_source_hashes"], "Frozen native sources changed"
    return {label: sum(name.startswith(label + "/") for name in current) for label in roots}


def evaluate(program, modules, samples, stack=()):
    """从输出递归求值，共享引用只算一次，模块调用按内部展开计费。"""
    width = len(program["input_types"])
    assert all(item == "scalar" for item in program["input_types"])
    assert samples.ndim == 2 and samples.shape[1] == width
    memo = {i: samples[:, i] for i in range(width)}
    operations = 0
    dependencies = set()

    def visit(reference):
        nonlocal operations
        assert type(reference) is int and 0 <= reference < width + len(program["nodes"])
        if reference in memo:
            return memo[reference]
        node = program["nodes"][reference - width]
        assert all(type(ref) is int and 0 <= ref < reference for ref in node["inputs"])
        values = [visit(ref) for ref in node["inputs"]]
        name = node["method"]
        if name in modules:
            assert name not in stack, "Recursive module cycle"
            result, cost, nested = evaluate(modules[name], modules, np.column_stack(values), (*stack, name))
            operations += cost
            dependencies.add(name)
            dependencies.update(nested)
        else:
            # 故意不用适配层的函数表，防止两个执行器共用错误函数而彼此一致。
            if name == "add":
                assert len(values) == 2
                result = values[0] + values[1]
            else:
                assert len(values) == 1
                x = values[0]
                if name == "increment": result = x + 1
                elif name == "multiply_by_2": result = x * 2
                elif name == "add_3": result = x + 3
                elif name == "multiply_by_05": result = x / 2
                elif name == "identity": result = x
                else: raise AssertionError(f"Unknown primitive {name}")
            operations += 1
        memo[reference] = result
        return result

    output = visit(program["output"])
    return output, operations, dependencies


def chain_program(names):
    # 独立构造公开的链式声明格式，不复用训练脚本中的转换函数。
    return {"input_types": ["scalar"],
            "nodes": [{"method": name, "inputs": [index]} for index, name in enumerate(names)],
            "output": len(names)}


def audit_case(record, snapshot, plan):
    train = (np.arange(-8, 9, dtype=np.float64) * (3 / 8)).reshape(-1, 1)
    test = (np.arange(-31, 33, 2, dtype=np.float64) / 4 + 1 / 16).reshape(-1, 1)
    assert not set(train[:, 0]) & set(test[:, 0])
    # 直接写出最终仿射函数，避免复用被测复合函数的构造逻辑。
    targets = [(2, 2), (2, 5), (1, 2.5)]
    names = PRIMITIVES.copy()
    random.Random(record["order_seed"]).shuffle(names)
    original_order = tuple(names)
    modules = {}
    costs = {name: 1 for name in names}
    audited = 0
    stage_dependencies = []
    assert len(record["stages"]) == 3
    assert [row["stage"] for row in record["trace"]] == sorted(row["stage"] for row in record["trace"])
    for stage, ((a, b), stage_record) in enumerate(zip(targets, record["stages"], strict=True), 1):
        assert stage_record["stage"] == stage
        rows = [row for row in record["trace"] if row["stage"] == stage]
        assert len(rows) == stage_record["evaluations"] <= plan["max_evaluations_per_stage"]
        assert rows
        candidates = (chain for depth in range(1, plan["max_top_level_calls"] + 1)
                      for chain in itertools.product(names, repeat=depth)
                      if sum(costs[name] for name in chain) <= plan["max_expanded_primitive_calls"])
        for evaluation, row in enumerate(rows, 1):
            chain = next(candidates)
            assert row["evaluation"] == evaluation and row["chain"] == list(chain)
            program = chain_program(chain)
            values, expanded, dependencies = evaluate(program, modules, train)
            assert expanded == sum(costs[name] for name in chain)
            assert row["expanded_calls"] == expanded <= plan["max_expanded_primitive_calls"]
            loss = float(np.mean((values - (a * train[:, 0] + b)) ** 2))
            assert row["mse"] == loss
            # 每阶段应停在第一个训练解；既不能漏掉成功，也不能继续筛解。
            assert (loss <= plan["success_mse"]) == (evaluation == len(rows))
            audited += 1
        assert stage_record["program"] == program
        assert stage_record["expanded_calls"] == expanded
        direct = [node["method"] for node in program["nodes"] if node["method"] in modules]
        assert stage_record["module_dependencies"] == direct
        if stage == 2:
            assert "learned_1" in direct, "Stage 2 failed to reuse the discovered module"
        if stage == 3:
            assert "learned_2" in direct and "learned_1" in dependencies
        stage_dependencies.append({"stage": stage, "direct": direct, "transitive": sorted(dependencies)})
        verification, check_cost, _ = evaluate(program, modules, test)
        np.testing.assert_array_equal(verification, a * test[:, 0] + b)
        np.testing.assert_array_equal(verification, stage_record["verification_prediction"])
        assert check_cost == expanded and stage_record["verification_mse"] == 0
        if stage < 3:
            name = f"learned_{stage}"
            assert stage_record["module_added"] == name and name not in modules
            modules[name] = program
            names.append(name)
            costs[name] = expanded
        else:
            assert stage_record["module_added"] is None
    assert audited == len(record["trace"])
    expected = {"modules": [{"name": name, "program": body} for name, body in modules.items()],
                "program": record["stages"][-1]["program"]}
    assert snapshot == record["declarative_snapshot"] == expected
    assert record["snapshot_reconstruction_passed"] is True
    # 真正重新读取的文件内容独立组成新库，再验证每个模块的作用域和输出。
    restored = {}
    for module in snapshot["modules"]:
        evaluate(module["program"], restored, test)
        restored[module["name"]] = module["program"]
    final, _, final_dependencies = evaluate(snapshot["program"], restored, test)
    np.testing.assert_array_equal(final, test[:, 0] + 2.5)
    assert final_dependencies == {"learned_1", "learned_2"}
    return audited, original_order, stage_dependencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=HERE / "results/formal")
    args = parser.parse_args()
    frozen = read(HERE / "protocol_workflow.json")
    plan = frozen["protocol"]
    source_counts = verify_sources(frozen)
    summary = read(args.directory / "summary.json")
    order_groups = defaultdict(list)
    records = []
    candidate_count = 0
    dependencies_by_backend = defaultdict(list)
    for seed in plan["order_seeds"]:
        pair = {}
        for backend in plan["backends"]:
            record = read(args.directory / f"{backend}_{seed}.json")
            assert record["backend"] == backend and record["order_seed"] == seed
            snapshot = read(args.directory / f"snapshot_{backend}_{seed}.json")
            count, order, dependencies = audit_case(record, snapshot, plan)
            candidate_count += count
            dependencies_by_backend[backend].append(dependencies)
            pair[backend] = record
            records.append(record)
        order_groups[order].append(seed)
        for field in ("trace", "stages", "declarative_snapshot"):
            assert pair[plan["backends"][0]][field] == pair[plan["backends"][1]][field]
    assert len(plan["order_seeds"]) == 12 and len(order_groups) == 11
    assert summary["case_count_per_backend"] == 12
    assert summary["paired_stages"] == 36
    assert summary["native_candidate_evaluations"] == candidate_count
    assert summary["all_paired_traces_match"] and summary["all_stages_solved_and_verified"]
    assert summary["native_source_hashes_before_after_equal"]
    assert summary["core_files_modified"] == {"ATF_V0": 0, "PyTensor": 0}
    # 最后再次确认未意外导入任何被测框架；本审计只读取其源文件哈希。
    assert not any(name == "ATF" or name.startswith("ATF.") or name == "pytensor"
                   or name.startswith("pytensor.") for name in sys.modules)
    report = {
        "all_checks_passed": True,
        "audit_source_sha256": sha(__file__),
        "backends_executed_during_audit": False,
        "run_count": len(records),
        "case_count_per_backend": len(plan["order_seeds"]),
        "distinct_primitive_orders": len(order_groups),
        "duplicate_order_seed_groups": [seeds for seeds in order_groups.values() if len(seeds) > 1],
        "paired_stages_checked": 3 * len(plan["order_seeds"]),
        "independently_replayed_candidate_evaluations": candidate_count,
        "disk_snapshots_independently_reconstructed": len(records),
        "stage_2_reuses_learned_1_in_every_run": True,
        "stage_3_reuses_learned_2_and_nested_learned_1_in_every_run": True,
        "frozen_native_source_files_matched": source_counts,
        "frozen_experiment_source_hashes_matched": True,
        "scope": "Deterministic workflow coverage with twelve order-seeded cases and eleven distinct primitive orders. JSON snapshots belong to shared outer logic; they are not native framework serialization. The audit verifies numeric traces and stored declarations, not all native interface probes.",
    }
    (args.directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
