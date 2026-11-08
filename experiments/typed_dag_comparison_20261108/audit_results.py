"""用独立向量算术重放每个搜索候选，并核验两后端的配对结果。

这里不调用任一被比较执行器完成预测，也不使用注册表中的原语函数。
随机提案算法按公开协议重建；数值语义、展开成本和模块血缘另行检查。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import random
import statistics

import numpy as np
from run_comparison import PLAN, HERE, canonical, digest, dump, random_program, mutate_program, task_data, project_module


def primitive(name, args):
    """独立公式表，避免两个被测后端共同调用错误原语而彼此一致。"""
    if name == "increment": return args[0] + 1
    if name == "add_3": return args[0] + 3
    if name == "multiply_by_2": return args[0] * 2
    if name == "multiply_by_05": return args[0] / 2
    if name == "identity": return args[0]
    if name == "subtract": return args[0] - args[1]
    if name == "absolute": return np.abs(args[0])
    if name == "multiply": return args[0] * args[1]
    if name == "one_minus": return 1 - args[0]
    if name == "add": return args[0] + args[1]
    raise AssertionError(f"Unknown primitive {name}")


def independent(program, registry, values):
    """递归按引用求值，每个程序调用中共享节点只计算一次。"""
    n = len(program["input_types"])
    memo = {i: (values[:, i], 0) for i in range(n)}
    def visit(ref):
        if ref in memo:
            return memo[ref][0]
        node = program["nodes"][ref-n]
        inputs = [visit(source) for source in node["inputs"]]
        entry = registry[node["method"]]
        if "body" in entry:
            answer, local_cost = independent(entry["body"], registry, np.column_stack(inputs))
        else:
            answer, local_cost = primitive(node["method"], inputs), 1
        memo[ref] = (answer, local_cost)
        return answer
    prediction = visit(program["output"])
    return prediction, sum(cost for _, cost in memo.values())


def independent_cost(program, registry):
    # 每个输入取零只用于运行独立成本遍历，不参与搜索或选模。
    return independent(program, registry, np.zeros((1, len(program["input_types"]))))[1]


def audit_run(directory):
    record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    with gzip.open(directory / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        traces = [json.loads(line) for line in stream]
    rng = random.Random(record["seed"])
    train, test, targets = task_data(record["family"])
    audited = 0
    for stage_record, target in zip(record["stages"], targets):
        registry = stage_record["registry_before"]
        for entry in registry.values():
            if "body" in entry:
                assert independent_cost(entry["body"], registry) == entry["expanded_cost"]
        stage = stage_record["stage"]
        rows = [row for row in traces if row["stage"] == stage]
        assert len(rows) == record["budget_per_stage"]
        parent, parent_loss, best, first = None, float("inf"), None, None
        rejected = 0
        for evaluation, row in enumerate(rows, 1):
            restart = parent is None or (evaluation-1) % PLAN["restart_every"] == 0
            for attempt in range(10000):
                program = (random_program(rng, train.shape[1], registry) if restart
                           else mutate_program(rng, parent, registry))
                cost = independent_cost(program, registry)
                if cost <= PLAN["max_expanded_primitive_operations"]:
                    break
                rejected += 1
            else:
                raise AssertionError("Proposal rejection loop did not terminate")
            prediction, cost_again = independent(program, registry, train)
            assert cost == cost_again
            loss = float(np.mean((prediction-target(train))**2))
            assert row["evaluation"] == evaluation and row["fingerprint"] == digest(program)
            assert row["loss"] == loss and row["expanded_ops"] == cost
            assert row["primitive_calls"] == len(train)*cost
            accepted = restart or loss <= parent_loss
            assert row["accepted"] == accepted and row["restart"] == restart
            if accepted:
                parent, parent_loss = program, loss
            if first is None and loss <= PLAN["success_mse"]:
                first = evaluation
            rank = (loss, cost, row["active_nodes"], canonical(program))
            if best is None or rank < best[0]:
                best = (rank, program)
            audited += 1
        assert best[1] == stage_record["best_program"]
        assert first == stage_record["first_training_success_evaluation"]
        assert rejected == stage_record["rejected_expanded_size_proposals"]
        final, _ = independent(best[1], registry, test)
        np.testing.assert_array_equal(final, stage_record["final_prediction"])
        test_loss = float(np.mean((final-target(test))**2))
        assert test_loss == stage_record["test_mse"]
        assert stage_record["success"] == (test_loss <= PLAN["success_mse"])
        if stage_record["module_added"]:
            module = record["registry_final"][stage_record["module_added"]]
            assert module["body"] == project_module(best[1])
            assert module["source_program_hash"] == digest(best[1])
            assert stage_record["training_success"] and record["library"] == "growing" and stage < 3
    return record, audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=HERE / "artifacts" / "formal")
    # 次级预算检查只有奇偶任务；显式指定族，防止把缺失主实验结果误当作完成。
    parser.add_argument("--families", nargs="+", choices=PLAN["families"], default=PLAN["families"])
    args = parser.parse_args()
    records, candidates = [], 0
    for path in sorted(args.directory.glob("*/result.json")):
        record, count = audit_run(path.parent)
        records.append(record)
        candidates += count
    assert len(records) == len(PLAN["formal_seeds"]) * len(args.families) * 4
    assert {r["family"] for r in records} == set(args.families)
    ratios = []
    paired = 0
    for family in args.families:
        for library in PLAN["libraries"]:
            condition_ratios = []
            for seed in PLAN["formal_seeds"]:
                pair = {r["backend"]: r for r in records if (r["family"], r["library"], r["seed"]) == (family, library, seed)}
                assert set(pair) == set(PLAN["backends"])
                for a, b in zip(pair["atf_v0"]["stages"], pair["typed_dag"]["stages"]):
                    for field in ("trace_hash", "best_program", "registry_before", "module_added", "final_prediction"):
                        assert a[field] == b[field], (family, library, seed, field)
                    paired += 1
                def accounted(r):
                    return sum(s["setup_seconds"]+s["compile_seconds"]+s["execution_seconds"]+s["admission_seconds"] for s in r["stages"])
                condition_ratios.append(accounted(pair["atf_v0"])/accounted(pair["typed_dag"]))
            ratios.append({"family": family, "library": library,
                           "median_atf_over_typed_dag_setup_compile_execute_ratio": statistics.median(condition_ratios),
                           "quartiles": np.quantile(condition_ratios, [.25, .75]).tolist()})
    summary = json.loads((args.directory / "summary.json").read_text(encoding="utf-8"))
    with (args.directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(summary["groups"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["groups"])
    report = {"runs": len(records), "independently_replayed_candidates": candidates,
              "paired_stages_with_identical_search": paired,
              "all_checks_passed": True, "timing_ratios": ratios,
              "timing_scope": "Descriptive same-host scalar Python implementation timings; reference expands modules at compile time whereas V0 EvolvedMethod calls a nested interpreter. Not a representation-level speed theorem."}
    dump(args.directory / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
