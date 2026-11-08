"""同一搜索器下的原生 ATF／独立类型化 DAG 与方法库增长对照。

本实验不调用新增的效果排序，也不复现某个已发表的 CGP 实现。
它使用冻结 V0 运行器和独立参考后端，固定候选生成、数据、预算与入库规则，
检验原始图执行／模块调用接口在同等搜索条件下是否产生不同结果。
所有新生成结果与历史实验分开保存；ATF-nodes 不参与运行和导入。
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V0 = REPO / "experiments" / "rebuttal_20261106" / "runtime_v0"
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atf-dag-comparison-mpl"))
sys.path.insert(0, str(V0))
import numpy as np
import networkx as nx
from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.core.evolved_method import EvolvedMethod
from reference_dag import make_evaluator, validate_program

assert Path(sys.modules[GraphProcessor.__module__].__file__).resolve() == (V0 / "ATF/GraphProcessor/graph_processor.py").resolve()

logging.getLogger().setLevel(logging.ERROR)

PLAN = {
    "experiment": "paired_typed_dag_search_and_library_control",
    "date": "2026-11-08",
    "backends": ["atf_v0", "typed_dag"],
    "libraries": ["fixed", "growing"],
    "families": ["affine", "parity"],
    "pilot_seeds": [810901, 810902],
    "formal_seeds": list(range(811000, 811030)),
    "pilot_evaluations_per_stage": 64,
    "formal_evaluations_per_stage": 256,
    "genome_slots": 8,
    "max_expanded_primitive_operations": 8,
    "restart_every": 32,
    "success_mse": 1e-12,
    "search": "One shared CGP-style proposal implementation. Fixed eight-slot feed-forward genomes; final slot is output. First candidate and every 32nd restart use a fresh genome; otherwise one function or input-reference mutation. Neutral moves accepted. Fixed evaluation count, including after success. No fitness cache. Invalid expanded-size proposals are rejected before evaluation and counted.",
    "library_admission": "After stages 1 and 2 only, promote the best training-successful active graph with at least two expanded primitives. Project only used root inputs to module ports. No oracle injection, no test feedback. Both growing arms use this exact rule; both support nested shared-node module bodies.",
    "capacity": "Eight slots and eight expanded active primitive operations for every arm. An invoked module costs its internal expanded operations, not one operation. All targets have explicit witnesses within the fixed-pool limit.",
    "data": "Affine: 17 fixed dyadic training points and 32 distinct dyadic extrapolation/interpolation checks. Parity: the complete 16-row four-bit truth table; no held-out generalization claim. All method interfaces are scalar.",
    "search_budget": "Equal realized candidate evaluations, not equal time or primitive calls; record both actual primitive calls and setup/compile/evaluation time. All three acquisition stages included in cumulative cost.",
    "primary_comparison": "ATF-growing vs typed-DAG-growing, and ATF-fixed vs typed-DAG-fixed. Library growing vs fixed is a mechanism ablation, not an ATF-specific representational advantage.",
    "scope": "Small pure-function synthesis control with a newly implemented typed-DAG baseline, not a benchmark of MT-CGP/ECGP/DreamCoder. Affine and parity are restricted task families. The common search and successful-module admission policy are experimental controls, not the historical GraphEvo trainer or its frequency extractor.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sources():
    names = ["run_comparison.py", "reference_dag.py", "reachability_audit.py"]
    result = {name: file_sha(HERE / name) for name in names}
    result["runtime_v0/ATF/GraphProcessor/graph_processor.py"] = file_sha(V0 / "ATF/GraphProcessor/graph_processor.py")
    result["runtime_v0/ATF/core/evolved_method.py"] = file_sha(V0 / "ATF/core/evolved_method.py")
    return result


def active_nodes(program):
    """只从程序输出回溯活跃节点；重复引用同一节点不会重复计数。"""
    n_inputs = len(program["input_types"])
    found = set()
    def visit(ref):
        if ref < n_inputs:
            return
        index = ref - n_inputs
        if index in found:
            return
        found.add(index)
        for source in program["nodes"][index]["inputs"]:
            visit(source)
    visit(program["output"])
    return sorted(found)


def expanded_cost(program, registry):
    """模块按内部展开成本收费；共享的外部节点仍只执行一次。"""
    return sum(registry[program["nodes"][i]["method"]]["expanded_cost"] for i in active_nodes(program))


def make_registry(family, counter):
    """两后端共享同一原语实现；独立有理数／真值表检查再核验正确性。"""
    if family == "affine":
        functions = {"increment": (1, lambda x: x + 1.), "add_3": (1, lambda x: x + 3.),
                     "multiply_by_2": (1, lambda x: x * 2.), "multiply_by_05": (1, lambda x: x * .5),
                     "identity": (1, lambda x: x)}
    else:
        functions = {"subtract": (2, lambda x, y: x-y), "absolute": (1, abs),
                     "multiply": (2, lambda x, y: x*y), "one_minus": (1, lambda x: 1.-x),
                     "add": (2, lambda x, y: x+y)}
    registry = {}
    for name, (arity, function) in functions.items():
        def counted(*args, function=function):
            counter[0] += 1
            return float(function(*args))
        registry[name] = {"function": counted, "input_types": ["scalar"] * arity,
                          "output_types": ["scalar"], "expanded_cost": 1}
    return registry


def make_graph(program):
    """把共同候选接到真实 ATF 端口；不在节点里隐藏整张候选图。"""
    n_inputs = len(program["input_types"])
    graph = nx.MultiDiGraph()
    graph.add_node("root", layer=0, output_types=program["input_types"])
    graph.add_node("collapse", layer=len(program["nodes"]) + 1)
    live = active_nodes(program)
    for index in live:
        node = program["nodes"][index]
        graph.add_node(f"n{index}", method_name=node["method"], layer=index+1, is_passthrough=False)
        for slot, source in enumerate(node["inputs"]):
            graph.add_edge("root" if source < n_inputs else f"n{source-n_inputs}", f"n{index}",
                           output_index=source if source < n_inputs else 0, input_slot=slot, data_type="scalar")
    output = program["output"]
    graph.add_edge("root" if output < n_inputs else f"n{output-n_inputs}", "collapse",
                   output_index=output if output < n_inputs else 0, data_coord=0, data_type="scalar")
    return graph


def native_methods(registry):
    """模块使用冻结 V0 的真实 EvolvedMethod；每阶段共用其惰性执行缓存。"""
    result = {}
    for name, entry in registry.items():
        metadata = {"input_types": entry["input_types"], "output_types": ["scalar"],
                    "input_count": len(entry["input_types"]), "output_count": 1,
                    "vectorized": False, "group": "comparison", "weight": 1.0}
        function = entry.get("function")
        if "body" in entry:
            function = EvolvedMethod(name, make_graph(entry["body"]), result, metadata)
        result[name] = dict(metadata, function=function)
    return result


def make_native_evaluator(program, methods):
    processor = GraphProcessor(make_graph(program), methods)
    def evaluate(values):
        return np.asarray(processor.infer_with_graph(values), dtype=float).reshape(-1)
    return evaluate


def project_module(program):
    """只封装输出的依赖图，保留内部共享；未使用输入不会成为虚假参数。"""
    old_n = len(program["input_types"])
    live = active_nodes(program)
    roots = sorted({source for i in live for source in program["nodes"][i]["inputs"] if source < old_n})
    mapping = {source: i for i, source in enumerate(roots)}
    nodes = []
    for old_index in live:
        old = program["nodes"][old_index]
        nodes.append({"method": old["method"], "inputs": [mapping[x] for x in old["inputs"]]})
        mapping[old_n + old_index] = len(roots) + len(nodes) - 1
    return {"input_types": [program["input_types"][i] for i in roots], "nodes": nodes,
            "output": mapping[program["output"]]}


def random_program(rng, n_inputs, registry):
    nodes = []
    names = list(registry)
    for index in range(PLAN["genome_slots"]):
        name = rng.choice(names)
        nodes.append({"method": name, "inputs": [rng.randrange(n_inputs+index)
                                                 for _ in registry[name]["input_types"]]})
    return {"input_types": ["scalar"] * n_inputs, "nodes": nodes,
            "output": n_inputs + len(nodes) - 1}


def mutate_program(rng, parent, registry):
    """局部方法／输入连接变异；方法元数改变时保留可保留的输入前缀。"""
    result = copy.deepcopy(parent)
    index = rng.randrange(len(result["nodes"]))
    node = result["nodes"][index]
    available = len(result["input_types"]) + index
    if rng.random() < .5:
        node["method"] = rng.choice(list(registry))
        count = len(registry[node["method"]]["input_types"])
        node["inputs"] = node["inputs"][:count]
        while len(node["inputs"]) < count:
            node["inputs"].append(rng.randrange(available))
    else:
        slot = rng.randrange(len(node["inputs"]))
        node["inputs"][slot] = rng.randrange(available)
    return result


def task_data(family):
    if family == "affine":
        train = (np.arange(-8, 9, dtype=float) * (3/8)).reshape(-1, 1)
        test = (np.arange(-31, 33, 2, dtype=float) / 4 + 1/16).reshape(-1, 1)
        assert set(train[:, 0]).isdisjoint(test[:, 0])
        targets = [lambda x: 2*(x[:, 0]+1), lambda x: 2*(x[:, 0]+1)+3,
                   lambda x: x[:, 0]+2.5]
    else:
        train = np.asarray(list(itertools.product((0., 1.), repeat=4)))
        test = train.copy()
        targets = [lambda x, width=width: (np.sum(x[:, :width], axis=1).astype(int) % 2).astype(float)
                   for width in (2, 3, 4)]
    return train, test, targets


def registry_export(registry):
    return {name: {key: value for key, value in entry.items() if key != "function"}
            for name, entry in registry.items()}


def run_one(backend, library, family, seed, budget, destination):
    """每个后端独立搜索和接受候选；没有把另一后端的最终结果复制过来。"""
    rng = random.Random(seed)
    counter = [0]
    registry = make_registry(family, counter)
    train, test, targets = task_data(family)
    result = {"backend": backend, "library": library, "family": family, "seed": seed,
              "budget_per_stage": budget, "stages": []}
    destination.mkdir(parents=True, exist_ok=True)
    all_trace = []
    run_started = time.perf_counter()
    for stage, target in enumerate(targets, 1):
        setup_start = time.perf_counter()
        methods = native_methods(registry) if backend == "atf_v0" else None
        module_cache = {}
        setup_seconds = time.perf_counter() - setup_start
        labels = target(train)
        parent, parent_loss, best = None, math.inf, None
        first_success = None
        rejected = 0
        compile_seconds = execution_seconds = 0.
        stage_calls = counter[0]
        trace = []
        unique = set()
        for evaluation in range(1, budget+1):
            restart = parent is None or (evaluation-1) % PLAN["restart_every"] == 0
            for attempt in range(10000):
                program = (random_program(rng, train.shape[1], registry) if restart
                           else mutate_program(rng, parent, registry))
                cost = expanded_cost(program, registry)
                if cost <= PLAN["max_expanded_primitive_operations"]:
                    break
                rejected += 1
            else:
                raise RuntimeError("Unable to propose a legal expanded-size candidate")
            t0 = time.perf_counter()
            # 独立后端的工厂本身做完整类型校验；原 ATF 在这里补同一校验。
            # 不让参考后端为同一候选重复校验两次，避免人为增加其构图成本。
            if backend == "atf_v0":
                validate_program(program, registry)
                evaluator = make_native_evaluator(program, methods)
            else:
                evaluator = make_evaluator(program, registry, module_cache=module_cache)
            t1 = time.perf_counter()
            before_calls = counter[0]
            prediction = evaluator(train)
            t2 = time.perf_counter()
            observed_calls = counter[0] - before_calls
            # 两后端必须实际执行同样数量的原语；禁止宏调用被当成免费计算。
            assert observed_calls == len(train) * cost, (backend, observed_calls, cost, len(train))
            loss = float(np.mean((prediction-labels)**2))
            assert np.isfinite(loss)
            compile_seconds += t1-t0
            execution_seconds += t2-t1
            fingerprint = digest(program)
            unique.add(fingerprint)
            rank = (loss, cost, len(active_nodes(program)), canonical(program))
            if best is None or rank < best[0]:
                best = (rank, copy.deepcopy(program))
            accepted = restart or loss <= parent_loss
            if accepted:
                parent, parent_loss = program, loss
            if first_success is None and loss <= PLAN["success_mse"]:
                first_success = evaluation
            trace.append({"evaluation": evaluation, "fingerprint": fingerprint, "loss": loss,
                          "expanded_ops": cost, "active_nodes": len(active_nodes(program)),
                          "primitive_calls": observed_calls, "accepted": accepted, "restart": restart})
        best_program = best[1]
        evaluator = (make_native_evaluator(best_program, methods) if backend == "atf_v0"
                     else make_evaluator(best_program, registry, module_cache=module_cache))
        final_prediction = evaluator(test)
        test_loss = float(np.mean((final_prediction-target(test))**2))
        train_calls = sum(item["primitive_calls"] for item in trace)
        stage_result = {"stage": stage, "success": bool(test_loss <= PLAN["success_mse"]),
                        "training_success": bool(best[0][0] <= PLAN["success_mse"]),
                        "training_mse": best[0][0], "test_mse": test_loss,
                        "first_training_success_evaluation": first_success,
                        "evaluations": budget, "unique_genotypes": len(unique),
                        "rejected_expanded_size_proposals": rejected,
                        "expanded_ops": best[0][1], "active_nodes": best[0][2],
                        "training_primitive_calls": train_calls,
                        "verification_primitive_calls": counter[0]-stage_calls-train_calls,
                        "registry_before": registry_export(registry), "best_program": best_program,
                        "final_prediction": final_prediction.tolist(), "trace_hash": digest(trace),
                        "setup_seconds": setup_seconds, "compile_seconds": compile_seconds,
                        "execution_seconds": execution_seconds, "module_added": None}
        # 使用训练输出决定入库；未成功的种子原样保留，不从其他种子借入模块。
        admission_start = time.perf_counter()
        if library == "growing" and stage < 3 and stage_result["training_success"] and best[0][1] >= 2:
            body = project_module(best_program)
            validate_program(body, registry)
            name = f"learned_stage_{stage}"
            registry[name] = {"body": body, "input_types": body["input_types"],
                              "output_types": ["scalar"], "expanded_cost": expanded_cost(body, registry),
                              "source_stage": stage, "source_program_hash": digest(best_program)}
            stage_result["module_added"] = name
        stage_result["admission_seconds"] = time.perf_counter()-admission_start
        result["stages"].append(stage_result)
        all_trace.extend(dict(stage=stage, **row) for row in trace)
    result["wall_seconds"] = time.perf_counter()-run_started
    result["registry_final"] = registry_export(registry)
    with gzip.open(destination / "trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in all_trace:
            stream.write(canonical(row) + "\n")
    dump(destination / "result.json", result)
    return result


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denom = 1+z*z/count
    center = (p+z*z/(2*count))/denom
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denom
    return [center-radius, center+radius]


def summarize(directory):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*/result.json"))]
    pairs = {}
    for record in records:
        pairs.setdefault((record["family"], record["library"], record["seed"]), {})[record["backend"]] = record
    mismatches = []
    paired_stages = 0
    for key, pair in pairs.items():
        if len(pair) != 2:
            continue
        for left, right in zip(pair["atf_v0"]["stages"], pair["typed_dag"]["stages"]):
            paired_stages += 1
            for field in ("trace_hash", "best_program", "final_prediction", "module_added", "registry_before",
                          "training_primitive_calls", "rejected_expanded_size_proposals"):
                if left[field] != right[field]:
                    mismatches.append({"condition": key, "stage": left["stage"], "field": field})
    groups = []
    for family, library, backend in itertools.product(PLAN["families"], PLAN["libraries"], PLAN["backends"]):
        subset = [r for r in records if (r["family"], r["library"], r["backend"]) == (family, library, backend)]
        if not subset:
            continue
        for stage in (1, 2, 3):
            rows = [r["stages"][stage-1] for r in subset]
            successes = sum(r["success"] for r in rows)
            groups.append({"family": family, "library": library, "backend": backend, "stage": stage,
                           "n": len(rows), "successes": successes, "success_wilson95": wilson(successes, len(rows)),
                           "mean_first_success_capped": statistics.mean(r["first_training_success_evaluation"] or (r["evaluations"]+1) for r in rows),
                           "mean_training_primitive_calls": statistics.mean(r["training_primitive_calls"] for r in rows),
                           "mean_compile_seconds": statistics.mean(r["compile_seconds"] for r in rows),
                           "mean_execution_seconds": statistics.mean(r["execution_seconds"] for r in rows),
                           "modules_added": sum(r["module_added"] is not None for r in rows)})
    cumulative = []
    for family, library, backend in itertools.product(PLAN["families"], PLAN["libraries"], PLAN["backends"]):
        subset = [r for r in records if (r["family"], r["library"], r["backend"]) == (family, library, backend)]
        if subset:
            cumulative.append({"family": family, "library": library, "backend": backend, "n": len(subset),
                               "mean_training_primitive_calls_all_stages": statistics.mean(sum(s["training_primitive_calls"] for s in r["stages"]) for r in subset),
                               "mean_verification_primitive_calls_all_stages": statistics.mean(sum(s["verification_primitive_calls"] for s in r["stages"]) for r in subset),
                               "mean_setup_compile_execute_admission_seconds_all_stages": statistics.mean(sum(s["setup_seconds"]+s["compile_seconds"]+s["execution_seconds"]+s["admission_seconds"] for s in r["stages"]) for r in subset),
                               "mean_wall_seconds_all_stages": statistics.mean(r["wall_seconds"] for r in subset)})
    result = {"run_count": len(records), "paired_stages": paired_stages,
              "backend_pair_mismatches": mismatches, "groups": groups,
              "cumulative_costs": cumulative,
              "interpretation": "Backend trace equality is a control result, not a synthesis superiority claim. Growing-vs-fixed differences belong to the shared library-admission/search policy."}
    dump(directory / "summary.json", result)
    return result


def check_witnesses():
    """先确认两个任务族的所有目标都落在固定原语预算内。"""
    for family in PLAN["families"]:
        train, _, targets = task_data(family)
        registry = make_registry(family, [0])
        if family == "affine":
            chains = [["increment", "multiply_by_2"], ["increment", "multiply_by_2", "add_3"],
                      ["increment", "multiply_by_2", "add_3", "multiply_by_05"]]
            programs = [{"input_types": ["scalar"], "nodes": [{"method": op, "inputs": [i]} for i, op in enumerate(chain)],
                         "output": len(chain)} for chain in chains]
        else:
            programs = []
            for width in (2, 3, 4):
                nodes, current = [], 0
                for index in range(1, width):
                    nodes.append({"method": "subtract", "inputs": [current, index]})
                    nodes.append({"method": "absolute", "inputs": [4+len(nodes)-1]})
                    current = 4+len(nodes)-1
                programs.append({"input_types": ["scalar"]*4, "nodes": nodes, "output": current})
        for program, target in zip(programs, targets):
            assert expanded_cost(program, registry) <= PLAN["max_expanded_primitive_operations"]
            reference = make_evaluator(program, registry)(train)
            native = make_native_evaluator(program, native_methods(registry))(train)
            np.testing.assert_array_equal(reference, target(train))
            np.testing.assert_array_equal(native, reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "check", "freeze", "run", "summarize"])
    parser.add_argument("--phase", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    if args.action == "prepare":
        path = HERE / "protocol_planned.json"
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == PLAN
        else:
            dump(path, PLAN)
        print("Protocol prepared before pilot.", flush=True)
    elif args.action == "check":
        check_witnesses()
        print("All primitive-only capacity witnesses match both executors.", flush=True)
    elif args.action == "freeze":
        assert (args.output / "pilot" / "summary.json").exists(), "Finish pilot before freezing code."
        payload = {"protocol": PLAN, "sources": sources(), "runtime_import": "runtime_v0/ATF",
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "networkx": nx.__version__, "platform": platform.system(),
                                   "machine": platform.machine()}}
        path = HERE / "protocol_frozen.json"
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == payload
        else:
            dump(path, payload)
        print("Formal protocol and source hashes frozen.", flush=True)
    elif args.action == "run":
        assert json.loads((HERE / "protocol_planned.json").read_text(encoding="utf-8")) == PLAN
        if args.phase == "formal":
            frozen = json.loads((HERE / "protocol_frozen.json").read_text(encoding="utf-8"))
            assert frozen["protocol"] == PLAN and frozen["sources"] == sources()
        check_witnesses()
        directory = args.output / args.phase
        seeds = PLAN[f"{args.phase}_seeds"]
        budget = PLAN[f"{args.phase}_evaluations_per_stage"]
        for family, library, seed in itertools.product(PLAN["families"], PLAN["libraries"], seeds):
            # 交替配对后端顺序，减轻固定先后顺序对耗时测量的影响。
            backends = PLAN["backends"] if seed % 2 == 0 else list(reversed(PLAN["backends"]))
            for backend in backends:
                run_dir = directory / f"{family}_{library}_{seed}_{backend}"
                if (run_dir / "result.json").exists():
                    continue
                record = run_one(backend, library, family, seed, budget, run_dir)
                print(f"{family} {library} {seed} {backend}: "
                      f"{[r['success'] for r in record['stages']]} {record['wall_seconds']:.3f}s", flush=True)
        summary = summarize(directory)
        assert not summary["backend_pair_mismatches"], summary["backend_pair_mismatches"]
        print(f"Completed {summary['run_count']} independent runs; paired stages {summary['paired_stages']}.", flush=True)
    else:
        print(json.dumps(summarize(args.output / args.phase), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
