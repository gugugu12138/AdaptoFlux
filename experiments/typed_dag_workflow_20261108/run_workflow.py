"""比较真实 ATF / PyTensor 系统如何接入发现、入库和再次组合的流程。

这是一项接口与可执行工作流对照，不用微型任务的成功率声称算法排名。
共同枚举器为双方分别提出相同候选；每次损失都由对应原生后端实际计算。
所有模块通过双方的原生封装接口生成，不以另一个执行器代算。
"""
from __future__ import annotations

import bootstrap
from bootstrap import HERE, V0
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from atf_backend import ATFBackend
from pytensor_backend import PyTensorBackend

PLAN = {
    "study": "concrete_typed_graph_workflow_integration",
    "date": "2026-11-08",
    "backends": ["atf_v0", "pytensor"],
    "pytensor_version": "3.3.8",
    "order_seeds": list(range(811100, 811112)),
    "max_top_level_calls": 4,
    "max_expanded_primitive_calls": 8,
    "max_evaluations_per_stage": 512,
    "success_mse": 1e-12,
    "task_chain": ["2*(x+1)", "2*(x+1)+3", "x+2.5"],
    "candidate_policy": "Enumerate unary method chains by length; primitive order is shuffled once per case, learned methods append to the same registry. All candidates are evaluated by the corresponding native backend. Stop a stage at its first training solution.",
    "admission": "Wrap the successful graph with native EvolvedMethod or OpFromGraph and append it to the method registry after stages 1 and 2. No verification feedback affects admission.",
    "capacity": "Both primitive-only and growing libraries can express all targets within four top-level and eight expanded calls. This study uses growing libraries to exercise the integration path, not to claim a search improvement over a fixed library.",
    "data": "17 dyadic training values and 32 disjoint verification values; no task-role type labels.",
    "purpose": "Workflow/API coverage. Twelve method-order permutations are coverage cases, not a task-level statistical sample or a state-of-the-art synthesis benchmark.",
    "shared_outer_logic": ["candidate enumeration", "training loss", "module admission policy", "declarative graph JSON persistence"],
    "primary_execution": "Pure numerical scalar programs. Object/vector/action interface boundaries are separate native-API probes.",
}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    """核心文件逐一哈希，不能仅依靠‘没有手动编辑’来报告零核心改动。"""
    import pytensor
    package = Path(pytensor.__file__).resolve().parent
    records = {}
    for label, root in [("ATF_V0", V0 / "ATF"), ("PyTensor", package)]:
        for path in sorted(root.rglob("*.py")):
            records[f"{label}/{path.relative_to(root).as_posix()}"] = sha(path)
    return records


def chain_program(names):
    return {"input_types": ["scalar"],
            "nodes": [{"method": name, "inputs": [index]} for index, name in enumerate(names)],
            "output": len(names)}


def task_data():
    train = (np.arange(-8, 9, dtype=float) * 3/8).reshape(-1, 1)
    test = (np.arange(-31, 33, 2, dtype=float)/4+1/16).reshape(-1, 1)
    assert not set(train[:, 0]) & set(test[:, 0])
    targets = [lambda x: 2*(x[:, 0]+1), lambda x: 2*(x[:, 0]+1)+3,
               lambda x: x[:, 0]+2.5]
    return train, test, targets


def one_case(backend_class, seed, snapshot_path):
    backend = backend_class()
    names = ["increment", "multiply_by_2", "add_3", "multiply_by_05", "identity"]
    random.Random(seed).shuffle(names)
    costs = {name: 1 for name in names}
    train, test, targets = task_data()
    modules, stages, trace = [], [], []
    started = time.perf_counter()
    for stage, target in enumerate(targets, 1):
        evaluations, solution = 0, None
        for depth in range(1, PLAN["max_top_level_calls"]+1):
            for chain in itertools.product(names, repeat=depth):
                expanded = sum(costs[name] for name in chain)
                if expanded > PLAN["max_expanded_primitive_calls"]:
                    continue
                program = chain_program(chain)
                evaluate = backend.compile(program)
                prediction = evaluate(train)
                loss = float(np.mean((prediction-target(train))**2))
                evaluations += 1
                trace.append({"stage": stage, "evaluation": evaluations,
                              "chain": list(chain), "mse": loss, "expanded_calls": expanded})
                if loss <= PLAN["success_mse"]:
                    solution = (program, evaluate, expanded)
                    break
                if evaluations == PLAN["max_evaluations_per_stage"]:
                    break
            if solution is not None or evaluations == PLAN["max_evaluations_per_stage"]:
                break
        assert solution is not None, (backend.name, seed, stage)
        program, evaluate, expanded = solution
        verified = evaluate(test)
        np.testing.assert_array_equal(verified, target(test))
        record = {"stage": stage, "evaluations": evaluations, "program": program,
                  "expanded_calls": expanded, "verification_prediction": verified.tolist(),
                  "verification_mse": float(np.mean((verified-target(test))**2)),
                  "module_dependencies": [n["method"] for n in program["nodes"] if n["method"].startswith("learned_")],
                  "module_added": None}
        if stage < 3:
            name = f"learned_{stage}"
            backend.register(name, program)
            names.append(name)
            costs[name] = expanded
            modules.append({"name": name, "program": copy.deepcopy(program)})
            record["module_added"] = name
        stages.append(record)
    # 保存并重读共同声明式规格后，使用全新的后端重建原生模块和任务图。
    # 这里不调用任何框架的原生对象序列化，因而不能报告为原生保存功能。
    dump(snapshot_path, {"modules": modules, "program": stages[-1]["program"]})
    spec = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    restored = backend_class()
    for module in spec["modules"]:
        restored.register(module["name"], module["program"])
    np.testing.assert_array_equal(restored.compile(spec["program"])(test), targets[-1](test))
    return {"backend": backend.name, "order_seed": seed, "stages": stages,
            "trace": trace, "declarative_snapshot": spec, "snapshot_reconstruction_passed": True,
            "wall_seconds": time.perf_counter()-started}


def common_contracts(backend_class):
    """同一组共享/嵌套/局部替换图同时在两个原生后端上运行。"""
    backend = backend_class()
    values = np.arange(-2, 3, dtype=float).reshape(-1, 1)
    shared = {"input_types": ["scalar"], "nodes": [
        {"method": "increment", "inputs": [0]},
        {"method": "add", "inputs": [1, 1]}], "output": 2}
    np.testing.assert_array_equal(backend.compile(shared)(values), 2*(values[:, 0]+1))
    backend.register("shared", shared)
    nested = chain_program(["shared", "shared"])
    backend.register("nested", nested)
    call = chain_program(["nested"])
    expected = 4*values[:, 0]+6
    np.testing.assert_array_equal(backend.compile(call)(values), expected)
    old = backend.compile(shared)
    replacement = copy.deepcopy(shared)
    replacement["nodes"][0]["method"] = "add_3"
    np.testing.assert_array_equal(backend.compile(replacement)(values), 2*(values[:, 0]+3))
    np.testing.assert_array_equal(old(values), 2*(values[:, 0]+1))
    return {"shared_fanout": True, "native_nested_modules": True,
            "local_replacement_and_recompile": True, "original_compiled_graph_preserved": True,
            "samples": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "formal"], default="smoke")
    args = parser.parse_args()
    import pytensor
    assert pytensor.__version__ == PLAN["pytensor_version"]
    before = source_hashes()
    files = ["bootstrap.py", "run_workflow.py", "atf_backend.py", "pytensor_backend.py"]
    frozen = {"protocol": PLAN, "experiment_source_hashes": {n: sha(HERE/n) for n in files},
              "native_source_hashes": before,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pytensor": pytensor.__version__, "platform": platform.system()}}
    if args.phase == "formal":
        path = HERE / "protocol_workflow.json"
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == frozen
        else:
            dump(path, frozen)
    seeds = PLAN["order_seeds"] if args.phase == "formal" else PLAN["order_seeds"][:1]
    directory = HERE / "results" / args.phase
    records = []
    for seed in seeds:
        classes = [ATFBackend, PyTensorBackend] if seed % 2 == 0 else [PyTensorBackend, ATFBackend]
        pair = {}
        for backend_class in classes:
            path = directory / f"{backend_class.name}_{seed}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                record = one_case(backend_class, seed, directory / f"snapshot_{backend_class.name}_{seed}.json")
                dump(path, record)
            records.append(record)
            pair[backend_class.name] = record
        for field in ("trace", "stages", "declarative_snapshot"):
            assert pair["atf_v0"][field] == pair["pytensor"][field], (seed, field)
        print(f"Completed paired method-order case {seed}", flush=True)
    contracts = {b.name: common_contracts(b) for b in (ATFBackend, PyTensorBackend)}
    probes = {b.name: b().probes() for b in (ATFBackend, PyTensorBackend)}
    after = source_hashes()
    assert before == after
    summary = {"case_count_per_backend": len(seeds), "paired_stages": 3*len(seeds),
               "native_candidate_evaluations": sum(len(r["trace"]) for r in records),
               "all_paired_traces_match": True, "all_stages_solved_and_verified": True,
               "common_contracts": contracts, "native_interface_probes": probes,
               "core_files_modified": {"ATF_V0": 0, "PyTensor": 0},
               "native_source_hashes_before_after_equal": True,
               "scope": PLAN["purpose"]}
    dump(directory / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("case_count_per_backend", "paired_stages", "native_candidate_evaluations", "core_files_modified")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
