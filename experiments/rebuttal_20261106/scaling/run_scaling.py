"""Read-only use of native ATF; all artifacts stay beside this script."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE / "cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / "cache")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import psutil
import scipy
from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.PathGenerator.path_generator import PathGenerator
from ATF.CollapseManager.collapse_functions import CollapseMethod

DEPTH = 8
WIDTHS = [4, 8, 16, 32, 64, 128]
BATCH = 32
INPUT_DIM = 4
ITERATIONS = 16
WARMUPS = 2
REPEATS = 7
WORKERS = 4
INPUT_SEED = 2026110600
ORDER_SEED = 2026110601
FORMAL_CASES = [(w, 51000 + w * 1000 + i) for w in WIDTHS for i in range(1, 6)]
PILOT_CASES = [(4, 49001), (128, 49002)]


def bounded_mix(a, b):
    """Same scalar CPU workload at every operation node; no state or sleep."""
    x = 0.6 * float(a) + 0.4 * float(b)
    for _ in range(ITERATIONS):
        x = 0.98 * x + 0.02 * math.tanh(0.9 * x + 0.05)
    return x


METHODS = {"bounded_mix": {
    "function": bounded_mix, "input_count": 2, "output_count": 1,
    "input_types": ["scalar", "scalar"], "output_types": ["float64"],
    "vectorized": False,
}}


def make_graph(width, seed):
    rng = np.random.default_rng(seed)
    graph = nx.MultiDiGraph()
    graph.add_node("root", output_types=["float64"] * INPUT_DIM)
    graph.add_node("collapse", input_type="scalar")
    for layer in range(DEPTH):
        permutation = rng.permutation(width)
        shift = int(rng.integers(1, width))
        for col in range(width):
            node = f"{layer}_{col}_bounded_mix"
            graph.add_node(node, method_name="bounded_mix", layer=layer,
                           is_passthrough=False)
            if layer == 0:
                src_indices = rng.choice(INPUT_DIM, 2, replace=False)
            else:
                src_indices = [permutation[col], permutation[(col + shift) % width]]
            for slot, source_index in enumerate(src_indices):
                source = "root" if layer == 0 else f"{layer - 1}_{source_index}_bounded_mix"
                graph.add_edge(source, node, output_index=int(source_index) if layer == 0 else 0,
                               input_slot=slot)
    for col in range(width):
        graph.add_edge(f"{DEPTH - 1}_{col}_bounded_mix", "collapse",
                       output_index=0, data_coord=col)
    return graph


def check_declared_types(graph, generator):
    """Independent benchmark pass, not an existing GraphProcessor validator."""
    checked = 0
    for source, target, edge in graph.edges(data=True):
        source_types = (graph.nodes[source]["output_types"] if source == "root"
                        else METHODS[graph.nodes[source]["method_name"]]["output_types"])
        expected = (graph.nodes[target]["input_type"] if target == "collapse"
                    else METHODS[graph.nodes[target]["method_name"]]["input_types"][edge["input_slot"]])
        if not generator._is_type_compatible(source_types[edge["output_index"]], expected):
            raise AssertionError(f"Incompatible edge: {source} -> {target}")
        checked += 1
    return checked


def reference(graph, values, width):
    """Independent vectorized NumPy evaluator: no native executor/function calls."""
    outputs = {"root": np.asarray(values)}
    for layer in range(DEPTH):
        for col in range(width):
            node = f"{layer}_{col}_bounded_mix"
            incoming = sorted(graph.in_edges(node, data=True), key=lambda e: e[2]["input_slot"])
            left, right = [outputs[src][:, edge["output_index"]] for src, _, edge in incoming]
            x = 0.6 * left + 0.4 * right
            for _ in range(ITERATIONS):
                x = 0.98 * x + 0.02 * np.tanh(0.9 * x + 0.05)
            outputs[node] = x[:, None]
    return np.mean(np.column_stack([outputs[f"{DEPTH-1}_{col}_bounded_mix"][:, 0]
                                   for col in range(width)]), axis=1)


def values_for_all_cases():
    return np.random.default_rng(INPUT_SEED).uniform(-1.0, 1.0, (BATCH, INPUT_DIM)).tolist()


def topology(graph):
    return [node for node in nx.topological_sort(graph) if node not in {"root", "collapse"}]


def validate(graph, processor, generator, values, width):
    assert nx.is_directed_acyclic_graph(graph)
    assert graph.number_of_nodes() == DEPTH * width + 2
    assert graph.number_of_edges() == 2 * DEPTH * width + width
    assert nx.dag_longest_path_length(graph) == DEPTH + 1
    for node in graph.nodes:
        if node not in {"root", "collapse"}:
            assert sorted(d["input_slot"] for _, _, d in graph.in_edges(node, data=True)) == [0, 1]
            assert nx.has_path(graph, "root", node) and nx.has_path(graph, node, "collapse")
    assert check_declared_types(graph, generator) == graph.number_of_edges()
    assert generator._is_type_compatible("float64", "scalar")
    assert not generator._is_type_compatible("unrelated_record", "scalar")
    expected = reference(graph, values, width)
    serial = processor.infer_with_graph(values)
    parallel = processor.infer_with_task_parallel(values, num_workers=WORKERS)
    assert serial.shape == parallel.shape == expected.shape == (BATCH,)
    assert np.all(np.isfinite(serial)) and np.all(np.abs(serial) <= 1.0)
    np.testing.assert_allclose(serial, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel, expected, rtol=1e-12, atol=1e-12)
    return {"max_abs_serial_reference": float(np.max(np.abs(serial - expected))),
            "max_abs_parallel_reference": float(np.max(np.abs(parallel - expected))),
            "max_abs_serial_parallel": float(np.max(np.abs(serial - parallel))),
            "output_sha256": hashlib.sha256(serial.tobytes()).hexdigest()}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    files = [Path(__file__), ROOT / "ATF/GraphProcessor/graph_processor.py",
             ROOT / "ATF/PathGenerator/path_generator.py",
             ROOT / "ATF/CollapseManager/collapse_functions.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def save_json(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def environment():
    cpu_name = subprocess.run(["powershell", "-NoProfile", "-Command",
                               "(Get-CimInstance Win32_Processor).Name"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": cpu_name or platform.processor(), "logical_cpus": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total, "numpy": np.__version__,
            "networkx": nx.__version__, "scipy": scipy.__version__, "psutil": psutil.__version__,
            "cpu_percent_before": psutil.cpu_percent(interval=1),
            "process_affinity": psutil.Process().cpu_affinity(),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
            "gc_enabled": gc.isenabled()}


def protocol():
    return {"version": "1.0", "freeze_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "depth": DEPTH, "widths": WIDTHS, "formal_cases": FORMAL_CASES,
            "pilot_cases_excluded": PILOT_CASES, "batch_size": BATCH, "input_dim": INPUT_DIM,
            "input_seed": INPUT_SEED, "case_order_seed": ORDER_SEED, "warmups": WARMUPS,
            "repetitions": REPEATS, "workers": WORKERS, "arity": 2,
            "operator": "x=0.6*a+0.4*b; repeat 16 times: x=0.98*x+0.02*tanh(0.9*x+0.05)",
            "collapse": "native AVERAGE over every final-layer node", "vectorized": False,
            "graph_distribution": "8 layers, random distinct predecessor pair; slot 0 is a permutation, ensuring every previous-layer node is used; first-layer edges select 2 of 4 fixed inputs",
            "counts": "N_operation=8*width; N_total=N_operation+2; E=2*N_operation+width",
            "timings": {
                "construction_ms": "NetworkX graph allocation, seeded random wiring and attributes; excludes processor construction",
                "processor_init_ms": "Native GraphProcessor constructor: graph copy, layer scan, collapse manager",
                "declared_type_pass_ms": "Independent per-edge pass calls native PathGenerator._is_type_compatible using source/target declarations; includes final collapse contract. Not a built-in precompiled validator",
                "topological_preparation_ms": "Standalone NetworkX topological_sort plus root/collapse filtering; not deducted from inference",
                "serial_ms": "Unmodified infer_with_graph: input normalization, topology sort, routing, scalar operations, collapse and output array",
                "parallel4_ms": "Unmodified infer_with_task_parallel: normalization, dependency setup, thread-pool creation/destruction, scheduling/locking, routing, scalar operations, collapse and output array"},
            "timing_design": "2 warmups per metric, 7 individual calls; shuffled metric order each repetition, shuffled cases. Wall-clock perf_counter_ns; ordinary GC remains enabled; no instrumented tracing during timings. Median per graph then median/min/max across 5 graph seeds",
            "memory_design": "One separately traced construction, serial inference and parallel inference per graph. tracemalloc incremental Python-allocation peak, not RSS or full native allocations; existing graph/processor/input are excluded from inference peaks",
            "validation": "DAG, depth, node/edge counts, every node reaches output, contiguous input slots, all declared edges compatible, negative compatibility control, serial/parallel vs independent NumPy reference at rtol=atol=1e-12",
            "limits": ["Node and edge counts covary at fixed arity; no isolated edge-density claim",
                       "Pure scalar CPU workload on one host; CPython thread/GIL and scheduling overhead can make parallel slower",
                       "Execution scaling only: not graph-search scaling, type-system soundness, arbitrary stateful-operation safety or accelerator throughput",
                       "Five seeds measure graph variability; host is not a dedicated isolated machine"],
            "code_sha256": code_hashes(), "environment": environment()}


def memory_peak(fn):
    gc.collect()
    tracemalloc.start()
    try:
        value = fn()
        _, peak = tracemalloc.get_traced_memory()
        del value
    finally:
        tracemalloc.stop()
    return peak


def run_case(width, seed, values, timings):
    graph = make_graph(width, seed)
    processor = GraphProcessor(graph, METHODS, collapse_method=CollapseMethod.AVERAGE)
    generator = PathGenerator(graph, METHODS)
    verification = validate(graph, processor, generator, values, width)
    result = {"width": width, "seed": seed, "operation_nodes": DEPTH * width,
              "total_nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
              "node_function_calls_per_inference": DEPTH * width * BATCH, **verification}
    if not timings:
        return result
    tasks = {
        "construction_ms": lambda: make_graph(width, seed),
        "processor_init_ms": lambda: GraphProcessor(graph, METHODS, collapse_method=CollapseMethod.AVERAGE),
        "declared_type_pass_ms": lambda: check_declared_types(graph, generator),
        "topological_preparation_ms": lambda: topology(graph),
        "serial_ms": lambda: processor.infer_with_graph(values),
        "parallel4_ms": lambda: processor.infer_with_task_parallel(values, num_workers=WORKERS),
    }
    for fn in tasks.values():
        for _ in range(WARMUPS):
            fn()
    samples = {name: [] for name in tasks}
    rng = random.Random(seed + ORDER_SEED)
    for repetition in range(REPEATS):
        order = list(tasks)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            value = tasks[name]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            samples[name].append(elapsed)
            del value
    result.update({name: statistics.median(v) for name, v in samples.items()})
    result["parallel_speedup"] = result["serial_ms"] / result["parallel4_ms"]
    result["construction_peak_python_bytes"] = memory_peak(tasks["construction_ms"])
    result["serial_peak_python_bytes"] = memory_peak(tasks["serial_ms"])
    result["parallel4_peak_python_bytes"] = memory_peak(tasks["parallel4_ms"])
    result["samples_ms"] = samples
    return result


def write_results(results, frozen, elapsed, cpu_seconds):
    scalar_rows = [{k: v for k, v in row.items() if k != "samples_ms"} for row in results]
    scalar_rows.sort(key=lambda r: (r["width"], r["seed"]))
    with (HERE / "per_graph.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    raw_rows = [{"width": row["width"], "seed": row["seed"], "metric": metric,
                 "repetition": i + 1, "milliseconds": value}
                for row in results for metric, values in row["samples_ms"].items()
                for i, value in enumerate(values)]
    with (HERE / "timing_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    metrics = [k for k in scalar_rows[0] if k.endswith("_ms") or k.endswith("_bytes") or k == "parallel_speedup"]
    summary = []
    for width in WIDTHS:
        cases = [r for r in scalar_rows if r["width"] == width]
        row = {"width": width, "operation_nodes": DEPTH * width, "edges": 17 * width, "graph_seeds": len(cases)}
        for metric in metrics:
            values = [r[metric] for r in cases]
            row.update({f"{metric}_median": statistics.median(values),
                        f"{metric}_min": min(values), f"{metric}_max": max(values)})
        summary.append(row)
    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    hashes_after = code_hashes()
    assert frozen["code_sha256"] == hashes_after, "Source changed during formal run"
    manifest = {"complete": True, "protocol_sha256": digest(HERE / "frozen_protocol.json"),
                "code_sha256_after": hashes_after, "formal_graphs": len(results),
                "all_validated": True, "wall_seconds": elapsed, "process_cpu_seconds": cpu_seconds,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "outputs": {name: digest(HERE / name) for name in
                            ["per_graph.csv", "timing_samples.csv", "summary.csv", "formal_results.json"]}}
    save_json("completion_manifest.json", manifest)
    lines = ["ATF controlled execution scaling — supplementary experiment", "",
             f"Complete: {len(results)} validated graphs, 5 independent seeds per width; 2 warmups and 7 repeats per metric.",
             "All times below are medians across graph-level repetition medians; milliseconds for batch 32.",
             "Depth 8, fixed binary scalar CPU operator, 16 tanh iterations per invocation; 4 worker threads.",
             f"Host: {frozen['environment']['processor']}; Python {platform.python_version()}; NetworkX {nx.__version__}.",
             "", "Ops | Edges | Build ms | Type pass ms | Topology ms | Serial ms [seed min,max] | Parallel4 ms [seed min,max] | Speedup",]
    for r in summary:
        lines.append(f"{r['operation_nodes']:4} | {r['edges']:5} | {r['construction_ms_median']:.3f} | "
                     f"{r['declared_type_pass_ms_median']:.3f} | {r['topological_preparation_ms_median']:.3f} | "
                     f"{r['serial_ms_median']:.3f} [{r['serial_ms_min']:.3f},{r['serial_ms_max']:.3f}] | "
                     f"{r['parallel4_ms_median']:.3f} [{r['parallel4_ms_min']:.3f},{r['parallel4_ms_max']:.3f}] | "
                     f"{r['parallel_speedup_median']:.3f}x")
    lines += ["", "Validity and scope:",
              "- Every graph is acyclic, has exactly the specified counts/depth, and all nodes feed the final output.",
              "- Native serial and native parallel outputs match an independently implemented vectorized NumPy reference at 1e-12 tolerance.",
              "- The type pass uses native PathGenerator._is_type_compatible on declared source/target types. This is a separate benchmark pass, not an existing full compiler validator. A deliberately incompatible signature is rejected.",
              "- Serial timing includes its per-call topological sort. Parallel timing includes per-call dependency bookkeeping and creation/destruction of the thread pool. The separate topology timing is not deducted from either.",
              "- Graph construction excludes the native processor's graph copy/layer scan; processor_init_ms is separately recorded in CSV.",
              "- Edges increase with nodes at fixed arity: this does not isolate edge density. Depth, batch and numerical workload stay fixed.",
              "- Python scalar CPU work holds the GIL; slower threaded execution is a meaningful result. No artificial sleep or I/O workload was used.",
              "- tracemalloc peaks are measured in separate calls and exclude preexisting inference inputs/graph; they are Python allocation peaks, not total process RAM.",
              "- Graph variability across 5 seeds and timing repetition dispersion are available. This is one shared host; no cross-machine or universally linear scaling claim is justified.",
              "- No claim about search scalability, arbitrary side effects, or type-system soundness follows from this benchmark.",
              "", f"Formal phase wall time: {elapsed:.2f} s; process CPU: {cpu_seconds:.2f} s (includes validation and separately traced calls).",
              "Protocol and source SHA-256 were frozen before formal cases; completion_manifest.json verifies source hashes stayed unchanged.",
              "Pilot cases are disjoint from formal cases. Formal results were not used to tune the workload or protocol.",
              "No ATF core, prior experiment, or ATF-nodes file was modified."]
    (HERE / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "formal"])
    args = parser.parse_args()
    values = values_for_all_cases()
    if args.mode == "pilot":
        results = [run_case(w, seed, values, timings=False) for w, seed in PILOT_CASES]
        save_json("pilot.json", {"code_sha256": code_hashes(), "validated": results,
                                 "purpose": "correctness and interface validation only; excluded from formal dataset"})
        print(json.dumps({"pilot_validated": len(results), "results": results}, indent=2), flush=True)
        return
    if (HERE / "frozen_protocol.json").exists():
        raise RuntimeError("Formal protocol already exists; refusing to overwrite a formal experiment")
    pilot = json.loads((HERE / "pilot.json").read_text(encoding="utf-8"))
    assert pilot["code_sha256"] == code_hashes(), "Repeat pilot after changing benchmark code"
    frozen = protocol()
    save_json("frozen_protocol.json", frozen)
    print("FROZEN", digest(HERE / "frozen_protocol.json"), flush=True)
    cases = FORMAL_CASES.copy()
    random.Random(ORDER_SEED).shuffle(cases)
    results = []
    start = time.perf_counter()
    cpu_start = time.process_time()
    for i, (width, seed) in enumerate(cases, start=1):
        result = run_case(width, seed, values, timings=True)
        results.append(result)
        save_json("formal_results.json", results)
        print(f"[{i:02}/30] width={width}, seed={seed}, serial={result['serial_ms']:.3f} ms, parallel4={result['parallel4_ms']:.3f} ms", flush=True)
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    write_results(results, frozen, elapsed, cpu_seconds)


if __name__ == "__main__":
    main()
