"""Post-run integrity check and scientific figure; never changes frozen data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ["MPLCONFIGDIR"] = str(HERE / "cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / "cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol = json.loads((HERE / "frozen_protocol.json").read_text(encoding="utf-8"))
manifest = json.loads((HERE / "completion_manifest.json").read_text(encoding="utf-8"))
formal = json.loads((HERE / "formal_results.json").read_text(encoding="utf-8"))
# 匿名副本与原始证据使用不同摘要；必须同时核对发布字节与原始来源映射。
# 新测量直接匹配当前摘要；历史测量仅在发布清单明确证明转换关系时接受。
release_files = json.loads((HERE.parent / "release_manifest.json").read_text(encoding="utf-8"))["files"]
def check_provenance(path, expected):
    actual = sha(path)
    if actual == expected:
        return
    relative = path.resolve().relative_to(HERE.parent.resolve()).as_posix()
    record = release_files.get(relative, {})
    if record.get("sha256") != actual or record.get("source_sha256") != expected:
        raise ValueError("Published bytes do not match the documented original: " + relative)
check_provenance(HERE / "frozen_protocol.json", manifest["protocol_sha256"])
for relative, expected in protocol["code_sha256"].items():
    check_provenance(ROOT / relative.replace("\\", "/"), expected)
for name, expected in manifest["outputs"].items():
    check_provenance(HERE / name, expected)
expected_cases = {tuple(case) for case in protocol["formal_cases"]}
actual_cases = {(row["width"], row["seed"]) for row in formal}
assert expected_cases == actual_cases and len(formal) == 30
assert not actual_cases.intersection(tuple(case) for case in protocol["pilot_cases_excluded"])
with (HERE / "per_graph.csv").open(encoding="utf-8", newline="") as f:
    per_graph = list(csv.DictReader(f))
assert len(per_graph) == 30
per_graph_map = {(int(r["width"]), int(r["seed"])): r for r in per_graph}
raw_count = 0
for row in formal:
    width = row["width"]
    assert row["operation_nodes"] == 8 * width
    assert row["total_nodes"] == 8 * width + 2
    assert row["edges"] == 17 * width
    assert row["node_function_calls_per_inference"] == 8 * width * 32
    assert row["max_abs_serial_parallel"] == 0
    assert row["max_abs_serial_reference"] < 1e-12
    for metric, values in row["samples_ms"].items():
        assert len(values) == 7 and all(v > 0 for v in values)
        assert statistics.median(values) == row[metric]
        assert float(per_graph_map[(width, row["seed"])][metric]) == row[metric]
        raw_count += len(values)
with (HERE / "timing_samples.csv").open(encoding="utf-8", newline="") as f:
    assert len(list(csv.DictReader(f))) == raw_count == 1260

widths = protocol["widths"]
nodes = np.array(widths) * protocol["depth"]


def stats(metric):
    groups = [[row[metric] for row in formal if row["width"] == w] for w in widths]
    middle = np.array([statistics.median(g) for g in groups])
    errors = np.array([[m - min(g) for m, g in zip(middle, groups)],
                       [max(g) - m for m, g in zip(middle, groups)]])
    return middle, errors


with (HERE / "summary.csv").open(encoding="utf-8", newline="") as f:
    summary = list(csv.DictReader(f))
for metric in ["serial_ms", "parallel4_ms", "construction_ms", "declared_type_pass_ms",
               "topological_preparation_ms", "processor_init_ms", "parallel_speedup"]:
    middle, _ = stats(metric)
    assert [float(row[metric + "_median"]) for row in summary] == middle.tolist()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
styles = [("serial_ms", "Native serial", "#0072B2", "o"),
          ("parallel4_ms", "Native parallel (4 threads)", "#D55E00", "s")]
for metric, label, color, marker in styles:
    middle, errors = stats(metric)
    axes[0].errorbar(nodes, middle, yerr=errors, label=label, color=color,
                     marker=marker, capsize=3, linewidth=1.5, markersize=4)
axes[0].set_title("(a) End-to-end execution")
axes[0].set_ylabel("Wall time per batch of 32 (ms)")
axes[0].set_yscale("log", base=2)
axes[0].legend(loc="upper left", fontsize=8)
for metric, label, color, marker in [
    ("construction_ms", "Graph construction", "#009E73", "o"),
    ("declared_type_pass_ms", "Declared-type pass*", "#CC79A7", "s"),
    ("topological_preparation_ms", "Topological preparation", "#E69F00", "^"),
]:
    middle, errors = stats(metric)
    axes[1].errorbar(nodes, middle, yerr=errors, label=label, color=color,
                     marker=marker, capsize=3, linewidth=1.5, markersize=4)
axes[1].set_title("(b) Graph preparation")
axes[1].set_ylabel("Wall time per graph (ms)")
axes[1].set_yscale("log", base=2)
axes[1].legend(loc="upper left", fontsize=8)
middle, errors = stats("parallel_speedup")
axes[2].errorbar(nodes, middle, yerr=errors, color="#333333", marker="o",
                 capsize=3, linewidth=1.5, markersize=4)
axes[2].axhline(1.0, linestyle="--", color="#777777", linewidth=1)
axes[2].set_title("(c) Paired parallel speedup")
axes[2].set_ylabel("Serial / 4-thread wall time (times)")
axes[2].set_ylim(0.45, 1.14)
axes[2].text(0.03, 0.93, "> 1: parallel is faster", transform=axes[2].transAxes,
             fontsize=9, va="top")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(nodes, labels=[str(x) for x in nodes])
    ax.set_xlabel("Operation nodes (fixed depth = 8)")
    ax.grid(alpha=0.2, linewidth=0.6)
fig.suptitle("ATF execution scaling: fixed binary CPU workload", fontsize=14, y=0.98)
fig.text(0.5, 0.055,
         "Points: median over 5 graph seeds, each with 2 warmups + 7 timing repeats. Bars: graph-seed min–max (not confidence intervals).",
         ha="center", fontsize=9)
fig.text(0.5, 0.014,
         "*Separate pass using native type compatibility; not a built-in compiler validator. No sleep/I/O workload. Edges = 17 × width.",
         ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
fig.savefig(HERE / "scaling_summary.png", dpi=200)
fig.savefig(HERE / "scaling_summary.svg")
plt.close(fig)

audit = {
    "all_passed": True, "formal_graphs": len(formal), "raw_timing_measurements": raw_count,
    "source_and_protocol_and_data_hashes_verified": True,
    "pilot_disjoint": True, "counts_and_medians_verified": True,
    "max_abs_serial_reference": max(row["max_abs_serial_reference"] for row in formal),
    "max_abs_parallel_reference": max(row["max_abs_parallel_reference"] for row in formal),
    "all_serial_parallel_exact": all(row["max_abs_serial_parallel"] == 0 for row in formal),
    "serial_growth_32_to_1024": stats("serial_ms")[0][-1] / stats("serial_ms")[0][0],
    "parallel_growth_32_to_1024": stats("parallel4_ms")[0][-1] / stats("parallel4_ms")[0][0],
    "speedup_statistic": "Median of paired per-graph ratios, not ratio of aggregated medians",
    "figure_error_bars": "minimum to maximum of five per-graph medians; not confidence intervals",
    "figure_sha256": {n: sha(HERE / n) for n in ["scaling_summary.png", "scaling_summary.svg"]},
}
(HERE / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
(HERE / "FIGURE_CAPTION.txt").write_text(
    "Controlled scaling of the unmodified ATF graph executor on pure scalar CPU operators. "
    "DAG depth is fixed at 8 and width varies from 4 to 128, corresponding to 32–1024 operation nodes "
    "and 68–2176 edges; batch size is 32. Every operation has two inputs and performs the same 16 bounded "
    "residual-tanh updates. (a) Native serial and four-thread end-to-end execution, including per-call graph "
    "preparation and, for parallel execution, thread-pool lifecycle. (b) Separate graph construction, declared-type "
    "compatibility and topological-order passes. The declared-type pass calls PathGenerator._is_type_compatible; "
    "it is an auxiliary benchmark, not an existing full compiler validator. (c) Speedup is the median of paired "
    "serial/parallel timing ratios. Points summarize five independent graph seeds, each using two warmups "
    "and seven timed repetitions; error bars show the min–max across per-graph medians and are not confidence "
    "intervals. Node and edge counts covary, so this experiment does not isolate edge density. "
    "For this scalar CPU workload, median four-thread speedup is below 1 at every tested size.\n",
    encoding="utf-8")
print(json.dumps(audit, indent=2))
