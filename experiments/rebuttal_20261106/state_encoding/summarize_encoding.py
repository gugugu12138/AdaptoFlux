"""Produce human-readable analysis without rerunning the completed experiment."""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
result_dir = HERE / "formal"
summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
with (result_dir / "candidate_results.csv").open(encoding="utf-8", newline="") as f:
    candidates = list(csv.DictReader(f))
with (result_dir / "timing_raw.csv").open(encoding="utf-8", newline="") as f:
    times = list(csv.DictReader(f))
finals = [r for r in candidates if r["kind"] == "final"]
paired = {}
for row in times:
    paired.setdefault((row["run_seed"], row["type_mode"]), {"implicit": [], "explicit": []})[row["mode"]].append(float(row["seconds"]))
ratios = [statistics.median(p["explicit"]) / statistics.median(p["implicit"]) for p in paired.values()]
by_mode = {}
for mode in ["fine", "coarse", "flat"]:
    subset = [r for r in finals if r["type_mode"] == mode]
    relevant = [p for (seed, m), p in paired.items() if m == mode]
    by_mode[mode] = {
        "final_graphs": len(subset),
        "successful_final_graphs": sum(float(r["accuracy"]) == 1. for r in subset),
        "mean_extra_state_edges": statistics.mean(int(r["extra_state_edges"]) for r in subset),
        "median_implicit_seconds_per_20_episodes": statistics.median(statistics.median(p["implicit"]) for p in relevant),
        "median_explicit_seconds_per_20_episodes": statistics.median(statistics.median(p["explicit"]) for p in relevant),
        "median_paired_explicit_over_implicit": statistics.median(statistics.median(p["explicit"]) / statistics.median(p["implicit"]) for p in relevant),
    }
analysis = {
    "candidate_records_including_repeated_final_graphs": len(candidates),
    "paired_initial_conditions": summary["checked_initial_conditions"],
    "paired_environment_step_comparisons": summary["checked_environment_steps"],
    "total_native_graph_executions_in_semantic_audit": 2 * summary["checked_environment_steps"],
    "all_matches": summary["all_matched"],
    "mean_extra_state_edges_final_graphs": statistics.mean(int(r["extra_state_edges"]) for r in finals),
    "median_paired_timing_ratio": statistics.median(ratios),
    "paired_timing_ratio_quartiles": statistics.quantiles(ratios, n=4, method="inclusive"),
    "by_type_mode": by_mode,
    "timing_context": "After native type search and scaling benchmark finished; no concurrent agent experiment computation. One warm-up, five paired randomized-order repetitions. Common adapter interface overhead included.",
}
(result_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
lines = [
    "Matched explicit/implicit state encoding — completed native-executor audit",
    "",
    "1007 candidate/final graph records, including repeated final graphs; not 1007 distinct program structures.",
    "20,140 paired graph-record/initial-state cases; 70,670 paired environment-step comparisons (141,340 native executions).",
    "All per-node numerical outputs, post-effect states, collapse outputs, full trajectories, final accuracy and loss matched.",
    "",
    "Construction",
    "The implicit arm uses the original methods and original in-memory STATE. The explicit arm receives and returns immutable (pos,target) tokens through actual native graph ports. Every state read and write is chained in the original native topological order. Each state-access node adds one state input/output port and one token edge; one final token edge returns state to the caller. No computation node is added. Pure operators and numerical edges are unchanged.",
    "Both arms execute the unchanged GraphProcessor.infer_with_graph. Explicit execution is checked with unrelated poisoned global STATE and must neither read nor modify it. Two-sample explicit batches also preserve independent states.",
    "",
    "Supplementary ordering counterexample",
    "A hand-built graph with two otherwise unordered bounded moves admits legal schedules yielding different positions (1 versus 2). A dependent state read must also be ordered. The adapter supplies that order; scalar type compatibility alone does not do so.",
    "",
    "Paired implementation timing (90 final models; 20 episodes per measurement)",
    "mode | implicit median ms | explicit median ms | median paired explicit/implicit | final-model mean added edges",
]
for mode, values in by_mode.items():
    lines.append(f"{mode} | {1000 * values['median_implicit_seconds_per_20_episodes']:.3f} | {1000 * values['median_explicit_seconds_per_20_episodes']:.3f} | {values['median_paired_explicit_over_implicit']:.3f} | {values['mean_extra_state_edges']:.3f}")
lines.extend([
    f"Across all 90 models, median paired ratio: {statistics.median(ratios):.3f}; final-model mean extra state edges: {analysis['mean_extra_state_edges_final_graphs']:.3f}.",
    "",
    "Interpretation limits",
    "This translates frozen candidates after search; identical success rates are expected and are not independent experimental evidence that the search methods perform equally. This is not an independently searched explicit-state GP/CGP baseline.",
    "The adapter is given state-access annotations and an execution order; it does not discover effects, prove general alias safety, or establish arbitrary parallel state safety.",
    "Timing includes each implementation's wrapper/dispatch, immutable token construction, state extraction, aggregation, native topological sorting and episode loop. The implicit common interface also constructs a Token per step, while the explicit route adds tuple handling and dictionary aggregation. Thus the ratio is implementation overhead under this common interface, not an intrinsic or best-possible cost of either representation.",
    "The finite task, selected first-20 candidate records, repeated final records, and one host limit generalization. Raw paired timings and by-model values are retained; no speed superiority claim is made.",
    "",
    "Artifacts: formal/protocol.json, formal/timing_protocol.json, formal/candidate_results.csv, formal/timing_raw.csv, formal/summary.json, formal/analysis.json.",
    "The semantic protocol records the runner hash before the later --timing-only convenience option was added. The timing protocol records the runner hash used for timing; the audit/translation functions were unchanged. Original ATF core and ATF-nodes were not modified.",
])
(HERE / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(analysis, indent=2))
