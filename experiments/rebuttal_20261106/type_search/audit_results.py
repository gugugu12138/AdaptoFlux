"""Post-run audit; never changes frozen protocol, runner, or raw results."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ("fine", "coarse", "flat")
frozen = json.loads((HERE/"protocol_frozen.json").read_text(encoding="utf-8"))
assert hashlib.sha256((HERE/"run_type_search.py").read_bytes()).hexdigest() == frozen["runner_sha256"]
rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE/"formal").glob("*/*/result.json"))]
assert len(rows) == 90 and all(r["status"] == "completed" for r in rows)
assert {(r["run_seed"],r["type_mode"]) for r in rows} == {(s,m) for s in frozen["formal_seeds"] for m in MODES}
catalog = [json.loads(line) for line in (HERE/"formal/candidate_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
assert sum(c["kind"] == "final" for c in catalog) == 90
audit = {"all_90_runs_complete": True, "frozen_runner_sha256_verified": True,
         "catalog_records": len(catalog), "catalog_first_unique_records": sum(c["kind"] == "first_unique" for c in catalog),
         "catalog_final_records": 90, "arms": {}, "reconstructed_acceptances": [],
         "native_log_caveat": "GraphEvoTrainer.train breaks for early stopping before adding that cycle's steps_taken to total_refinement_steps and before updating the best snapshot. Hence result.json accepted_refinements is the raw native counter, incomplete on success; model/final plus final reevaluation is authoritative, not model/best.",
         "acceptance_reconstruction": "Native random_single calls accuracy once for each initial model (3), once at each refinement phase start (cycles_completed), once per accepted substitution, and twice at final/best reporting. Thus accepted replacements = accuracy_calls - 3 - cycles_completed - 2. This follows the inspected native call sites; callback totals are independently verified against evaluation_events.jsonl.",
         "timing_caveat": "Other experiments ran concurrently; wall times are execution records and must not be used for comparative speed claims.",
         "budget_caveat": "Nominal search caps matched; observed callback/unique graph budgets differ. Prefix success tables below are post-hoc descriptive summaries, not separately rerun equal-budget trials.",
         "unique_signature_caveat": "Ordered attributed graph signatures, excluding literal node names; not semantic equivalence or isomorphism classes. Node replacement may change insertion order and execution order.",
         "scope": "An exhaustive tiny finite task with a supplied action primitive and task-semantic type prior; no external-task or held-out-generalization inference."}
for mode in MODES:
    group = [r for r in rows if r["type_mode"] == mode]
    accepted = []
    for r in group:
        n = r["accuracy_calls"] - 3 - r["cycles_completed"] - 2
        assert n >= r["accepted_refinements"] >= 0
        events_path = HERE/"formal"/mode/str(r["run_seed"])/"evaluation_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == r["callback_evaluations"]
        assert sum(e["callback_kind"] == "loss" for e in events) == r["loss_calls"]
        assert sum(e["callback_kind"] == "accuracy" for e in events) == r["accuracy_calls"]
        assert len({e["fingerprint"] for e in events}) == r["unique_candidates"]
        # Re-evaluating an attributed graph must have exactly the same outcome.
        outcomes = {}
        for e in events:
            value = (e["loss"],e["accuracy"])
            assert outcomes.setdefault(e["fingerprint"],value) == value
        # Accepted substitutions are the extra accuracy calls, after the first
        # six initialization callbacks, with nonbaseline preceding loss calls.
        # Native phases are already fixed by source inspection; no mutation.
        accepted.append(n)
        audit["reconstructed_acceptances"].append({"seed":r["run_seed"],"mode":mode,"native_logged":r["accepted_refinements"],"reconstructed":n})
    audit["arms"][mode] = {
        "native_logged_accepted_total": sum(r["accepted_refinements"] for r in group),
        "reconstructed_accepted_total":sum(accepted),
        "reconstructed_accepted_mean":statistics.mean(accepted),
        "first_success_unique_median_successful_runs": statistics.median(r["first_success"]["candidate_index"] for r in group if r["first_success"]),
        "strict_successes_by_unique_budget_posthoc": {str(b):sum(r["first_success"] is not None and r["first_success"]["candidate_index"] <= b for r in group) for b in [1,3,5,10,20]},
        "strict_successes_by_callback_budget_posthoc": {str(b):sum(r["first_success"] is not None and r["first_success"]["callback_index"] <= b for r in group) for b in [5,10,20,50,100,500,1000]},
    }
environment = {"python":platform.python_version(),"platform":platform.platform(),
               "packages":{name:importlib.metadata.version(name) for name in ["numpy","networkx","pandas","matplotlib"]},
               "core_source_hashes_collected_after_run":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"ATF/core/adaptoflux.py",ROOT/"ATF/GraphProcessor/graph_processor.py",ROOT/"ATF/PathGenerator/path_generator.py",ROOT/"ATF/ModelTrainer/GraphEvoTrainer/graph_evo_trainer.py",ROOT/"ATF/ModelTrainer/GraphEvoTrainer/refinement_strategies.py",ROOT/"experiments/minimal_stateful_comparison/env.py"]}}
(HERE/"formal/audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
(HERE/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
summary = json.loads((HERE/"formal/summary.json").read_text(encoding="utf-8"))
lines = ["Native GraphEvo type-prior replication: completed 90/90 formal runs", "", "Design", "30 fixed formal seeds 110600..110629; pilot seed 110500 excluded. Nine original operator bodies unchanged. Fine: pos_type,target_type; coarse: quantity,quantity (distinct from scalar); flat: scalar,scalar. Every callback exhaustively evaluates all 20 unequal positions/targets in {-2,-1,0,1,2}, at most four graph executions per episode. Every evaluation asserts unchanged Python/NumPy RNG states. No evaluation caching.", "", "Primary results (strict success on all 20 states)", "mode | successes | Wilson 95% interval | mean final accuracy | mean loss | mean loss callbacks | mean accuracy callbacks | mean distinct graphs"]
for mode in MODES:
    a = summary["arms"][mode]
    lo,hi = a["success_wilson_95"]
    lines.append(f'{mode} | {a["successes"]}/30 | {100*lo:.1f}%-{100*hi:.1f}% | {a["accuracy"]["mean"]:.4f} | {a["loss"]["mean"]:.4f} | {a["loss_calls"]["mean"]:.2f} | {a["accuracy_calls"]["mean"]:.2f} | {a["unique_candidates"]["mean"]:.2f}')
lines += ["", "The legacy >=95% threshold yields the same run-success counts. Initialization alone succeeds in 20/30 fine, 4/30 coarse, and 0/30 flat runs. Final full-search results therefore must not be described as entirely initialization-only.", "", "Interpretation", "Task-semantic type declarations strongly reduce the candidate space and improve success in this controlled tiny environment. Merging position/target types degrades success, showing that directional role information contributes beyond a generic numeric-type distinction. This is evidence for a supplied type prior and routing mechanism, not proof of universal search superiority or a uniquely stateful semantics.", "", "Audit caveats", audit["native_log_caveat"], audit["acceptance_reconstruction"], audit["timing_caveat"], audit["budget_caveat"], audit["unique_signature_caveat"], "", "Acceptance counts (native logged / reconstructed total)"]
for mode in MODES:
    a=audit["arms"][mode]
    lines.append(f'{mode}: {a["native_logged_accepted_total"]} / {a["reconstructed_accepted_total"]}')
lines += ["", "Artifacts", "protocol_planned.json and protocol_frozen.json: fixed design and hashes.", "validation.json: target/no-action cases and function-body checks.", "formal/per_seed.csv and formal/summary.json: all runs and aggregate results.", "formal/<mode>/<seed>/evaluation_events.jsonl: every evaluation callback with outcomes and counters.", "formal/candidate_catalog.jsonl: first 20 unique candidates per run plus final graph; node/edge iteration order, method signatures and explicit topological order retained.", "formal/<mode>/<seed>/result.json: final graph and all 20 trajectories, including failures.", "formal/audit.json: count checks, deterministic repeated-graph outcomes, acceptance reconstruction and post-hoc budget-prefix summaries.", "environment.json: package versions and source hashes.", "", "Execution", "Set PYTHONHASHSEED=0 and PYTHONDONTWRITEBYTECODE=1. Use python -B run_type_search.py prepare, pilot, freeze, formal in sequence. The runner refuses to overwrite a completed stage. All writes are confined to this new type_search directory; ATF core and ATF-nodes were not modified."]
(HERE/"RESULTS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"catalog_records":audit["catalog_records"],"arms":audit["arms"]},indent=2))
