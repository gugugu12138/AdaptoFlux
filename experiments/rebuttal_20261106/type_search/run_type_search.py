"""Isolated, exhaustive GraphEvo type-prior replication. Run with Python -B."""
from __future__ import annotations
import argparse
import ast
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "cache" / "matplotlib"))
import numpy as np
import networkx as nx
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.GraphEvoTrainer.graph_evo_trainer import GraphEvoTrainer
from experiments.minimal_stateful_comparison.env import STATE

MODES = ("fine", "coarse", "flat")
ROOT_TYPES = {"fine": ["pos_type", "target_type"], "coarse": ["quantity", "quantity"], "flat": ["scalar", "scalar"]}
PAIRS = [(p, t) for p in range(-2, 3) for t in range(-2, 3) if p != t]
CONFIG = {"verbose": False, "init_mode": "fixed", "max_init_layers": 3,
          "num_initial_models": 3, "max_refinement_steps": 30,
          "enable_evolution": False, "candidate_pool_mode": "all"}
SOURCE = ROOT / "experiments/minimal_stateful_comparison/methods_constrained.py"

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def method_file(mode):
    return HERE / "methods" / ("methods_" + mode + ".py")

def prepare():
    if (HERE / "protocol_frozen.json").exists():
        raise RuntimeError("Protocol already frozen; preparation cannot overwrite it.")
    source = SOURCE.read_text(encoding="utf-8")
    for mode in MODES:
        value = source
        if mode != "fine":
            label = "quantity" if mode == "coarse" else "scalar"
            value = value.replace("'pos_type'", repr(label)).replace("'target_type'", repr(label))
        method_file(mode).parent.mkdir(parents=True, exist_ok=True)
        method_file(mode).write_text(value, encoding="utf-8")
    protocol = {
        "experiment": "native Graphevo type-prior replication, exhaustive finite task",
        "status": "planned_before_pilot", "formal_seeds": list(range(110600, 110630)),
        "pilot_seeds": [110500], "modes": list(MODES), "root_types": ROOT_TYPES,
        "coarse_definition": "pos_type and target_type merged as quantity; quantity remains distinct from scalar",
        "operators": "All nine original operator bodies, arities, groups, weights and ordering unchanged; only type declarations/root token types differ.",
        "search_config": CONFIG, "train_config": {"max_evo_cycles": 5, "enable_early_stop": True, "early_stop_eps": 0.0, "save_model": True},
        "evaluation": {"initial_states": PAIRS, "max_steps": 4, "loss": "mean absolute final integer distance", "accuracy": "fraction reaching target within four graph executions", "success": "accuracy == 1.0 (all 20 initial states)", "legacy_secondary_success": "accuracy >= 0.95", "randomness": "none; reset STATE.pos and STATE.target explicitly for each episode; every callback reevaluates all 20 states without caching"},
        "randomness": "random.seed and numpy.random.seed reset to run seed before each arm; evaluator asserts their states unchanged; PYTHONHASHSEED=0 at interpreter startup",
        "candidate_uniqueness": "SHA256 of ordered node attrs and ordered multiedge attrs after mapping node IDs to insertion ordinals, omitting literal node IDs; not graph isomorphism or functional deduplication",
        "candidate_export": {"rule": "first 20 distinct ordered graph signatures in callback order per seed/mode, plus final graph even if repeated", "file": "<stage>/candidate_catalog.jsonl", "schema": "run_seed,type_mode,candidate_index,kind,fingerprint,callback_index,loss,accuracy,graph{nodes:[{id,attributes}],edges:[{source,target,key,attributes}],topological_order,graph_attributes,methods:[{name,attributes}],root_types,collapse_method}"},
        "budgets": "Native nominal caps retained, actual loss and accuracy callback counts plus unique graph counts reported. First success counted at first successful callback; separate first-success unique-candidate index.",
        "analysis": "Report all seeds, strict success with Wilson 95% CI, mean accuracy/loss, actual budget distributions; paired exact McNemar fine-flat and fine-coarse as exploratory comparisons.",
        "scope_limits": "Tiny fully enumerated environment; no held-out generalization claim; task-specific type semantics are a strong supplied prior; all groups retain the goal-aware move operator.",
        "source_sha256": sha(SOURCE), "method_files_sha256": {m: sha(method_file(m)) for m in MODES},
    }
    dump(HERE / "protocol_planned.json", protocol)
    print("Prepared protocol and three method files.", flush=True)

def new_model(mode):
    return AdaptoFlux(values=np.array([[0.0, 0.0]], dtype=np.float32), labels=None,
                     methods_path=str(method_file(mode)), input_types_list=ROOT_TYPES[mode])

def evaluate(model, traces=False):
    before = random.getstate()
    before_np = np.random.get_state()
    records = []
    for pos, target in PAIRS:
        STATE.pos, STATE.target = pos, target
        trajectory = [pos]
        for step in range(4):
            model.infer_with_graph(STATE.get_obs())
            trajectory.append(STATE.pos)
            if STATE.is_success():
                break
        distance = abs(int(round(STATE.pos)) - int(round(STATE.target)))
        records.append({"pos": pos, "target": target, "distance": distance,
                        "success": STATE.is_success(), "steps": len(trajectory)-1,
                        "trajectory": trajectory})
    assert before == random.getstate(), "Evaluation altered Python search RNG."
    after_np = np.random.get_state()
    assert before_np[0] == after_np[0] and np.array_equal(before_np[1], after_np[1]) and before_np[2:] == after_np[2:], "Evaluation altered NumPy search RNG."
    loss = sum(r["distance"] for r in records) / len(PAIRS)
    accuracy = sum(r["success"] for r in records) / len(PAIRS)
    return loss, accuracy, records if traces else None

def graph_record(model, mode):
    graph = model.graph
    return {"nodes": [{"id": n, "attributes": dict(d)} for n, d in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, "key": k, "attributes": dict(d)} for u, v, k, d in graph.edges(keys=True, data=True)],
            "topological_order": list(nx.topological_sort(graph)), "graph_attributes": dict(graph.graph),
            "methods": [{"name": name, "attributes": {k:v for k,v in info.items() if k != "function"}} for name,info in model.methods.items()],
            "root_types": ROOT_TYPES[mode], "collapse_method": "SUM"}

def fingerprint(record):
    ids = {n["id"]: i for i,n in enumerate(record["nodes"])}
    value = {"nodes": [n["attributes"] for n in record["nodes"]],
             "edges": [[ids[e["source"]], ids[e["target"]], e["key"], e["attributes"]] for e in record["edges"]]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def validate():
    results = {"all_function_bodies_identical": True, "checks": []}
    bodies = {}
    for mode in MODES:
        tree = ast.parse(method_file(mode).read_text(encoding="utf-8"))
        bodies[mode] = {n.name: ast.dump(ast.Module(body=n.body, type_ignores=[])) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert len(bodies["fine"]) == 9
    assert bodies["fine"] == bodies["coarse"] == bodies["flat"]
    for mode in MODES:
        model = new_model(mode)
        loss, acc, trajectories = evaluate(model, True)
        assert acc == 0 and loss == 2.0
        results["checks"].append({"mode": mode, "case": "no_action", "loss": loss, "accuracy": acc})
        g = nx.MultiDiGraph()
        g.add_nodes_from(["root", "collapse"])
        g.add_node("direction", method_name="calc_direction", is_passthrough=False)
        g.add_node("action", method_name="move_towards_target", is_passthrough=False)
        for slot, ty in enumerate(ROOT_TYPES[mode]):
            g.add_edge("root", "direction", output_index=slot, data_coord=slot, data_type=ty, input_slot=slot)
        g.add_edge("direction", "action", output_index=0, data_coord=0, data_type="scalar", input_slot=0)
        g.add_edge("action", "collapse", output_index=0, data_coord=0, data_type="scalar", input_slot=0)
        model.graph_processor.graph = g
        loss, acc, trajectories = evaluate(model, True)
        assert loss == 0 and acc == 1 and max(r["steps"] for r in trajectories) == 4
        results["checks"].append({"mode": mode, "case": "target_graph", "loss": loss, "accuracy": acc, "trajectories": trajectories, "graph": graph_record(model, mode)})
    dump(HERE / "validation.json", results)
    return results

class Recorder:
    def __init__(self, seed, mode, catalog, events):
        self.seed, self.mode, self.catalog, self.events = seed, mode, catalog, events
        self.loss_calls = self.accuracy_calls = 0
        self.unique = {}
        self.first_success = None
        self.started = time.perf_counter()

    def callback(self, model, kind):
        if kind == "loss": self.loss_calls += 1
        else: self.accuracy_calls += 1
        graph = graph_record(model, self.mode)
        key = fingerprint(graph)
        is_new = key not in self.unique
        if is_new: self.unique[key] = len(self.unique) + 1
        loss, accuracy, _ = evaluate(model)
        event = {"run_seed": self.seed, "type_mode": self.mode,
                 "callback_index": self.loss_calls + self.accuracy_calls,
                 "callback_kind": kind, "candidate_index": self.unique[key],
                 "fingerprint": key, "loss": loss, "accuracy": accuracy,
                 "new_candidate": is_new, "loss_calls": self.loss_calls,
                 "accuracy_calls": self.accuracy_calls,
                 "elapsed_seconds": time.perf_counter() - self.started}
        self.events.write(json.dumps(event) + "\n")
        if accuracy == 1 and self.first_success is None:
            self.first_success = dict(event)
        if is_new and self.unique[key] <= 20:
            self.catalog.write(json.dumps({**event, "kind": "first_unique", "graph": graph}) + "\n")
            self.catalog.flush()
        return loss if kind == "loss" else accuracy

    def loss(self, model, inputs, target): return self.callback(model, "loss")
    def accuracy(self, model, inputs, target): return self.callback(model, "accuracy")

def one_run(seed, mode, stage, catalog):
    folder = HERE / stage / mode / str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    with (folder / "native_output.log").open("w", encoding="utf-8") as log, (folder / "evaluation_events.jsonl").open("w", encoding="utf-8") as events, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        recorder = Recorder(seed, mode, catalog, events)
        handler = logging.StreamHandler(log)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.WARNING)
        try:
            af = new_model(mode)
            trainer = GraphEvoTrainer(adaptoflux_instance=af, task_type="regression",
                custom_loss_evaluator=recorder.loss, custom_accuracy_evaluator=recorder.accuracy,
                save_dir=str(folder / "model"), **CONFIG)
            native = trainer.train(input_data=np.array([[0.0, 0.0]], dtype=np.float32),
                target=np.array([0.0]), max_evo_cycles=5, enable_early_stop=True,
                early_stop_eps=0.0, save_model=True, model_save_path=str(folder / "model"))
            model = trainer.adaptoflux
            loss, acc, traces = evaluate(model, True)
            graph = graph_record(model, mode)
            key = fingerprint(graph)
            catalog.write(json.dumps({"run_seed": seed, "type_mode": mode,
                "candidate_index": recorder.unique.get(key), "kind": "final",
                "fingerprint": key, "callback_index": recorder.loss_calls + recorder.accuracy_calls,
                "loss": loss, "accuracy": acc, "graph": graph}) + "\n")
            catalog.flush()
            result = {"run_seed": seed, "type_mode": mode, "status": "completed",
                "loss": loss, "accuracy": acc, "success": acc == 1.0,
                "legacy_success_ge_095": acc >= .95,
                "loss_calls": recorder.loss_calls, "accuracy_calls": recorder.accuracy_calls,
                "callback_evaluations": recorder.loss_calls + recorder.accuracy_calls,
                "unique_candidates": len(recorder.unique), "first_success": recorder.first_success,
                "elapsed_seconds": time.perf_counter()-start,
                "accepted_refinements": native["total_refinement_steps"],
                "native_refinement_attempts": native["total_refinement_attempts"],
                "cycles_completed": native["evo_cycles_completed"],
                "initial_loss": native["phase_results"][0]["result"]["best_loss"],
                "initial_accuracy": native["phase_results"][0]["result"]["best_accuracy"],
                "extra_final_validation_evaluations": 1, "final_graph": graph, "final_traces": traces}
        except BaseException as e:
            traceback.print_exc()
            result = {"run_seed": seed, "type_mode": mode, "status": "failed", "error": repr(e),
                "loss_calls": recorder.loss_calls, "accuracy_calls": recorder.accuracy_calls,
                "unique_candidates": len(recorder.unique), "elapsed_seconds": time.perf_counter()-start}
        finally:
            logging.getLogger().removeHandler(handler)
    dump(folder / "result.json", result)
    return result

def wilson(k, n):
    z = 1.959963984540054
    d = 1 + z*z/n
    center = (k/n + z*z/(2*n))/d
    half = z*math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/d
    return [center-half, center+half]

def summarize(results, stage):
    summary = {"stage": stage, "n_attempted": len(results), "arms": {}}
    for mode in MODES:
        rows = [r for r in results if r["type_mode"] == mode]
        good = [r for r in rows if r["status"] == "completed"]
        if not good: continue
        k, n = sum(r["success"] for r in good), len(good)
        arm = {"n_completed": n, "n_failed": len(rows)-n, "successes": k,
               "success_rate": k/n, "success_wilson_95": wilson(k,n),
               "legacy_successes_ge_095": sum(r["legacy_success_ge_095"] for r in good),
               "initial_successes": sum(r["initial_accuracy"] == 1.0 for r in good)}
        for key in ["loss", "accuracy", "loss_calls", "accuracy_calls", "callback_evaluations", "unique_candidates", "elapsed_seconds", "accepted_refinements", "native_refinement_attempts"]:
            values = [r[key] for r in good]
            arm[key] = {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
        successes = [r["first_success"] for r in good if r["first_success"]]
        arm["first_success_callback_successes_only"] = [r["callback_index"] for r in successes]
        arm["first_success_unique_candidate_successes_only"] = [r["candidate_index"] for r in successes]
        summary["arms"][mode] = arm
    summary["paired_comparisons"] = {}
    for other in ["flat", "coarse"]:
        a = {r["run_seed"]:r for r in results if r["type_mode"] == "fine" and r["status"] == "completed"}
        b = {r["run_seed"]:r for r in results if r["type_mode"] == other and r["status"] == "completed"}
        common = sorted(a.keys() & b.keys())
        ab = sum(a[s]["success"] and not b[s]["success"] for s in common)
        ba = sum(b[s]["success"] and not a[s]["success"] for s in common)
        discord = ab+ba
        p = min(1.0, 2*sum(math.comb(discord,i) for i in range(min(ab,ba)+1))/2**discord) if discord else 1.0
        summary["paired_comparisons"]["fine_vs_"+other] = {"paired_n": len(common), "fine_only": ab, "other_only": ba, "exact_mcnemar_two_sided_p": p, "multiplicity_adjusted": False}
    dump(HERE / stage / "summary.json", summary)
    fields = ["run_seed", "type_mode", "status", "loss", "accuracy", "success", "legacy_success_ge_095", "initial_accuracy", "loss_calls", "accuracy_calls", "callback_evaluations", "unique_candidates", "elapsed_seconds", "accepted_refinements", "native_refinement_attempts", "cycles_completed"]
    with (HERE/stage/"per_seed.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    return summary

def run_stage(stage):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Launch with PYTHONHASHSEED=0 for deterministic ordering.")
    stage_dir = HERE / stage
    stage_dir.mkdir(exist_ok=True)
    if (stage_dir/"candidate_catalog.jsonl").exists():
        raise RuntimeError("Existing catalog: refusing to overwrite a recorded stage.")
    if stage == "pilot":
        with (stage_dir/"validation_output.log").open("w",encoding="utf-8") as out, contextlib.redirect_stdout(out):
            validate()
        protocol = json.loads((HERE/"protocol_planned.json").read_text(encoding="utf-8"))
        seeds = protocol["pilot_seeds"]
    else:
        protocol = json.loads((HERE/"protocol_frozen.json").read_text(encoding="utf-8"))
        seeds = protocol["formal_seeds"]
        assert protocol["runner_sha256"] == sha(Path(__file__)), "Frozen runner was changed."
    for mode in MODES:
        assert sha(method_file(mode)) == protocol["method_files_sha256"][mode]
    results = []
    with (stage_dir/"candidate_catalog.jsonl").open("w",encoding="utf-8") as catalog:
        for seed in seeds:
            for mode in MODES:
                result = one_run(seed,mode,stage,catalog)
                results.append(result)
                summarize(results,stage)
                print(json.dumps({k:result.get(k) for k in ["run_seed","type_mode","status","accuracy","loss","unique_candidates","loss_calls","accuracy_calls","elapsed_seconds"]}),flush=True)
    print(json.dumps(summarize(results,stage)),flush=True)

def freeze():
    protocol = json.loads((HERE/"protocol_planned.json").read_text(encoding="utf-8"))
    pilot = json.loads((HERE/"pilot/summary.json").read_text(encoding="utf-8"))
    assert all(pilot["arms"][m]["n_completed"] == 1 and pilot["arms"][m]["n_failed"] == 0 for m in MODES)
    assert (HERE/"validation.json").exists()
    if (HERE/"protocol_frozen.json").exists(): raise RuntimeError("Protocol is already frozen.")
    protocol.update(status="frozen_after_validation_and_pilot_before_formal", runner_sha256=sha(Path(__file__)), pilot_summary_sha256=sha(HERE/"pilot/summary.json"), validation_sha256=sha(HERE/"validation.json"), frozen_unix_time=time.time())
    dump(HERE/"protocol_frozen.json",protocol)
    print("Protocol frozen; formal seed list unchanged.",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","pilot","freeze","formal"])
    action = parser.parse_args().action
    if action == "prepare": prepare()
    elif action == "freeze": freeze()
    else: run_stage(action)
