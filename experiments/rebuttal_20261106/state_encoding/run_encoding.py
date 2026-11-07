"""Matched state-encoding audit using the unchanged native GraphProcessor.

Run with the workspace Python and -B. All outputs go beside this script.
This compares encodings of frozen candidates, NOT two independent searches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import logging
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import networkx as nx
import numpy as np
from ATF.GraphProcessor.graph_processor import GraphProcessor
from experiments.minimal_stateful_comparison import methods_constrained as original
from experiments.minimal_stateful_comparison.env import STATE

logging.disable(logging.CRITICAL)
EFFECTS = {"read_current_pos", "read_target", "move_towards_target"}
INITIAL = [(p, t) for p in range(-2, 3) for t in range(-2, 3) if p != t]


@dataclass(frozen=True)
class Token:
    pos: float
    target: float


def transition(name, args, state):
    """Pure state transition: no reference to the implicit environment."""
    if name == "read_current_pos":
        return float(state.pos), state
    if name == "read_target":
        return float(state.target), state
    if name == "move_towards_target":
        diff = args[0]
        pos = state.pos
        if diff > 0.5:
            pos = max(-2, pos - 1)
        elif diff < -0.5:
            pos = min(2, pos + 1)
        return float(pos), Token(pos, state.target)
    raise ValueError(name)


def unpack(payload):
    graph = nx.MultiDiGraph(**payload.get("graph_attributes", {}))
    for item in payload["nodes"]:
        graph.add_node(item["id"], **item["attributes"])
    for item in payload["edges"]:
        graph.add_edge(item["source"], item["target"], key=item["key"], **item["attributes"])
    assert list(nx.topological_sort(graph)) == payload["topological_order"]
    return graph


def build(payload, explicit, trace=None):
    graph = unpack(payload)
    methods = {item["name"]: dict(item["attributes"], function=getattr(original, item["name"]))
               for item in payload["methods"]}
    effect_order = [node for node in payload["topological_order"]
                    if graph.nodes[node].get("method_name") in EFFECTS]
    if explicit:
        # A token traverses EVERY state read and write in the original serial order.
        # Chaining writes alone would fail to preserve read-after-write semantics.
        previous, output_index = "root", 2
        for node in effect_order:
            name = graph.nodes[node]["method_name"]
            incoming = list(graph.in_edges(node, keys=True, data=True))
            if all(edge[3].get("input_slot") is None for edge in incoming):
                for slot, (_, _, _, data) in enumerate(incoming):
                    data["input_slot"] = slot
            graph.add_edge(previous, node, output_index=output_index,
                           input_slot=methods[name]["input_count"],
                           data_type="state_token", encoding_edge=True)
            previous, output_index = node, 1
        coords = [data["data_coord"] for _, _, data in graph.in_edges("collapse", data=True)]
        graph.add_edge(previous, "collapse", output_index=output_index,
                       data_coord=max(coords, default=-1) + 1,
                       data_type="state_token", encoding_edge=True)
        for name in EFFECTS:
            info = methods[name]
            info["input_count"] += 1
            info["output_count"] += 1
            info["input_types"] = info["input_types"] + ["state_token"]
            info["output_types"] = info["output_types"] + ["state_token"]

            def explicit_op(*args, _name=name):
                return transition(_name, args[:-1], args[-1])

            info["function"] = explicit_op

    if trace is not None:
        # Per-node instrumentation is used only for semantic validation, never timing.
        for node, data in graph.nodes(data=True):
            name = data.get("method_name")
            if name not in methods:
                continue
            entry = methods[name].copy()
            fn = entry["function"]

            def observed(*args, _fn=fn, _node=node, _effect=name in EFFECTS):
                result = _fn(*args)
                if explicit and _effect:
                    number, token = result
                    trace[_node] = (number, token.pos, token.target)
                elif _effect:
                    trace[_node] = (result, STATE.pos, STATE.target)
                else:
                    trace[_node] = (result,)
                return result

            entry["function"] = observed
            instrumented = name + "@" + str(node)
            methods[instrumented] = entry
            data["method_name"] = instrumented

    processor = GraphProcessor(graph, methods)
    if explicit:
        # The extra state output is returned to the caller; it is not hidden in a closure.
        processor.collapse_manager.set_custom_collapse(
            lambda values: {"numeric": np.sum(values[:-1]), "state": values[-1]})
    return processor, len(effect_order)


def step(processor, explicit, state):
    if explicit:
        answer = processor.infer_with_graph([[state.pos, state.target, state]])[0]
        return answer["numeric"], answer["state"]
    answer = processor.infer_with_graph([[state.pos, state.target]])[0]
    return answer, Token(STATE.pos, STATE.target)


def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))


def audit(record):
    traces_i, traces_e = {}, {}
    implicit, effects = build(record["graph"], False, traces_i)
    explicit, effects_e = build(record["graph"], True, traces_e)
    assert effects == effects_e
    steps, solved, total_distance = 0, 0, 0
    for pos, target in INITIAL:
        state_i = state_e = Token(pos, target)
        for _ in range(4):
            traces_i.clear()
            traces_e.clear()
            STATE.pos, STATE.target = state_i.pos, state_i.target
            output_i, state_i = step(implicit, False, state_i)
            # Poison implicit state while explicit execution runs. It must neither
            # consult nor mutate this unrelated state.
            STATE.pos, STATE.target = 1234567, -7654321
            output_e, state_e = step(explicit, True, state_e)
            assert (STATE.pos, STATE.target) == (1234567, -7654321)
            assert state_i == state_e, (record["fingerprint"], state_i, state_e)
            assert close(output_i, output_e), (output_i, output_e)
            assert traces_i.keys() == traces_e.keys()
            for node in traces_i:
                assert close(traces_i[node], traces_e[node]), (node, traces_i[node], traces_e[node])
            steps += 1
            if int(round(state_i.pos)) == int(round(state_i.target)):
                solved += 1
                break
        total_distance += abs(int(round(state_i.pos)) - int(round(state_i.target)))
    accuracy, loss = solved / len(INITIAL), total_distance / len(INITIAL)
    assert close(accuracy, record["accuracy"]), (accuracy, record["accuracy"])
    assert close(loss, record["loss"]), (loss, record["loss"])
    return dict(run_seed=record["run_seed"], type_mode=record["type_mode"],
                kind=record["kind"], fingerprint=record["fingerprint"],
                logical_nodes=implicit.graph.number_of_nodes() - 2,
                implicit_edges=implicit.graph.number_of_edges(),
                explicit_edges=explicit.graph.number_of_edges(),
                extra_state_edges=effects + 1, state_access_nodes=effects,
                initial_states=len(INITIAL), checked_steps=steps,
                accuracy=accuracy, loss=loss, matched=True)


def fixture_payload():
    """Two unordered writes and a read: boundary clipping makes order observable."""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["root", "collapse"])
    for name, method in [("write_plus", "move_towards_target"),
                         ("write_minus", "move_towards_target"),
                         ("read_after", "read_current_pos")]:
        graph.add_node(name, method_name=method, layer=1, is_passthrough=False)
    graph.add_edge("root", "write_plus", output_index=0, input_slot=0, data_type="scalar")
    graph.add_edge("root", "write_minus", output_index=1, input_slot=0, data_type="scalar")
    graph.add_edge("write_plus", "read_after", output_index=0, input_slot=0, data_type="scalar")
    graph.add_edge("read_after", "collapse", output_index=0, data_coord=0, data_type="scalar")
    metadata = []
    for name in ["move_towards_target", "read_current_pos", "read_target"]:
        fn = getattr(original, name)
        metadata.append({"name": name, "attributes": {
            "input_count": len(inspect.signature(fn).parameters), "output_count": 1,
            "input_types": fn.input_types, "output_types": fn.output_types}})
    return dict(nodes=[dict(id=n, attributes=d) for n, d in graph.nodes(data=True)],
                edges=[dict(source=u, target=v, key=k, attributes=d)
                       for u, v, k, d in graph.edges(keys=True, data=True)],
                topological_order=list(nx.topological_sort(graph)), methods=metadata)


def fixtures():
    payload = fixture_payload()
    original_graph = unpack(payload)
    # Original order is plus, minus, read, while an alternative legal serial
    # schedule can execute minus, plus, read and change the boundary result.
    order = payload["topological_order"]
    assert order.index("write_minus") < order.index("read_after")
    processor, _ = build(payload, True)
    inputs = [[-1., 1., Token(2, 0)], [1., -1., Token(-2, 0)]]
    STATE.pos, STATE.target = 999, -999
    answers = processor.infer_with_graph(inputs)
    assert [x["state"].pos for x in answers] == [1, -1]
    assert [x["numeric"] for x in answers] == [1, -1]
    assert (STATE.pos, STATE.target) == (999, -999)
    # A second legal ordering demonstrates that a bare DAG/type signature
    # does not itself impose an effect order.
    alternate = ["root", "write_minus", "write_plus", "read_after", "collapse"]
    rank = {n: i for i, n in enumerate(alternate)}
    assert all(rank[u] < rank[v] for u, v in original_graph.edges())
    s = Token(2, 0)
    _, s = transition("move_towards_target", (1,), s)
    _, s = transition("move_towards_target", (-1,), s)
    assert s.pos == 2
    # Frozen tokens enforce functional state transfer.
    import dataclasses
    try:
        s.pos = 0
        raise AssertionError("Token unexpectedly mutable")
    except dataclasses.FrozenInstanceError:
        pass
    return dict(native_object_state_ports=True, independent_two_sample_batch=True,
                explicit_does_not_access_global_state=True,
                all_state_reads_and_writes_ordered=True,
                immutable_tokens=True, legal_alternative_order_changes_result=True,
                original_order_result=1, alternative_order_result=2)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_all(processor, explicit):
    for pos, target in INITIAL:
        state = Token(pos, target)
        if not explicit:
            STATE.pos, STATE.target = pos, target
        for _ in range(4):
            _, state = step(processor, explicit, state)
            if int(round(state.pos)) == int(round(state.target)):
                break


def measure(records, repeats):
    # Fixed final graphs only. Paired order is shuffled with an independent RNG.
    rng = random.Random(110606)
    rows = []
    for record in records:
        if record["kind"] != "final":
            continue
        processors = {name: build(record["graph"], name == "explicit")[0]
                      for name in ["implicit", "explicit"]}
        for name, processor in processors.items():
            run_all(processor, name == "explicit")
        times = {"implicit": [], "explicit": []}
        for repetition in range(repeats):
            order = ["implicit", "explicit"]
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                run_all(processors[name], name == "explicit")
                seconds = (time.perf_counter_ns() - start) / 1e9
                times[name].append(seconds)
                rows.append(dict(run_seed=record["run_seed"], type_mode=record["type_mode"],
                                 fingerprint=record["fingerprint"], repeat=repetition,
                                 mode=name, seconds=seconds, initial_states=20))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-only", action="store_true",
                        help="Reuse the completed semantic audit of the same catalog.")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    output = HERE / args.phase
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.catalog.read_text(encoding="utf-8").splitlines() if line]
    protocol = dict(experiment="matched native implicit/explicit state encoding",
                    phase=args.phase, catalog=str(args.catalog.resolve()),
                    catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
                    executor="Unmodified ATF.GraphProcessor.graph_processor.GraphProcessor.infer_with_graph",
                    candidate_selection="All first-20-unique and final graph records exported by the frozen native type search",
                    logical_candidate_budget="Identical candidates; encoding is applied after search, not searched independently",
                    semantics="Immutable (pos,target) state token through every read/write in original topological order",
                    encoding_cost="One extra input/output port per state-access node, one token edge per such node plus final state output; no additional computation nodes",
                    initial_states=INITIAL, maximum_episode_steps=4,
                    assertions="Per-node numerical outputs, post-effect state, collapse output, every environment step, recorded accuracy/loss, poisoned-global-state isolation",
                    timing="Final graphs; one full warm-up; five paired order-randomized repetitions over all 20 episodes; includes native topological sorting, state construction/extraction, aggregation and episode loop",
                    limitations=["Encoding equivalence is expected by construction and is not evidence of better search success.",
                                 "This is not an independent explicit-state GP/CGP baseline or a test of automatic state-token discovery.",
                                 "State-access annotations are supplied by the adapter; native type tags do not infer effects or rule out races.",
                                 "The native implicit environment is evaluated one episode at a time.",
                                 "Timing depends on this Python/object-port implementation and is not a universal representation ranking."],
                    python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, networkx=nx.__version__)
    protocol["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "ATF/GraphProcessor/graph_processor.py",
                  ROOT / "experiments/minimal_stateful_comparison/methods_constrained.py"]}
    if args.timing_only:
        assert args.timing, "--timing-only requires --timing"
        previous_protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        assert previous_protocol["catalog_sha256"] == protocol["catalog_sha256"]
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["all_matched"] and summary["candidate_records"] == len(records)
    else:
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        checks = fixtures()
        rows = []
        for index, record in enumerate(records):
            rows.append(audit(record))
            if (index + 1) % 100 == 0:
                print(f"Validated {index + 1}/{len(records)} candidate records", flush=True)
        write_csv(output / "candidate_results.csv", rows)
        summary = dict(candidate_records=len(rows), checked_initial_conditions=20 * len(rows),
                       checked_environment_steps=sum(row["checked_steps"] for row in rows),
                       all_matched=all(row["matched"] for row in rows), fixture_checks=checks,
                       final_graphs=sum(row["kind"] == "final" for row in rows),
                       mean_added_edges=statistics.mean(row["extra_state_edges"] for row in rows),
                       timing_run=False)
    if args.timing:
        (output / "timing_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        times = measure(records, args.repeats)
        write_csv(output / "timing_raw.csv", times)
        paired = {}
        for row in times:
            key = (row["run_seed"], row["type_mode"])
            paired.setdefault(key, {"implicit": [], "explicit": []})[row["mode"]].append(row["seconds"])
        ratios = [statistics.median(v["explicit"]) / statistics.median(v["implicit"]) for v in paired.values()]
        summary.update(timing_run=True, timed_final_graphs=len(paired),
                       median_explicit_over_implicit_ratio=statistics.median(ratios),
                       timing_repeats=args.repeats)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
