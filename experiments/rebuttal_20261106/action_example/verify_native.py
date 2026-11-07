"""Replay frozen derived controllers on the unmodified native ATF processor.

No training, no source edits, no imports of ATF-nodes experiment packages.
All output paths are confined to this script's directory.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# 发布副本只读取随包提供的冻结参考，不依赖外部研究目录。
SOURCE = HERE / "reference"
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(HERE / "cache" / "numba")
os.environ["MPLCONFIGDIR"] = str(HERE / "cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / "cache")
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from ATF.GraphProcessor.graph_processor import GraphProcessor
from ATF.CollapseManager.collapse_functions import CollapseMethod


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(name, obj):
    path = (HERE / name).resolve()
    assert path.is_relative_to(HERE.resolve())
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_references():
    """Load guarded NumPy-only interpreter; isolate exact tick body without imports."""
    spec = importlib.util.spec_from_file_location("frozen_portable_reference", SOURCE / "reproduce.py")
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    tree = ast.parse((SOURCE / "kernel.py").read_text(encoding="utf-8"))
    tick = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tick")
    tick.decorator_list = []
    isolated = ast.Module(body=[tick], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"np": np}
    exec(compile(isolated, str(SOURCE / "kernel.py"), "exec"), namespace)
    return portable, namespace["tick"]


@dataclass
class Context:
    events: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    sinks: list = field(default_factory=list)
    committed: bool = False
    next_state: tuple | None = None


def finite_feature(value):
    return min(4.0, max(-4.0, value)) if math.isfinite(value) else 0.0


class NativeController:
    """Expand frozen parameters and all arithmetic into native method nodes.

    The adapter supplies synchronous state semantics explicitly. The native
    processor does not infer effects or offer a transaction/ownership checker.
    """

    def __init__(self, record, insertion_seed=None):
        self.record = record
        self.methods = {}
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("root", layer=-1, output_types=["Observation", "OldState", "UnitUniform", "StepContext"])
        self.graph.add_node("collapse", layer=-1)
        self.context = None
        self.features = []
        self.scores = []

        def node(name, fn, inputs, outputs, sources):
            self.methods[name] = {"function": fn, "input_count": len(inputs), "output_count": len(outputs),
                                  "input_types": inputs, "output_types": outputs, "vectorized": False}
            self.graph.add_node(name, method_name=name, layer=0, is_passthrough=False)
            for slot, (source, index) in enumerate(sources):
                self.graph.add_edge(source, name, output_index=index, input_slot=slot, data_type=inputs[slot])
            return name

        for i, (op, a, b, owner) in enumerate(record["features"]):
            name = f"feature_{i}"
            if op == 0:
                fn = lambda obs, index=a: finite_feature(float(obs[index]))
                sources, types = [("root", 0)], ["Observation"]
            elif op == 7:
                fn = lambda old, index=a: finite_feature(float(old[index])) if record["mode"] else 0.0
                sources, types = [("root", 1)], ["OldState"]
            else:
                operations = {1: lambda x, y: x * y, 2: lambda x, y: x - y,
                              3: lambda x, y: x + y, 4: lambda x, y: float(x > y),
                              5: lambda x: abs(x), 6: lambda x: min(1.0, max(-1.0, x))}
                operation = operations[op]
                fn = lambda *values, operation=operation: finite_feature(operation(*values))
                sources = [(self.features[a], 0)] + ([(self.features[b], 0)] if op in (1, 2, 3, 4) else [])
                types = ["Scalar"] * len(sources)
            self.features.append(node(name, fn, types, ["Scalar"], sources))
            self.graph.nodes[name]["owner"] = owner
            self.graph.nodes[name]["frozen_feature"] = [op, a, b, owner]

        for action, weights in enumerate(record["weights"]):
            total = None
            for j, weight in enumerate(weights):
                # Native GP rejects predecessor-free nodes: a context edge is
                # an inert invocation trigger for each frozen coefficient.
                const = node(f"weight_{action}_{j}", lambda _, w=weight: w,
                             ["StepContext"], ["Scalar"], [("root", 3)])
                product = node(f"multiply_{action}_{j}", lambda x, y: x * y,
                               ["Scalar", "Scalar"], ["Scalar"], [(const, 0), (self.features[j], 0)])
                if total is None:
                    total = product
                else:
                    total = node(f"sum_{action}_{j}", lambda x, y: x + y,
                                 ["Scalar", "Scalar"], ["Scalar"], [(total, 0), (product, 0)])
            self.scores.append(total)

        emissions = []
        for k, feature in enumerate(record["emissions"]):
            emissions.append(node(f"emission_{k}", lambda value: min(1.0, max(-1.0, value)),
                                  ["Scalar"], ["Scalar"], [(self.features[feature], 0)]))

        def commit(old, context, *values):
            emitted = values[:4]
            mode = record["mode"]
            rho = .9 if mode in (2, 4) else 0.0
            new = []
            for k in range(4):
                if mode in (3, 4):
                    value = rho * .5 * (old[k % 2] + old[2 + k % 2]) + emitted[k % 2] + emitted[2 + k % 2]
                elif mode:
                    value = rho * old[k] + emitted[k]
                else:
                    value = 0.0
                new.append(min(1.0, max(-1.0, value)))
            assert not context.committed
            context.next_state = tuple(new)
            context.committed = True
            context.events.append("state_commit")
            return (*new, context)

        commit_sources = [("root", 1), ("root", 3)] + [(e, 0) for e in emissions] + [(s, 0) for s in self.scores]
        node("state_commit", commit, ["OldState", "StepContext"] + ["Scalar"] * (4 + len(self.scores)),
             ["Scalar"] * 4 + ["CommittedStep"], commit_sources)

        def router(*args):
            priorities = args[:-2]
            u, context = args[-2:]
            assert context.committed and context.events == ["state_commit"]
            epsilon = record.get("epsilon", 0.0)
            choice = -1
            if record["enabled"]:
                if epsilon > 0 and u < epsilon:
                    choice = min(len(priorities) - 1, int(u / epsilon * len(priorities)))
                else:
                    routed = (u - epsilon) / (1.0 - epsilon) if epsilon > 0 else u
                    best = max(priorities)
                    ties = [i for i, value in enumerate(priorities) if value == best]
                    choice = ties[min(len(ties) - 1, int(routed * len(ties)))]
            permits = tuple(i == choice for i in range(len(priorities)))
            assert sum(permits) <= 1
            context.events.append("exclusive_route")
            return permits

        actions = record["action_nodes"]
        node("exclusive_router", router, ["Scalar"] * len(actions) + ["UnitUniform", "CommittedStep"],
             ["ActionPermit"] * len(actions), [(s, 0) for s in self.scores] + [("root", 2), ("state_commit", 4)])
        for index, action_name in enumerate(actions):
            def sink(permit, index=index, action_name=action_name):
                context = self.context
                assert context.committed and "exclusive_route" in context.events
                context.sinks.append(action_name)
                context.events.append(f"sink:{action_name}:{int(permit)}")
                if permit:
                    context.actions.append(index + 1)
                return []
            node(f"action_{action_name}", sink, ["ActionPermit"], [], [("exclusive_router", index)])

        # Observability-only edges; action sinks remain zero-output/out-degree.
        monitor = [(f, 0) for f in self.features] + [(s, 0) for s in self.scores]
        monitor += [("exclusive_router", i) for i in range(len(actions))]
        monitor += [("state_commit", i) for i in range(4)]
        for coord, (source, index) in enumerate(monitor):
            datatype = self.methods[source]["output_types"][index]
            self.graph.add_edge(source, "collapse", output_index=index, data_coord=coord, data_type=datatype)
        self.validate()
        if insertion_seed is not None:
            rng = np.random.default_rng(insertion_seed)
            nodes = list(self.graph.nodes(data=True)); edges = list(self.graph.edges(keys=True, data=True))
            rng.shuffle(nodes); rng.shuffle(edges)
            reordered = nx.MultiDiGraph(); reordered.add_nodes_from(nodes)
            for u, v, k, d in edges:
                reordered.add_edge(u, v, key=k, **d)
            self.graph = reordered
        self.processor = GraphProcessor(self.graph, self.methods, collapse_method=CollapseMethod.IDENTITY)

    def validate(self):
        assert nx.is_directed_acyclic_graph(self.graph)
        for name, info in self.methods.items():
            incoming = list(self.graph.in_edges(name, data=True))
            assert len(incoming) == info["input_count"]
            assert sorted(d["input_slot"] for _, _, d in incoming) == list(range(len(incoming)))
            for source, _, edge in incoming:
                source_types = self.graph.nodes["root"]["output_types"] if source == "root" else self.methods[source]["output_types"]
                assert source_types[edge["output_index"]] == info["input_types"][edge["input_slot"]]
            if name.startswith("action_"):
                assert info["output_count"] == 0 and self.graph.out_degree(name) == 0
            if name.startswith("feature_"):
                assert nx.has_path(self.graph, name, "state_commit")
        assert nx.has_path(self.graph, "state_commit", "exclusive_router")

    def tick(self, obs, old, u):
        context = Context(); self.context = context
        snapshot = tuple(float(x) for x in old)
        outputs = self.processor.infer_with_graph([[obs, snapshot, u, context]])[0]
        n, a = len(self.features), len(self.scores)
        features = np.asarray(outputs[:n], dtype=float)
        scores = np.asarray(outputs[n:n+a], dtype=float)
        permits = tuple(bool(x) for x in outputs[n+a:n+2*a])
        new = np.asarray(outputs[n+2*a:], dtype=float)
        assert len(context.actions) <= 1 and len(context.sinks) == a
        assert context.events[0:2] == ["state_commit", "exclusive_route"]
        assert tuple(new) == context.next_state
        assert snapshot == tuple(float(x) for x in old)
        action = context.actions[0] if context.actions else 0
        assert action == (permits.index(True) + 1 if any(permits) else 0)
        return action, new, features, scores, permits, context.events

    def export(self):
        return {"nodes": [{"id": n, **data} for n, data in self.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, "key": k, **data} for u, v, k, data in self.graph.edges(keys=True, data=True)],
                "methods": {name: {k: v for k, v in info.items() if k != "function"} for name, info in self.methods.items()},
                "notes": ["Input edges to frozen coefficients are native invocation triggers.",
                          "state_commit and its completion token explicitly implement synchronous state semantics.",
                          "Edges to collapse are monitoring outputs, not policy decisions.",
                          "Metadata validation here belongs to the adapter; GraphProcessor itself is unmodified."]}


def compare(controller, portable, kernel, obs, h, u):
    record = controller.record
    features = np.asarray(record["features"], dtype=np.int32)
    weights = np.asarray(record["weights"], dtype=np.float64)
    emissions = np.asarray(record["emissions"], dtype=np.int32)
    expected = kernel(obs, features, weights, emissions, np.asarray(h, dtype=float), record["mode"], u,
                      record["enabled"], -1, record.get("epsilon", 0.0))
    action, new, values, scores, permits, events = controller.tick(obs, h, u)
    portable_action, portable_h = portable.graph_tick(obs, record, list(h), u)
    assert action == expected[0] == portable_action
    assert expected[2] == (action if action else -1)
    for got, want in ((new, expected[1]), (new, portable_h), (values, expected[3]), (scores, expected[4])):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    if record["mode"] in (3, 4):
        assert new[0] == new[2] and new[1] == new[3]
    assert sum(permits) == int(action != 0)
    errors = [float(np.max(np.abs(values - expected[3]))), float(np.max(np.abs(scores - expected[4]))),
              float(np.max(np.abs(new - expected[1])))]
    return action, new, values, scores, permits, events, errors


def boundary_checks(record, portable, kernel):
    import copy
    cases = []
    for label, enabled, epsilon, zeros, mode in [
        ("frozen", True, record.get("epsilon", 0.0), False, 4),
        ("uniform_ties", True, 0.0, True, 4),
        ("exploration_and_ties", True, .1, True, 4),
        ("disabled", False, 0.0, False, 4),
        ("no_state", True, 0.0, False, 0),
        ("instant_shared", True, 0.0, False, 3),
        ("private_memory", True, 0.0, False, 2),
    ]:
        rec = copy.deepcopy(record); rec.update(enabled=enabled, epsilon=epsilon, mode=mode)
        if zeros: rec["weights"] = np.zeros_like(rec["weights"]).tolist()
        native = NativeController(rec)
        reordered = NativeController(rec, insertion_seed=20261106)
        obs = np.linspace(-7.0, 7.0, 157)
        # Unequal channel copies exercise the exact shared-state average.
        old = np.array([.7, -.2, -.4, .8])
        for u in (0.0, .049, .05, .099999, .1, .5, .999999999):
            result = compare(native, portable, kernel, obs, old, u)
            shuffled = compare(reordered, portable, kernel, obs, old, u)
            assert result[0] == shuffled[0]
            for index in (1, 2, 3): np.testing.assert_array_equal(result[index], shuffled[index])
            cases.append({"case": label, "u": u, "action": result[0], "permits": result[4],
                          "max_errors_features_scores_state": result[6], "event_order": result[5]})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["bird"], choices=["bird"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--label", default="formal")
    args = parser.parse_args()
    assert args.label.replace("_", "").isalnum() and 1 <= args.episodes <= 20
    sources = [SOURCE / "reproduce.py", SOURCE / "kernel.py", SOURCE / "world_test.npz", SOURCE / "final_raw.npz",
               SOURCE / "models/system_s104.json", ROOT / "ATF/GraphProcessor/graph_processor.py",
               ROOT / "ATF/CollapseManager/collapse_functions.py", Path(__file__)]
    sources += [SOURCE / f"models/graph_s104_{name}.json" for name in args.tasks]
    before = {str(p): digest(p) for p in sources}
    protocol = {"purpose": "Execution equivalence of frozen controllers; no retraining or new generalization estimate",
                "tasks": args.tasks, "episodes_per_task": args.episodes, "training_seed": 104,
                "environment_seed": 200000104, "episode_start": 83000000,
                "selection": "First consecutive episodes from the previously used frozen test stream; no outcome selection",
                "native_executor": "Unmodified GraphProcessor.infer_with_graph, serial execution",
                "references": ["Unmodified portable interpreter graph_tick", "Exact original kernel.tick function body, AST-isolated without its JIT decorator/imports",
                               "Previously archived final_raw.npz trajectories wherever available in the fixed episode prefix"],
                "randomness": "Same externally supplied draw(seed,episode,tick,1900+task) for all executors",
                "tolerance": {"rtol": 1e-12, "atol": 1e-12}, "source_sha256": before,
                "limitations": ["Adapter explicitly implements synchronous commit and routing methods.",
                                "Not proof of automatic effect scheduling or arbitrary concurrent safety.",
                                "No training, search, optimization, or evaluation-driven model selection.",
                                "Boundary probes modify configurations only to test executor behavior; they are not performance ablations."]}
    save_json(f"protocol_{args.label}.json", protocol)
    portable, kernel = load_references()
    with np.load(SOURCE / "world_test.npz", allow_pickle=False) as data:
        world = {k: data[k].copy() for k in data.files}
    started = time.perf_counter(); report = []; boundary = None
    for name in args.tasks:
        graph_source = json.loads((SOURCE / f"models/graph_s104_{name}.json").read_text(encoding="utf-8"))
        record = graph_source["record"]
        native = NativeController(record)
        save_json(f"native_graph_{name}.json", native.export())
        if name == "bird":
            boundary = boundary_checks(record, portable, kernel)
            save_json(f"boundary_checks_{args.label}.json", boundary)
        task = record["task"]; seed = 200000104
        trace = {key: [] for key in ("episode", "step", "u", "old_state", "new_state", "features", "scores", "permits", "action")}
        outcomes = []; max_errors = np.zeros(3); task_start = time.perf_counter(); archived_steps = 0
        with np.load(SOURCE / "final_raw.npz", allow_pickle=False) as data:
            archived = {k: data[k].copy() for k in data.files if k.startswith(f"s104_{name}_trace_ep") or k.startswith(f"s104_{name}_values_ep")}
        for ep in range(83000000, 83000000 + args.episodes):
            state = portable.initialize(task, seed, ep, world)
            reference_state = state.copy(); h = np.zeros(4)
            while not state[1]:
                obs = portable.observe(task, seed, ep, state, world)
                u = portable.draw(seed, ep, int(state[0]), 1900 + task)
                old = h.copy(); step = int(state[0])
                action, h, values, scores, permits, events, errors = compare(native, portable, kernel, obs, old, u)
                max_errors = np.maximum(max_errors, errors)
                ref_action, _ = portable.graph_tick(portable.observe(task, seed, ep, reference_state, world), record, list(old), u)
                portable.advance(task, seed, ep, state, action, world)
                portable.advance(task, seed, ep, reference_state, ref_action, world)
                np.testing.assert_array_equal(state, reference_state)
                archive_key = f"s104_{name}_trace_ep{ep}"
                if archive_key in archived:
                    saved = archived[archive_key][step]
                    assert int(saved[0]) == step and int(saved[1]) == int(saved[2]) == action
                    np.testing.assert_allclose(h, saved[3:7], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(scores, saved[7:7+len(scores)], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(values, archived[f"s104_{name}_values_ep{ep}"][step], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal([state[3], state[1], state[2], state[0]], saved[11:15])
                    archived_steps += 1
                for key, value in (("episode", ep), ("step", step), ("u", u), ("old_state", old), ("new_state", h),
                                   ("features", values), ("scores", scores), ("permits", permits), ("action", action)):
                    trace[key].append(value)
            outcomes.append({"episode": ep, "success": int(state[2]), "progress": float(state[3]), "steps": int(state[0])})
            print(json.dumps({"task": name, "episode": ep, "steps": int(state[0]), "matched": True}), flush=True)
        np.savez_compressed(HERE / f"native_trace_{name}_{args.label}.npz", **{key: np.asarray(value) for key, value in trace.items()})
        report.append({"task": name, "episodes": args.episodes, "steps_checked": len(trace["action"]), "all_matched": True,
                       "max_absolute_error_features_scores_state": max_errors.tolist(), "outcomes": outcomes,
                       "features": len(record["features"]), "actions": len(record["action_nodes"]),
                       "frozen_graph_nodes": len(graph_source["nodes"]), "frozen_graph_edges": len(graph_source["edges"]),
                       "native_adapter_nodes": native.graph.number_of_nodes(), "native_adapter_edges": native.graph.number_of_edges(),
                       "zero_output_sinks": record["action_nodes"], "independent_shared_channels": 2,
                       "steps_also_checked_against_preexisting_saved_trace": archived_steps,
                       "seconds": time.perf_counter() - task_start})
    after = {str(p): digest(p) for p in sources}
    assert before == after, "A protected source changed during verification"
    result = {"passed": True, "tasks": report, "total_steps": sum(t["steps_checked"] for t in report),
              "boundary_probes": len(boundary or []), "boundary_reordered_probes": len(boundary or []),
              "protected_sources_unchanged": before == after, "source_sha256": after,
              "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "networkx": nx.__version__,
              "seconds": time.perf_counter() - started, "interpretation": protocol["purpose"], "limitations": protocol["limitations"]}
    save_json(f"results_{args.label}.json", result)
    print(json.dumps({"passed": True, "steps": result["total_steps"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
