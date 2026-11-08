# Concrete typed-graph workflow integration

This study implements the same discover–register–reuse workflow in frozen ATF V0 and **PyTensor 3.3.8**, a concrete typed computational-graph system. It asks which native interfaces and external adapters are needed to discover a numerical subgraph, register it as a callable module, reuse it in later programs, and rebuild the resulting program from a stored declaration.

Both systems completed the tested numerical workflow without modifications to their core Python sources. All paired candidate traces, admitted modules, final outputs, and stored declarations agreed. This is an integration and interface-coverage result. It does not rank synthesis algorithms, establish a unique representational capability, or measure development effort or runtime speed.

## Implementation boundary

| Responsibility | ATF V0 | PyTensor 3.3.8 |
| --- | --- | --- |
| Native executable graph | `GraphProcessor` over `networkx.MultiDiGraph` | Native symbolic variables, arithmetic operations, and compiled `function` |
| Native reusable module | `EvolvedMethod` | Non-inlined `OpFromGraph` |
| Numerical graph adapter | `atf_backend.py` translates ports and live nodes | `pytensor_backend.py` translates ports and symbolic expressions |
| Method-name registry | Append-only adapter dictionary using ATF method metadata | Append-only adapter dictionary containing primitive builders and `OpFromGraph` objects |
| Candidate enumeration, training loss, and admission decision | Shared external driver | Shared external driver |
| Persistence and restoration | Shared JSON declarations, rebuilt through native interfaces | Shared JSON declarations, rebuilt through native interfaces |
| Core Python source files changed | 0 | 0 |

The driver does not use one backend to calculate the other backend's fitness. Every recorded candidate is evaluated by the corresponding native system. Neither backend hides the complete candidate inside a single callback implementing a separate graph interpreter.

ATF is loaded from `experiments/rebuttal_20261106/runtime_v0/ATF`, rather than the current top-level package. The later effect-ordering extension is not used in the numerical comparison. PyTensor outer functions explicitly use the Python linker with `optimizer=None`. `OpFromGraph` uses its native non-inlined, lazy inner-function linking. The adapter does not pass ignored compilation options to the inner module or claim to override its native linking policy.

The adapter interfaces are `register(name, program)` and `compile(program)`. A program declares ordered input ports, a list of method calls with forward references, and a single output reference. The numerical interface accepts an `N x D` sample matrix and returns one scalar per row. All numerical ports carry the label `scalar`; this is **not a semantic-type-prior experiment**. The adapters translate a common file format but do not supply a new type system shared by the two frameworks.

## Numerical workflow protocol

The three sequential tasks are:

1. `f1(x) = 2(x+1)`;
2. `f2(x) = 2(x+1)+3`;
3. `f3(x) = x+2.5`.

Each stage uses 17 fixed dyadic training values and 32 disjoint verification values. A stage stops at its first candidate with training MSE at most `1e-12`. Verification data do not affect search or module admission.

The common driver enumerates unary method chains in increasing length. Its initial library contains increment, multiply-by-two, add-three, multiply-by-one-half, and identity. Method order is shuffled once for each case. The binary add primitive is available for separate shared-subgraph contracts, but is not part of the unary enumeration.

Every stage has limits of four top-level method calls, eight recursively expanded primitive calls, and 512 candidate evaluations. All three targets are representable using only the initial primitive library within these limits. After stages 1 and 2, the successful program is wrapped using the backend's native module interface and appended to the method registry. The growing library is the workflow under examination; this study does not include a fixed-library ablation or estimate a benefit attributable to library growth.

The predeclared order seeds are `811100` through `811111`. They define **12 coverage cases per backend and 11 distinct initial primitive orders**: seeds `811109` and `811111` produce the same order. Both records are retained. These cases are not 12 independent tasks, unique permutations, or independent learning trials supporting statistical generalization.

`protocol_workflow.json` records the numerical protocol, software environment, experiment source hashes, and hashes of every Python source file in the two native packages before the formal run. Source hashes are checked again after execution.

## Numerical results

| Check | ATF V0 | PyTensor 3.3.8 |
| --- | ---: | ---: |
| Completed three-stage coverage cases | 12/12 | 12/12 |
| Stages solved and verified | 36/36 | 36/36 |
| Stage 2 calls the discovered stage-1 module | Every case | Every case |
| Stage 3 calls stage 2, transitively reusing stage 1 | Every case | Every case |
| Disk declaration reconstructed in a fresh backend instance | 12/12 | 12/12 |
| Shared fan-out and native nested-module contracts | Passed | Passed |
| Local replacement, recompilation, and preservation of the old program | Passed | Passed |
| Native Python source modifications | 0 | 0 |

There are **36 paired stages and 2,638 native candidate evaluations across both backends**. Every paired search trace, selected program, admitted module, and verification prediction agrees. Evaluation records are not independent observations; the same deterministic proposal policy is deliberately used in both systems.

The common shared-graph contract computes an intermediate once and feeds it to two add ports. A second contract registers this graph, composes two calls into a new module, and invokes the nested module. A local-replacement contract changes a primitive in a copied program, recompiles it, and verifies both the new result and the unchanged old compiled program. Each contract uses five fixed inputs. Separate native probes count actual primitive invocations to inspect sharing.

Persistence is an explicit disk round trip: the driver writes module and program declarations as JSON, reads the file, creates a fresh backend instance, registers the modules in dependency order, and checks the reconstructed final program. This is **external declarative persistence**, not a test or claim of either framework's native object-serialization facility.

`audit_workflow.py` independently reconstructs enumeration order and candidate arithmetic using NumPy, without importing or executing ATF, PyTensor, either adapter, or the experiment runner. It checks stopping, costs, module dependencies, verification outputs, snapshots, duplicate order cases, and summary counts. The completed audit replayed all 2,638 evaluation records, reconstructed 24 disk snapshots, and verified the frozen source hashes of **38 ATF V0 and 405 PyTensor Python files**. This arithmetic audit does not independently rerun every native interface probe.

## Native interface observations

The native probes deliberately examine interfaces outside the common scalar workflow. Their graphs and inputs differ where necessary to expose each API's behavior; they should not be treated as a matched aggregate capability score.

| Observation | Frozen ATF V0 | PyTensor 3.3.8 |
| --- | --- | --- |
| Scalar reusable subgraph | Accepted through `EvolvedMethod` | Accepted through `OpFromGraph` |
| Vector behavior | A primitive may return a vector; a single vector-valued `EvolvedMethod` result is rejected | Vector input/output `OpFromGraph` is accepted |
| Shared intermediate | Counted primitive runs once per sample despite two consumer edges | Counted primitive runs once per sample despite two consumer edges |
| Custom Python object input | Registered Python method reads an `Observation` object; the scalar-output wrapper also accepts it | A custom `Type` and `Op` accept the object without editing core sources |
| Wrong Python object class | Method access fails during execution | Custom `Type` rejects the wrong class |
| Rank boundary | Native batch executor rejects a one-dimensional batch array | Native symbolic interface rejects a vector supplied to a scalar module before compilation |

The vector result identifies a specific V0 wrapper limitation: `EvolvedMethod` attempts to convert a single output to `float` and rejects an array containing two elements. The underlying `GraphProcessor` vector primitive succeeds. No patch was made to hide or remove this boundary.

The rank probes are **different checks**. The ATF case concerns the executor's batch container shape; the PyTensor case concerns a symbolic port's tensor rank. Their rejection results cannot establish equivalent type guarantees. Likewise, an ATF method's runtime attribute access is not the same mechanism as a PyTensor custom type filter. Full observations and safe error categories are preserved under `native_interface_probes` in `results/formal/summary.json`.

Both systems expose useful extension interfaces, but the amount and form of adapter code differ. Source-line counts are not used as a development-cost metric. The unchanged core hashes establish that these particular integrations were performed externally, not that their engineering costs are equal or that all possible workloads need no core changes.

## Separate action-controller study

`action_comparison.py` extends the comparison to the already frozen Catch, Maze, Snake, and Bird controllers in `experiments/rebuttal_20261106/action_example/`. It constructs their numerical features, action scores, routing, and explicit state updates as native PyTensor operations. Every step is compared with a fresh execution of the existing native ATF adapter and the reference kernel; Bird is additionally compared with its archived step trace. This is execution migration of fixed learned controllers, not retraining or a new learning-performance evaluation.

| Task | Consecutive evaluation episodes | Verified steps | Maximum absolute error in features, scores, and memory |
| --- | ---: | ---: | ---: |
| Catch | 2 | 320 | 0 |
| Maze | 2 | 163 | 0 |
| Snake | 2 | 237 | 0 |
| Bird | 20 | 20,480 | 0 |
| Total | 26 | 21,200 | 0 |

Actions, permits, and the resulting environment states agree at every step. All 49 additional routing/state boundary probes pass. The original reference files and all 443 backend Python source files are unchanged. The protocol was written before the formal run, and the episode selection is the consecutive prefix of the existing evaluation stream. Neither the steps nor the episodes constitute independent training replications. The recorded run duration is descriptive, not a backend speed comparison. Native Apply-node and ATF-node counts use different operator granularities and are not a graph-compactness score.

The separate action probe performs actual changes to a small Python environment state. ATF uses connected zero-output action sinks. This experiment's PyTensor controller adapter uses custom action `Op` objects and an output-anchored chain of completion tokens. Outputs or updates can keep effectful operations reachable; separate probes inspect anchoring and unrequested branches. The chosen custom `Op` implementation is not a claim that a custom class is mandatory for every Python callback. This identifies an integration convention; it does not show that PyTensor cannot execute actions or establish automatic resource-conflict ordering or concurrency safety in either system. Environment advancement in the controller replay remains outside both graphs, as in the frozen study.

Action protocols and completed results are stored separately as `results/action_protocol_<label>.json` and `results/action_results_<label>.json`. Their final numerical totals should be read from those records rather than inferred from episode counts. They are not included in the 2,638-candidate or 36-paired-stage numerical workflow totals above.

`effect_mode_checks.py` separately tests the small action example with the Python linker under both `optimizer=None` and `optimizer="fast_run"`. Both modes pass. The same mutually exclusive action permits make the two backends update a real Python environment through the position sequence `[1, 7, 7]`. For PyTensor, both requested-output anchoring and shared-variable `updates` retain the action calls. The `updates` version has empty user-visible outputs. An unanchored callback branch is not executed.

The public **`pytensor.wrap_py` decorator** also passes this example in both modes and with both anchoring paths, without a handwritten `Op` subclass. Therefore the custom class used in the full-controller adapter must not be interpreted as the minimum integration requirement. These observations support a concrete difference in execution conventions: ATF retains connected zero-output sinks in the executed graph, while the tested PyTensor paths retain effects through an explicit output or update root. They establish neither an exclusive action capability nor a general engineering-cost advantage. Ordering between conflicting non-exclusive actions still requires an explicit execution design. The complete floating-point controllers were tested only in the unoptimized mode; FAST_RUN coverage applies to the small action probes.

## Reproduction and artifacts

Run commands from the repository root. The recorded numerical environment is Python 3.12.9, NumPy 2.1.3, and PyTensor 3.3.8 on Windows. Install dependencies in a dedicated environment:

```console
python -B -m pip install -r experiments/rebuttal_20261106/requirements.txt
python -B -m pip install pytensor==3.3.8
python -B experiments/typed_dag_workflow_20261108/run_workflow.py --phase smoke
python -B experiments/typed_dag_workflow_20261108/run_workflow.py --phase formal
python -B experiments/typed_dag_workflow_20261108/audit_workflow.py
```

`bootstrap.py` also supports a separately installed dependency directory through the environment variable `ATF_WORKFLOW_DEPS`. Point it at the directory containing the `pytensor` package. For example, in PowerShell, after installing PyTensor into a temporary target directory:

```powershell
$env:ATF_WORKFLOW_DEPS = Join-Path $env:TEMP "atf-workflow-deps"
python -B -m pip install --target $env:ATF_WORKFLOW_DEPS pytensor==3.3.8 numpy==2.1.3
python -B experiments/typed_dag_workflow_20261108/run_workflow.py --phase smoke
```

The formal runner resumes existing case files; it does not replace completed records. For a fresh execution, use a separate checkout with no generated `results/formal/` directory while retaining the frozen protocol and sources. The formal command requires the recorded source hashes and environment metadata to match. A different environment requires a separately documented protocol rather than silently overwriting the archived one. The independent audit reads saved traces and declarations and also requires the frozen native source packages to be available for hashing.

Run the separate controller comparison with:

```console
python -B experiments/typed_dag_workflow_20261108/action_comparison.py --bird-episodes 20 --other-episodes 2 --label reproduction
python -B experiments/typed_dag_workflow_20261108/effect_mode_checks.py
```

The principal artifacts are:

- `protocol_workflow.json`: frozen numerical protocol, environment, and source hashes;
- `results/formal/atf_v0_<seed>.json` and `pytensor_<seed>.json`: complete traces and selected programs;
- `results/formal/snapshot_<backend>_<seed>.json`: disk-persisted module and final-program declarations;
- `results/formal/summary.json`: paired numerical results, contracts, and native interface observations;
- `results/formal/audit.json`: independent arithmetic, dependency, snapshot, and source-hash audit;
- `results/smoke/`: separately retained preliminary interface checks;
- `results/action_*_<label>.json`: separate controller migration protocols and results.
- `results/effect_modes.json`: output/update anchoring and decorated Python callbacks in both optimization modes.

Imports use the frozen runtime. Bytecode is disabled, and compiler/font caches are directed to temporary storage. The action comparison reads the published reference copies in the ATF repository. Neither study imports, runs, or modifies `ATF-nodes`, and neither changes the earlier `typed_dag_comparison_20261108` experiment.
