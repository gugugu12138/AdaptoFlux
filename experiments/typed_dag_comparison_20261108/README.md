# Typed DAG execution and method-library controls

This experiment separates execution backend from method-library growth under one common search procedure. It compares the frozen V0 ATF runtime with an independently implemented typed DAG evaluator, giving both backends the same primitive methods, candidate proposals, evaluation budgets, and module-admission rule.

The primary experiment found identical paired search trajectories and final outputs. Library growth increased success on the final affine task in both backends; it did not solve the final parity task at the primary budget. The reference evaluator was faster on these small scalar workloads. These results characterize the tested interfaces and implementations, rather than establish an ATF-specific synthesis or speed advantage.

## Protocol

The design crosses two factors:

| Execution backend | Fixed library | Growing library |
| --- | --- | --- |
| Frozen V0 ATF | Original primitives only | Original primitives plus admitted modules |
| Independent typed DAG | Original primitives only | Original primitives plus admitted modules |

The ATF arm imports `experiments/rebuttal_20261106/runtime_v0/ATF`, **not** the current top-level `ATF` package. It constructs native `GraphProcessor` graphs and wraps admitted modules with V0 `EvolvedMethod`. The added effect-ordering implementation is not part of this comparison. The independent backend is implemented in `reference_dag.py`; it validates nominal types and forward references, then compiles active nodes and module bodies into an instruction sequence.

Every port in this experiment has the single nominal type `scalar`. Consequently, this experiment does not measure the benefit of semantic type distinctions. It also does not reproduce a published CGP, MT-CGP, ECGP, or DreamCoder implementation. The common CGP-style proposal procedure and successful-module admission rule are experimental controls, not the historical GraphEvo trainer or its frequency-based extractor.

Each candidate has eight feed-forward genome slots, with the final slot as output. Search starts with a random genome, restarts every 32 evaluations, and otherwise mutates one method or one input reference. Method changes repair input arity while retaining existing input references where possible. Neutral moves are accepted. All arms perform exactly 256 candidate evaluations per stage, including evaluations after first success; there is no fitness cache.

All arms also have the same limit of **eight expanded active primitive operations**. An admitted module costs its recursively expanded body, rather than one operation. A shared node executes once within its graph invocation; separate module invocations incur separate costs. Proposals exceeding the expanded limit are rejected before evaluation, counted, and included in elapsed run time. Primitive-only witnesses verify that every target is representable within the common limit.

After stages 1 and 2, growing-library arms admit the best training-successful active graph if it contains at least two expanded primitive operations. Only used external inputs become module ports. Internal sharing is retained, and modules may call previously admitted modules. No oracle modules or modules from other seeds are injected. Admission uses training results only; verification outputs do not affect search, selection, or admission. Both backends independently follow this same rule.

The task families are:

| Family | Stage 1 | Stage 2 | Stage 3 | Evaluation data |
| --- | --- | --- | --- | --- |
| Affine | `2(x+1)` | `2(x+1)+3` | `x+2.5` | 17 fixed dyadic training points and 32 disjoint dyadic verification points, including interpolation and extrapolation |
| Parity | Parity of the first 2 inputs | Parity of the first 3 inputs | Parity of all 4 inputs | Complete 16-row four-bit truth table at every stage |

Success requires verification MSE at most `1e-12`. Parity uses the complete finite domain for both search and verification; it is not a held-out generalization experiment. The affine library contains increment, add-three, multiply-by-two, multiply-by-one-half, and identity. The parity library contains subtraction, absolute value, multiplication, one-minus, and addition.

## Primary results: 256 evaluations per stage

There are 30 paired search seeds, `811000` through `811029`, per condition. Tasks and data are fixed. Each backend initializes its own `random.Random(seed)` and runs the search independently. Backend execution order alternates across seeds.

Successes out of 30 seeds are:

| Family | Stage | ATF, fixed | Typed DAG, fixed | ATF, growing | Typed DAG, growing |
| --- | --- | ---: | ---: | ---: | ---: |
| Affine | 1 | 26 | 26 | 26 | 26 |
| Affine | 2 | 20 | 20 | 23 | 23 |
| Affine | 3 | 2 | 2 | 10 | 10 |
| Parity | 1 | 5 | 5 | 5 | 5 |
| Parity | 2 | 0 | 0 | 0 | 0 |
| Parity | 3 | 0 | 0 | 0 | 0 |

All 360 paired stages matched in search-trace hashes, selected programs, available modules, and final verification outputs. The independent arithmetic audit replayed 184,320 candidate-evaluation records and checked losses, expanded costs, acceptances, final outputs, and module provenance. The audit reuses the published proposal functions to reconstruct candidates, but uses a separate formula table and evaluator for arithmetic and costs.

The statistical unit is a search seed within a condition, not a candidate, a stage, or an execution backend. The 240 complete runs comprise two families, two library conditions, two backends, and 30 seeds. Stages from the same run are dependent, and paired backends are not additional independent replications. Evaluation records include repeated candidates. Per-condition Wilson intervals are available in `artifacts/formal/summary.json`.

The final affine improvement from 2/30 to 10/30 belongs to the shared library-growth mechanism. It appears identically in both backends. At the primary budget, no library condition solved parity stages 2 or 3. Matching traces establish agreement for the executed workload, not equivalence of the complete systems or of their capabilities outside this protocol.

`mean_first_success_capped` assigns unsuccessful runs the index 257. It is a descriptive capped index, not the mean number of evaluations needed to solve a task.

## Cost accounting

The acquisition stages are included in the cost record. Mean actual primitive calls per complete three-stage run are identical between paired backends:

| Family | Library | Training primitive calls | Verification primitive calls |
| --- | --- | ---: | ---: |
| Affine | Fixed | 36,007.7 | 248.5 |
| Affine | Growing | 40,847.6 | 292.3 |
| Parity | Fixed | 30,709.9 | 85.9 |
| Parity | Growing | 32,044.8 | 86.9 |

Every training evaluation checks that its actual primitive-call count equals the number of samples times the expanded active cost. Growing the library does not make a module invocation free. In the affine experiment, increased final-stage success accompanied a higher total primitive-call count.

The median paired ratio of ATF time to reference-backend time was:

| Family | Library | Median ATF/reference ratio | Interquartile range |
| --- | --- | ---: | --- |
| Affine | Fixed | 3.28 | 3.20–3.32 |
| Affine | Growing | 5.56 | 5.07–6.12 |
| Parity | Fixed | 3.05 | 3.02–3.09 |
| Parity | Growing | 3.10 | 3.08–3.19 |

These ratios sum setup, candidate compilation, training execution, and module admission over all three stages. They exclude final verification from those component timers; the separately recorded run wall time includes verification and proposal work. Artifact serialization occurs after the run wall timer. Complete component values and wall times are retained in `summary.json` and individual `result.json` files.

The reference backend expands modules at compile time, whereas V0 `EvolvedMethod` invokes a nested interpreter. Reference module compilation is cached within a stage, and V0 module processors initialize lazily and are reused within a stage. ATF candidate preparation includes the common type validator before native graph construction. These are descriptive timings of two scalar Python implementations on one host, not a representation-level speed theorem. They do not show an ATF speed advantage.

## Historical depth-cap audit

`reachability_audit.py` reads the historical `GraphEvo_exp3` source through its AST without importing or running that script. It extracts the actual unary affine method pools, verifies that task 3 used `max_init_layers=3`, and enumerates exact affine maps `a*x+b` using `fractions.Fraction`.

| Historical pool | Target `x+2.5` reachable within 3 calls? | Minimum calls | One shortest witness |
| --- | --- | ---: | --- |
| Minimal | No | 4 | increment → multiply-by-two → add-three → multiply-by-one-half |
| Extended, adding add-two | No | 4 | Same witness |
| Oracle, adding `f1_direct` | Yes | 3 | `f1_direct` → add-three → multiply-by-one-half |

The historical 13/100 versus 0/100 comparison therefore includes a representational-capacity difference: a reusable method can shorten a program under a fixed top-level call limit, while the fixed primitive pool cannot express the target under that limit. It does not independently establish a search advantage at matched expanded capacity. The new experiment uses a common expanded bound and a different, explicitly specified search and admission protocol; its counts are not a direct replication of the historical counts.

`reachability.json` contains every depth's syntactic-candidate and distinct-function counts, shortest witnesses, exact coefficients, and source hashes. This finite chain audit makes no claim about arbitrary DAGs.

## Pilot and provenance

The initial pilot used seeds `810901` and `810902`, with 64 evaluations per stage. Its records remain under `artifacts/pilot_initial/`. Before the primary source freeze, the affine verification points were shifted by `1/16` to remove four overlaps with training points, and duplicate validation of the reference backend was removed. The corrected pilot is retained separately under `artifacts/pilot/`.

`protocol_planned.json` describes the primary design. `protocol_frozen.json` records the final protocol, source hashes, and software environment before the 30-seed primary run. Pilot results are excluded from the primary success counts. The primary results and audit are under `artifacts/formal/`.

## Reproduction

Run commands from the repository root. The recorded environment used Python 3.12.9, NumPy 2.1.3, and NetworkX 3.4.2. The existing supplementary requirements file records the remaining package versions:

```console
python -B -m pip install -r experiments/rebuttal_20261106/requirements.txt
python -B experiments/typed_dag_comparison_20261108/run_comparison.py check
python -B experiments/typed_dag_comparison_20261108/reachability_audit.py --output experiments/typed_dag_comparison_20261108/reproduction/reachability.json
python -B experiments/typed_dag_comparison_20261108/run_comparison.py run --phase formal --output experiments/typed_dag_comparison_20261108/reproduction
python -B experiments/typed_dag_comparison_20261108/audit_results.py --directory experiments/typed_dag_comparison_20261108/reproduction/formal
```

Use a fresh output directory for an independent rerun. The runner resumes existing completed condition directories rather than overwriting them. Formal runs verify the frozen source hashes; the published protocol does not need to be regenerated. The output directory above keeps reproduction outputs separate from the archived results. Reproduced timing values may vary by host and load.

Each condition directory contains `result.json` and `trace.jsonl.gz`. The trace records every evaluation's candidate fingerprint, loss, expanded cost, actual primitive calls, acceptance, and restart status. The audit reconstructs candidates from the seed and protocol. `summary.json`, `summary.csv`, and `audit.json` provide aggregate records and verification results.

## Post-primary parity budget check

A separate parity-only sensitivity check was specified after the primary results showed no stage-2 or stage-3 successes at 256 evaluations. It increases the budget to 1,024 evaluations per stage while retaining the frozen methods, proposal rules, expanded capacity, admission policy, and the same 30 seeds. It is a post-primary analysis, not a replacement primary experiment or an additional independent seed sample.

The plan is recorded in `budget_sensitivity_protocol.json`, with a fixed completion rule of 120 runs irrespective of outcomes. Completed outputs are under `artifacts/parity_budget1024/`. Successes out of the same 30 seeds are:

| Parity stage | ATF, fixed | Typed DAG, fixed | ATF, growing | Typed DAG, growing |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16 | 16 | 16 | 16 |
| 2 | 0 | 0 | 6 | 6 |
| 3 | 0 | 0 | 1 | 1 |

All 180 paired stages matched. The independent arithmetic audit checked all 368,640 candidate-evaluation records and passed. At the larger budget, the growing-library condition produced six stage-2 successes and one stage-3 success in each backend; the fixed-library condition produced none at those stages. These secondary results are not incorporated into the primary tables above.

Run or resume this separate check and audit its completed output with:

```console
python -B experiments/typed_dag_comparison_20261108/budget_sensitivity.py
python -B experiments/typed_dag_comparison_20261108/audit_results.py --directory experiments/typed_dag_comparison_20261108/artifacts/parity_budget1024 --families parity
```

The longer-budget runs restart the complete three-stage experiment with the same seeds; they are not continuations of the saved 256-evaluation runs. All post-primary results should be reported separately, including null results.
