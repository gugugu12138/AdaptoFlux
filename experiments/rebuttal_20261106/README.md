# AdaptoFlux supplementary experiments

This package contains the frozen evidence and reproduction scripts for the type-prior study, matched implicit/explicit state encodings, fixed-depth execution scaling, and frozen action-controller replay. These studies use the bundled original **V0 runtime**. They do not benchmark the revised effect-ordering runtime in the current repository.

Use `reproduce.py` to run a study. It verifies the published checksums, copies this package into a new workspace, installs the bundled V0 source into that workspace, and starts the original study script in a separate Python process. The source release is read-only to the launcher. All generated results, plots, logs, and caches stay in the chosen output directory.

## Environment

The recorded environment used Python 3.12.9 on Windows and the versions in `requirements.txt`. A Python 3.12 virtual environment is recommended. The launcher itself requires Python 3.10 or newer.

From this package directory:

```text
python -m pip install -r requirements.txt
```

The V0 package imports NumPy, NetworkX, pandas, Matplotlib, and SciPy. The scaling runner additionally uses psutil. The frozen controller replay uses NumPy and an isolated reference interpreter; it does not need a separate training repository or Numba installation.

**Fresh scaling measurements require Windows PowerShell**, as used by the original measurement protocol. The frozen runner records Windows processor and process-affinity information. Use `scaling-audit` to inspect the published scaling evidence without taking new timings. Execution times and rendered figure bytes can vary with hardware, operating system, package versions, and system load.

## Commands

Every invocation requires a different, nonexistent `--output` directory **outside the repository and this package**. An existing directory, even an empty one, is rejected. Replace the example output paths below with suitable new locations on your machine.

```text
python reproduce.py type-audit --output /outside/repository/atf-type-audit
python reproduce.py encoding --output /outside/repository/atf-encoding
python reproduce.py scaling-audit --output /outside/repository/atf-scaling-audit
python reproduce.py action --output /outside/repository/atf-action
```

For example, on Windows an output path can be `C:/atf-reproductions/type-audit-01`, provided it is new and outside the checkout.

| Command | Work performed | Published reference |
|---|---|---|
| `type-audit` | Audit all saved search runs, evaluation callbacks, candidate records, and reconstructed acceptance counts; regenerate the textual analysis in the working copy. | 90 runs; 30 shared seeds per type condition; strict successes 29/30, 15/30, and 1/30. |
| `encoding` | Execute **all 1,007 catalogue records** in both encodings, then time the 90 final graphs with five paired repetitions and regenerate the analysis. | 20,140 paired initial conditions and 70,670 paired environment steps; all compared values match. |
| `scaling-audit` | Check frozen source/protocol/data provenance and recompute count, median, and reference checks; regenerate the plot from existing measurements. | 30 graphs, 32–1,024 operation nodes, fixed depth 8 and batch size 32. |
| `action` | Replay the frozen seed-104 bird controller for the first 20 fixed test episodes, using the original ATF executor and the bundled references. | 20,480 matched steps, plus 49 boundary probes and 49 reordered checks. |

`type-audit` and `scaling-audit` analyze recorded evidence; they do not repeat training or timing. `encoding` executes the paired audit and takes new timings. `action` performs inference replay, not controller training. The 1,007 catalogue records include repeated final graphs. The explicit/implicit comparison translates identical candidates after search; it is not an independent explicit-state search baseline.

Optional fresh experiments use the same fixed study settings:

```text
python reproduce.py type-search --output /outside/repository/atf-type-search
python reproduce.py scaling --output /outside/repository/atf-scaling-rerun
```

`type-search` runs the 90 formal searches with the published frozen protocol, then audits their outputs. `scaling` reruns the pilot, freezes a new measurement record using the original settings, measures all 30 formal graphs, and audits the new outputs. New timings, timestamps, and environment records are newly measured evidence; they are not substitutes for the original frozen files.

Before these commands overwrite fixed output names in the **working copy**, the launcher moves their copied published results to `<output>/archived_published/`. No published source files are deleted, rewritten, or rehashed to make a check pass. `encoding` similarly archives its copied previous result directory before writing new paired results. Action replay uses the separate `reproduction` output label.

To verify and prepare a workspace without executing a study:

```text
python reproduce.py encoding --output /outside/repository/atf-prepared --prepare-only
```

Preparation is an inspection step, not a completed experiment. Use another new output directory for an actual run; the launcher does not resume or reuse an existing output directory.

## Output layout

```text
<output>/
  reproduction_record.json
  logs/
  archived_published/                 # when an output stage is regenerated
  workspace/
    ATF/                             # copied V0 runtime
    experiments/
      minimal_stateful_comparison/   # copied V0 environment and operators
      rebuttal_20261106/              # working copy of this package
```

The result paths under the working package are:

- `type_search/formal/audit.json`, `summary.json`, and `per_seed.csv`;
- `state_encoding/formal/summary.json`, `analysis.json`, `candidate_results.csv`, and `timing_raw.csv`;
- `scaling/audit.json`, `summary.csv`, `per_graph.csv`, `timing_samples.csv`, and `scaling_summary.svg`;
- `action_example/results_reproduction.json`, `boundary_checks_reproduction.json`, and `native_trace_bird_reproduction.npz`.

`reproduction_record.json` records the command, Python/package versions, status, and source-file checksums. Successful completion also requires that the source package's file inventory is unchanged. A failed command retains its workspace and logs for inspection and exits with a nonzero status.

## Provenance and scope

`release_manifest.json` records each published file's SHA-256. Where anonymization changed a file, `source_sha256` and `transformation` identify its source digest and the change. Original frozen hashes remain original hashes. The release verifier resolves the documented mapping; it does not claim that anonymized bytes equal the original bytes.

The bundled `runtime_v0/` preserves the original study runtime and environment source, including line endings needed for byte-level provenance checks. Current repository runtime changes are not imported during these reproductions. `action_example/reference/` contains the frozen model, graph, test world, archived trace, and reference code required for the bird replay.

The earliest state-encoding semantic-audit script was not saved separately. The release retains the original semantic records and protocol hashes, and supplies the later script that added `--timing-only`, with its hash recorded in the timing protocol.

The package covers these four supplementary studies. It does not retrain the four-task CMA controller family or reproduce every subsequent application study. The action replay checks a fixed controller and a fixed test prefix; it is not a new generalization estimate. The measured scaling range is a tested range, not an upper bound on supported graph size.
