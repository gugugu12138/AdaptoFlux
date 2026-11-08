# Current revised paper

**AdaptoFlux: A Typed Graph Substrate for Stateful Program Construction and Reuse**

`AdaptoFlux_revised.pdf` is the current anonymous manuscript: nine pages of main text and 28 pages in total. `source/` contains its LaTeX sources, bibliography, conference style, and all figures required to rebuild it.

## Build

From `source/`, run:

```console
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build iclr2027_rebuttal_revision.tex
```

A standard TeX Live installation with pdfLaTeX, BibTeX, and latexmk is sufficient. Generated build files are not part of the published evidence.

## Evidence map

- Sections 4.2 and Appendix I: [matched search and library-growth controls](../../experiments/typed_dag_comparison_20261108/README.md). The controlled comparison uses an independent reference DAG backend; it is distinct from PyTensor.
- Section 4.8 and Appendix I: [PyTensor workflow and action-execution comparisons](../../experiments/typed_dag_workflow_20261108/README.md), including the 36 paired workflow stages, 2,638 candidate evaluations, and 21,200 controller replay steps.
- Sections 4.3-4.7 and Appendices A-H: [earlier supplementary experiments and frozen runtime](../../experiments/rebuttal_20261106/).
- [Extended historical technical reference](../extended_materials/README.txt): preserved independently, with its original numbering. The current manuscript is authoritative for current claims and section references.

Frozen experiments retain their original protocols, raw records, and audits. The controlled reference-DAG study attributes the observed search improvement to the shared module-reuse policy. The PyTensor study checks workflow and execution interfaces of fixed programs. Section 5 describes continual learning as future work.

`MANIFEST.sha256` records the exact bytes of all files in this directory except the manifest itself.
