Extended technical reference
============================

This directory preserves the full technical material accompanying AdaptoFlux,
including detailed derivations, historical experiment configurations and run
tables, model illustrations, profiling figures, and implementation descriptions.
The compact paper retains the execution contract, necessary protocols, and
the results used to assess its main contributions.

Files
-----
extended_reference.pdf is the complete self-contained reference preceding
appendix condensation. Its original main text, references, and appendix
numbering are retained so that every extended table, figure, and citation can
be interpreted in context. The current paper is authoritative for its claims
and section references.

source/ contains the LaTeX source, bibliography, conference style, and every
referenced figure needed to rebuild this fixed reference. Build
iclr2027_rebuttal_revision.tex with pdfLaTeX and BibTeX, placing generated
files in a separate output directory.

Material map (sections in extended_reference.pdf)
------------------------------------------------
I: related-work discussion and representation positioning.
J: complete inference/training cost derivations and algorithm details.
K: original search, transfer, scheduling, and stateful-task protocols/results.
L: geometric-task configurations, model illustrations, and learning curves.
M: restricted symbolic-regression protocols, result tables, and expressions.
N: hand-crafted Snake controller and graph illustration.
O: decorator implementation, experiment paths, and environment instructions.
P: metadata examples, boundary interfaces, and runtime-loop semantics.
Q-U: stateful-policy challenges, prospective integration, and discussion.

Frozen supplementary evidence
-----------------------------
The four controlled supplementary studies, their original runtime snapshot,
and portable reproduction entry point are in experiments/rebuttal_20261106/
at the repository root. This reference introduces no additional measurements.

Integrity
---------
MANIFEST.sha256 records the exact bytes of all files in this directory except
the manifest itself. Original data and plots have not been replaced by a rerun.
