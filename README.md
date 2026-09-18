# MOGSK — Reproducibility Package

This package accompanies the revised manuscript *"A Multi-Objective
Gaining-Sharing Knowledge-Based Optimization Approach for Feature Selection
in Classification Problems"* and the corresponding response to reviewers.

## Contents

| File | Description |
|---|---|
| `code/mogsk_full_reproducibility_notebook.ipynb` | The full, executable Jupyter notebook: MOGSK implementation, all six baseline algorithms (NSGA-II, MOPSO, MOEA/D, BPSO, BHHO, BGWO), the nested cross-validation protocol, and every table/figure generation script used in the revised manuscript. |
| `code/mogsk_full_pipeline.py` | A flat, non-interactive `.py` export of the same notebook (markdown cells kept as comment blocks) for environments without Jupyter. |
| `code/master_results.json` | Consolidated results: mean/SD of accuracy, Hypervolume, and selected-feature count for every algorithm × dataset × classifier-backend combination, plus per-run completion counts. |
| `code/table3_accuracy_SVM.csv` | MOGSK accuracy and feature-reduction results, SVM backend, all 10 datasets. |
| `code/table7_hypervolume_{SVM,KNN,RF}.csv` | Hypervolume results grouped by classifier backend. |
| `code/table8_ablation_all8.csv` | Ablation study (Baseline / FS-only / PT-only / Joint accuracy and superadditivity gap), all datasets × backends. |
| `code/table9_sensitivity.csv` | Hyperparameter sensitivity analysis (SpamBase). |
| `code/table10_wilcoxon_all8.csv` | Wilcoxon signed-rank test results, all datasets × backends, Bonferroni-corrected. |
| `code/runtime_complexity.csv` | Wall-clock runtime (seconds) and archive size per algorithm × dataset × backend. |
| `code/hyperparameter_ranges.csv` | Search-space bounds used for each classifier backend (SVM, KNN, RF). |

## How to reproduce

1. Open `mogsk_full_reproducibility_notebook.ipynb` in Jupyter (or run
   `mogsk_full_pipeline.py` directly with Python).
2. Run cells top to bottom. The experiment runner is **resumable**: partial
   progress is checkpointed, so an interrupted run can be restarted without
   losing completed runs.
3. `master_results.json` and the per-table CSV files listed above are
   (re)generated automatically; the figures used in the manuscript are
   produced by the plotting cells at the end of the notebook.

## Important note on run count (N)

The results reported in this revision are based on **N = 11 completed
independent runs** per dataset × backend × algorithm configuration (the
notebook is configured for the originally planned N = 31; some runs remain
in progress at the time of this submission due to wall-clock runtime,
see `runtime_complexity.csv`). We flag this explicitly rather than
reporting the target N; the authors are extending the run count to the
full N = 31 ahead of any camera-ready version and will update all tables
accordingly. Sensitivity checks (comparing standard error at N = 11 to the
magnitude of the corrected-vs-original discrepancies) indicate the
qualitative conclusions reported in the manuscript are not expected to
change with additional runs, but exact figures may shift slightly.

## Validation protocol (why results differ from the original submission)

In response to Reviewer 1 (comment 4.1) and Reviewer 2 (comment on
CV placement), this codebase implements a **strictly nested
cross-validation protocol**:

- An outer stratified train/test split (80/20) is performed once per run.
- The outer test fold is **never** passed to the fitness function, the
  MOGSK search process, or any baseline algorithm during optimisation.
- Fitness evaluation during the search uses k-fold CV computed only on the
  outer training partition.
- After the search terminates, the best archive member is refit on the
  full outer training partition and scored **exactly once** on the
  held-out outer test fold.

This differs from the evaluation protocol used to produce the originally
submitted manuscript's numbers, and is the primary reason several
figures (notably the Lung Cancer accuracy gain and the mean feature
reduction percentage) differ between the original submission and this
revision. See the Response to Reviewers letter (comment 4.1 / comment 4)
for the full explanation.

## Software / hardware environment

Please record and complete the following before final submission (fields
left as placeholders should be filled in with the actual environment used
to generate the reported results):

- Python version: `3.10.16`
- Key library versions: scikit-learn `1.4.0`, numpy `1.26.3`, scipy `1.12.0`,
  pandas `2.0.0`
- CPU: `Linux x86_64 gvisor (multi-core cloud environment)`
- RAM: `Standard Cloud RAM Allocation`
- Approximate total wall-clock time for the full experiment suite: see
  `runtime_complexity.csv`.

## Contact

Corresponding author: Shereen Fathy El-Feky (selfeky@msa.edu.eg)
