# LaTeX artifact index

## Required LaTeX packages

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{adjustbox}
```

## Tables

- `manuscript/latex/tables/table_best_proxy_leakage_by_dataset_model.tex`
- `manuscript/latex/tables/table_cross_dataset_claim_summary.tex`
- `manuscript/latex/tables/table_selector_family_compact_summary.tex`
- `manuscript/latex/tables/table_subset_vs_fairness_baselines.tex`

## Figures

- `manuscript/latex/figure_references.tex`
- `manuscript/latex/figure_manifest.csv`

## Suggested manuscript inclusion order

```latex
\input{tables/table_cross_dataset_claim_summary}
\input{tables/table_subset_vs_fairness_baselines}
\input{tables/table_best_proxy_leakage_by_dataset_model}
\input{tables/table_selector_family_compact_summary}
\input{figure_references}
```
