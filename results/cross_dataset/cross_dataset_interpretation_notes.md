# Cross-dataset interpretation notes

Primary proxy-leakage metric: `joint_subset_mi_sensitive`.

Lower values indicate less sensitive information retained by the selected feature subset.

## Dataset-level claim summary

### acs_income

- Models where Subset-aware FA-CMIM beats CMIM on joint MI: 0 / 3
- Models where Subset-aware FA-CMIM beats the best fairness-aware baseline on joint MI: 0 / 3
- Mean joint-MI advantage versus CMIM: 0.000000
- Mean joint-MI advantage versus best fairness-aware baseline: -0.000000
- Interpretation: `flat_or_no_subset_proxy_leakage_advantage`

### adult

- Models where Subset-aware FA-CMIM beats CMIM on joint MI: 3 / 3
- Models where Subset-aware FA-CMIM beats the best fairness-aware baseline on joint MI: 3 / 3
- Mean joint-MI advantage versus CMIM: 0.340729
- Mean joint-MI advantage versus best fairness-aware baseline: 0.007881
- Interpretation: `strong_support_for_subset_proxy_leakage_advantage`

### compas

- Models where Subset-aware FA-CMIM beats CMIM on joint MI: 0 / 3
- Models where Subset-aware FA-CMIM beats the best fairness-aware baseline on joint MI: 0 / 3
- Mean joint-MI advantage versus CMIM: 0.000000
- Mean joint-MI advantage versus best fairness-aware baseline: 0.000000
- Interpretation: `flat_or_no_subset_proxy_leakage_advantage`

### german_credit

- Models where Subset-aware FA-CMIM beats CMIM on joint MI: 3 / 3
- Models where Subset-aware FA-CMIM beats the best fairness-aware baseline on joint MI: 3 / 3
- Mean joint-MI advantage versus CMIM: 0.240169
- Mean joint-MI advantage versus best fairness-aware baseline: 0.178352
- Interpretation: `strong_support_for_subset_proxy_leakage_advantage`

## Safe manuscript-level conclusion

The results should be interpreted as evidence that fairness-aware feature selection does not necessarily eliminate proxy leakage. Subset-aware FA-CMIM directly targets joint subset leakage and may provide stronger proxy mitigation when selectors have meaningful room to choose among competing features. However, in datasets where k is large relative to the available feature space, selectors can converge to similar subsets, which mutes cross-method leakage differences.
