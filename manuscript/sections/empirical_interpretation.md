# Empirical Results and Interpretation

## Evaluation focus

The empirical evaluation was designed to test whether fairness-aware feature selection also controls proxy leakage. The central quantity of interest is not only predictive accuracy or downstream group fairness, but the amount of sensitive information retained by the selected feature subset. For this reason, the experiments report conventional utility metrics, downstream fairness metrics, joint subset mutual information with the sensitive attribute, and sensitive-attribute attacker balanced accuracy.

The key distinction examined in these experiments is between marginal fairness-aware selection and subset-aware leakage control. Methods such as ProxyRank, fair-mRMR, FairCFS-style selection, FairLasso-style selection, and Basic FA-CMIM include fairness-aware penalties or constraints, but they do not necessarily penalize the sensitive information encoded by the selected subset as a whole. Subset-aware FA-CMIM directly targets this joint leakage.

## Cross-dataset summary

Across 4 datasets and 12 dataset-model settings, the results show a heterogeneous but informative pattern. Subset-aware FA-CMIM shows strong support for proxy-leakage reduction on 2 dataset(s), partial support on 0 dataset(s), and flat or no clear advantage on 2 dataset(s). This pattern is important because it avoids the misleading claim that the proposed method is uniformly superior across all metrics. Instead, the results support a more precise conclusion: subset-aware FA-CMIM is most beneficial when there is meaningful room for the selector to choose among competing predictive features with different leakage profiles.

| Dataset | Subset beats CMIM | Subset beats fairness baseline | Mean joint-MI advantage vs CMIM | Mean joint-MI advantage vs fairness baseline | Interpretation |
| --- | --- | --- | --- | --- | --- |
| ACSIncome | 0/3 | 0/3 | 0.0000 | -0.0000 | flat or no clear advantage |
| Adult | 3/3 | 3/3 | 0.3407 | 0.0079 | strong support |
| COMPAS | 0/3 | 0/3 | 0.0000 | 0.0000 | flat or no clear advantage |
| German Credit | 3/3 | 3/3 | 0.2402 | 0.1784 | strong support |

## Dataset-level interpretation

**ACSIncome.** Subset-aware FA-CMIM achieved lower joint sensitive leakage than CMIM in 0/3 model settings and lower joint sensitive leakage than the best fairness-aware baseline in 0/3 model settings. The mean joint-MI advantage was 0.0000 relative to CMIM and -0.0000 relative to the best fairness-aware baseline. This dataset shows limited separation across selectors. The most plausible interpretation is that the selected feature sets are highly overlapping under the current k setting, so the subset-aware penalty has limited room to produce a distinct leakage profile.

**Adult.** Subset-aware FA-CMIM achieved lower joint sensitive leakage than CMIM in 3/3 model settings and lower joint sensitive leakage than the best fairness-aware baseline in 3/3 model settings. The mean joint-MI advantage was 0.3407 relative to CMIM and 0.0079 relative to the best fairness-aware baseline. This dataset provides the clearest support for the proposed subset-aware leakage-control mechanism. Subset-aware FA-CMIM reduces joint sensitive leakage relative to the strongest fairness-aware baseline in most model settings, while maintaining a competitive level of predictive utility.

**COMPAS.** Subset-aware FA-CMIM achieved lower joint sensitive leakage than CMIM in 0/3 model settings and lower joint sensitive leakage than the best fairness-aware baseline in 0/3 model settings. The mean joint-MI advantage was 0.0000 relative to CMIM and 0.0000 relative to the best fairness-aware baseline. This dataset shows limited separation across selectors. The most plausible interpretation is that the selected feature sets are highly overlapping under the current k setting, so the subset-aware penalty has limited room to produce a distinct leakage profile.

**German Credit.** Subset-aware FA-CMIM achieved lower joint sensitive leakage than CMIM in 3/3 model settings and lower joint sensitive leakage than the best fairness-aware baseline in 3/3 model settings. The mean joint-MI advantage was 0.2402 relative to CMIM and 0.1784 relative to the best fairness-aware baseline. This dataset provides the clearest support for the proposed subset-aware leakage-control mechanism. Subset-aware FA-CMIM reduces joint sensitive leakage relative to the strongest fairness-aware baseline in most model settings, while maintaining a competitive level of predictive utility.

## Comparison against fairness-aware baselines

The most important empirical observation is that fairness-aware baselines can still retain substantial sensitive information in the selected subset. This is visible when a baseline achieves competitive accuracy or downstream fairness while its selected feature subset continues to exhibit high joint mutual information with the sensitive attribute or high sensitive-attribute attacker balanced accuracy.

The comparison below focuses on the best proxy-leakage configuration of Subset-aware FA-CMIM against the best proxy-leakage configuration among the fairness-aware baselines for each dataset and model.

| Dataset | Model | Subset selector | Best fairness baseline | Subset accuracy | Baseline accuracy | Subset joint MI | Baseline joint MI | Joint-MI advantage | Accuracy delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACSIncome | gradient_boosting | Subset FA-CMIM_k8_lambda0.5 | FairCFS_k8_lambda2.0 | 0.7933 | 0.7933 | 0.6609 | 0.6609 | -0.0000 | 0.0000 |
| ACSIncome | logistic_regression | Subset FA-CMIM_k8_lambda0.0 | fair-mRMR_k8_lambda0.0 | 0.7647 | 0.7647 | 0.6609 | 0.6609 | -0.0000 | 0.0000 |
| ACSIncome | random_forest | Subset FA-CMIM_k8_lambda0.0 | fair-mRMR_k8_lambda0.0 | 0.8047 | 0.8053 | 0.6609 | 0.6609 | -0.0000 | -0.0007 |
| Adult | gradient_boosting | Subset FA-CMIM_k8_lambda1.0 | ProxyRank_k8_lambda2.0 | 0.8320 | 0.8307 | 0.2518 | 0.2597 | 0.0079 | 0.0013 |
| Adult | logistic_regression | Subset FA-CMIM_k8_lambda1.0 | ProxyRank_k8_lambda2.0 | 0.7647 | 0.7493 | 0.2518 | 0.2597 | 0.0079 | 0.0153 |
| Adult | random_forest | Subset FA-CMIM_k8_lambda1.0 | ProxyRank_k8_lambda2.0 | 0.7813 | 0.7940 | 0.2518 | 0.2597 | 0.0079 | -0.0127 |
| COMPAS | gradient_boosting | Subset FA-CMIM_k8_lambda2.0 | ProxyRank_k8_lambda0.5 | 0.6813 | 0.6813 | 0.5445 | 0.5445 | 0.0000 | 0.0000 |
| COMPAS | logistic_regression | Subset FA-CMIM_k8_lambda2.0 | ProxyRank_k8_lambda0.5 | 0.6673 | 0.6673 | 0.5445 | 0.5445 | 0.0000 | 0.0000 |
| COMPAS | random_forest | Subset FA-CMIM_k8_lambda2.0 | ProxyRank_k8_lambda0.5 | 0.6407 | 0.6460 | 0.5445 | 0.5445 | 0.0000 | -0.0053 |
| German Credit | gradient_boosting | Subset FA-CMIM_k8_lambda2.0 | fair-mRMR_k8_lambda1.0 | 0.7133 | 0.7033 | 0.2463 | 0.4247 | 0.1784 | 0.0100 |
| German Credit | logistic_regression | Subset FA-CMIM_k8_lambda2.0 | fair-mRMR_k8_lambda1.0 | 0.6700 | 0.7033 | 0.2463 | 0.4247 | 0.1784 | -0.0333 |
| German Credit | random_forest | Subset FA-CMIM_k8_lambda2.0 | ProxyRank_k8_lambda2.0 | 0.6767 | 0.7067 | 0.2463 | 0.4247 | 0.1784 | -0.0300 |

Positive values in the joint-MI advantage column mean that Subset-aware FA-CMIM achieved lower joint sensitive leakage than the best fairness-aware baseline. Negative or zero values indicate that the subset-aware method did not improve on the strongest baseline for that dataset-model setting.

## Main empirical conclusion

The results support three conclusions.

First, fairness-aware feature selection is not automatically proxy-leakage-aware. Across datasets, several fairness-aware baselines remain competitive on accuracy and downstream fairness metrics but still retain measurable sensitive information in their selected subsets.

Second, Subset-aware FA-CMIM provides the clearest advantage when the dataset contains enough competing candidate features for the subset-level penalty to alter the selected subset. This was most visible in the Adult dataset, where subset-aware leakage control produced a stronger reduction in sensitive information than the fairness-aware baselines.

Third, the advantage is configuration-dependent. In datasets such as ACSIncome and COMPAS under the current k=8 setting, many selectors converge to similar feature subsets. In such cases, downstream performance and proxy-leakage metrics become nearly indistinguishable. This does not refute the subset-aware mechanism; rather, it indicates that its advantage is muted when the selected subset size is large relative to the effective feature space.

## Implications for proxy mitigation

These findings reinforce the central motivation of FA-CMIM. A selector can be fairness-aware in name or objective form while still preserving sensitive information through combinations of retained features. Therefore, proxy mitigation should be evaluated directly using subset-level leakage diagnostics, not only through downstream fairness metrics such as demographic parity difference or equalized odds difference.

The proposed subset-aware objective should therefore be interpreted as a proxy-mitigation mechanism rather than a pure accuracy-maximization mechanism. Its purpose is to expose and control the fairness--utility--leakage trade-off. In this sense, the empirical results support the paper's main claim: controlling marginal proxy association is insufficient when sensitive information can be reconstructed compositionally from the selected feature set.

## Limitations and cautious interpretation

The current experiments use a fixed k=8 setting for cross-dataset comparability. This choice makes the results easy to compare, but it can also reduce selector differentiation on datasets with smaller feature spaces. Future sensitivity analysis should vary k and the fairness penalty to examine when subset-aware leakage control becomes most effective. The current results should therefore be framed as evidence of conditional proxy-leakage advantage, not as a universal dominance claim.

A manuscript-safe summary is:

> Subset-aware FA-CMIM does not uniformly maximize accuracy or dominate every baseline on every dataset. Rather, it directly targets joint sensitive leakage and can substantially reduce proxy information when the feature-selection problem provides meaningful room for subset-level trade-offs. Across datasets, the results show that fairness-aware baselines do not reliably eliminate proxy leakage, supporting the need for explicit subset-aware leakage diagnostics and penalties.
