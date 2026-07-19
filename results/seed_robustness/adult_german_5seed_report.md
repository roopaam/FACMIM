# Five-seed robustness analysis

## Configuration

- Seeds: `[42, 43, 44, 45, 46]`
- Models: `['logistic_regression', 'random_forest', 'gradient_boosting']`
- Feature budget k: `8`
- Lambdas: `[0.0, 0.5, 1.0, 2.0]`
- Adult sample size: `5000`
- German Credit sample size: `1000`
- Runtime seconds: `1589.7`

## Output files

- `adult_german_5seed_full_results.csv`
- `adult_german_5seed_summary_all_methods.csv`
- `adult_german_5seed_key_table.csv`
- `adult_german_5seed_key_table.tex`
- `adult_german_5seed_paired_joint_mi_differences.csv`

## Interpretation note

The robustness analysis is intended as a stability check for the main leakage-reduction claims. Positive paired differences in joint MI indicate lower leakage for Subset-aware FA-CMIM relative to CMIM under the same random seed.
