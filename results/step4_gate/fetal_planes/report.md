# Matrix results

Metric: `balanced_accuracy`

## Cells

| backbone   | regime   | train_variant   | test_variant   |   balanced_accuracy_mean |   balanced_accuracy_sd |   ece_mean |   confidence_minus_accuracy_mean |   n_runs |
|:-----------|:---------|:----------------|:---------------|-------------------------:|-----------------------:|-----------:|---------------------------------:|---------:|
| resnet50   | r_naive  | a_full          | a_full         |                 0.956723 |             0.00374301 |  0.0238542 |                        0.0208543 |        6 |
| resnet50   | r_naive  | a_full          | a_only         |                 0.176081 |             0.0185465  |  0.508427  |                        0.506714  |        6 |
| resnet50   | r_naive  | a_full          | a_removed      |                 0.9569   |             0.00355676 |  0.0238705 |                        0.0205193 |        6 |
| resnet50   | r_naive  | a_full          | ctrl_random    |                 0.956465 |             0.004406   |  0.0234948 |                        0.0202145 |        6 |

## Hypothesis tests (Holm-Bonferroni across the family)

| test | n pairs | mean difference | effect size dz | p raw | p adjusted | reject |
|---|---|---|---|---|---|---|
| H1_a_only_above_chance | 6 | 0.0094 | 0.508 | 2.69e-01 | 1.00e+00 | False |
| H1_removal_costs_performance | 6 | -0.0002 | -0.132 | 7.60e-01 | 1.00e+00 | False |
| R5_annotation_vs_equal_area_control | 6 | 0.0004 | 0.269 | 5.39e-01 | 1.00e+00 | False |
| H2_overconfidence_under_removal | 6 | -0.0003 | -0.190 | 6.61e-01 | 1.00e+00 | False |

## Fold vs seed variance

If `within_fold_var_across_seeds` is comparable to `total_var`, single-run
comparisons in this configuration are meaningless and only distributions
should be reported.

```json
{
  "r_naive:a_full": {
    "total_var": 1.4010155600616125e-05,
    "within_fold_var_across_seeds": 1.843942917426997e-05,
    "within_seed_var_across_folds": 1.722982626661977e-05,
    "mean": 0.9567225917499566,
    "sd": 0.003743014239969723,
    "n_runs": 6
  },
  "r_naive:a_only": {
    "total_var": 0.0003439724884805653,
    "within_fold_var_across_seeds": 0.00037798415114530075,
    "within_seed_var_across_folds": 0.00020568372248397024,
    "mean": 0.17608081794307487,
    "sd": 0.01854649531530325,
    "n_runs": 6
  },
  "r_naive:a_removed": {
    "total_var": 1.2650526029858299e-05,
    "within_fold_var_across_seeds": 1.619472125825749e-05,
    "within_seed_var_across_folds": 1.544629324406095e-05,
    "mean": 0.9569003623016271,
    "sd": 0.0035567577974692484,
    "n_runs": 6
  },
  "r_naive:ctrl_random": {
    "total_var": 1.9412841419359653e-05,
    "within_fold_var_across_seeds": 3.067589768939357e-05,
    "within_seed_var_across_folds": 2.303420243837044e-05,
    "mean": 0.9564649664900228,
    "sd": 0.0044060006149976485,
    "n_runs": 6
  }
}
```
