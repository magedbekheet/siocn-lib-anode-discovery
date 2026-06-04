# Notebook Run Comparison

Run date: 2026-06-01

Executed notebooks:

- `reports/executed_notebooks/02_advanced_modeling_feature_sets_groupcv_executed.ipynb`
- `reports/executed_notebooks/03_advanced_modeling_feature_sets_groupcv_executed.ipynb`
- `reports/executed_notebooks/04_final_sioc_phase_physics_modeling_executed.ipynb`

## Notebook 02

Best grouped-CV results:

| Feature set | Model | Features | Grouped-CV R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| C_echem_assisted_full | Gradient Boosting | 42 | 0.699 | 91.415 | 121.388 |
| B_structure_assisted | Gradient Boosting | 36 | 0.304 | 159.849 | 190.947 |
| A_design_stage | Gradient Boosting | 34 | 0.289 | 160.666 | 192.767 |

Grouped holdout:

| Feature set | Model | Holdout R2 | Holdout MAE | Holdout RMSE |
| --- | --- | ---: | ---: | ---: |
| C_echem_assisted_full | Gradient Boosting | 0.788 | 79.092 | 110.580 |
| B_structure_assisted | Gradient Boosting | 0.405 | 144.679 | 185.410 |
| A_design_stage | Gradient Boosting | 0.387 | 147.506 | 188.330 |

Interpretation: notebook 02 shows excellent performance for the full electrochemistry-assisted feature set, but this is not a valid virtual-discovery model because it uses target-adjacent performance descriptors such as irreversible capacity and Coulombic efficiency.

## Notebook 03

Best grouped-CV results:

| Feature set | Model | Features | Grouped-CV R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| A_design_stage | Gradient Boosting | 7 | 0.415 | 143.301 | 173.334 |
| B_structure_assisted | Gradient Boosting | 8 | 0.391 | 145.326 | 176.016 |
| C_echem_assisted_full | Gradient Boosting | 10 | 0.357 | 152.786 | 183.114 |

Grouped holdout:

| Feature set | Model | Holdout R2 | Holdout MAE | Holdout RMSE |
| --- | --- | ---: | ---: | ---: |
| A_design_stage | Gradient Boosting | 0.448 | 131.924 | 178.665 |
| B_structure_assisted | Gradient Boosting | 0.440 | 135.068 | 179.980 |
| C_echem_assisted_full | Gradient Boosting | 0.262 | 155.286 | 206.591 |

Interpretation: notebook 03 is much more useful for discovery than notebook 02. The reduced design-stage feature set is best under DOI-grouped validation and holdout.

## Notebook 04

Best grouped-CV results:

| Feature set | Model | Features | Grouped-CV R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| C_design_plus_measured_structure | Gradient Boosting | 14 | 0.352 | 151.391 | 183.031 |
| A_design_stage_final | Gradient Boosting | 12 | 0.344 | 151.578 | 183.599 |
| B_design_plus_phase_physics | Gradient Boosting | 17 | 0.346 | 153.216 | 184.874 |

Grouped holdout:

| Feature set | Model | Holdout R2 | Holdout MAE | Holdout RMSE |
| --- | --- | ---: | ---: | ---: |
| A_design_stage_final | Gradient Boosting | 0.361 | 164.645 | 193.762 |
| B_design_plus_phase_physics | Gradient Boosting | 0.350 | 167.133 | 195.450 |
| C_design_plus_nonleaky_echem_context | Gradient Boosting | 0.292 | 170.151 | 204.025 |

Interpretation: notebook 04 is the cleanest final workflow. It now uses the pre-pyrolysis schema, excludes voltage window from the primary model, and includes an exhaustive phase-physics ablation. Measured structure gives a very small improvement, but surface-area coverage is sparse, so it is best treated as a second-stage model after characterization.

Phase-physics ablation:

| Phase setting | MAE | Delta vs base |
| --- | ---: | ---: |
| ceramic_confinement_index + effective_bandgap_proxy_ev + phase_weighted_electronegativity | 149.304 | -2.274 |
| ceramic_confinement_index + phase_weighted_electronegativity | 149.437 | -2.140 |
| phase_weighted_electronegativity | 149.492 | -2.086 |
| base_design_only | 151.578 | 0.000 |
| all five phase-physics descriptors | 153.216 | 1.638 |

The best phase subset improves MAE by only about 2.27 mAh/g, smaller than the fold-to-fold uncertainty, so the primary model should remain the simpler design-stage model. Phase descriptors are still useful for discussion, interpretation, and candidate filtering.

## Overall conclusion

For discovering new SiOC/SiOCN anode compositions, trust DOI/reference-grouped validation, not random CV. Random CV is consistently optimistic because related rows from the same paper/material family can leak across folds.

Recommended workflow:

1. Use notebook 04 as the main scientific workflow.
2. Use notebook 03 as a compact ablation showing that a small design-stage feature set can outperform larger feature sets.
3. Treat notebook 02 as a literature-analysis notebook, not a virtual-discovery notebook, because its best result depends on post-test electrochemical descriptors.
4. For candidate discovery, train/rank with pre-test design features only, then use phase-physics descriptors as secondary interpretation or filtering until they prove stable under grouped validation.
