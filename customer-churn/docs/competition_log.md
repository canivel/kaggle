# Customer Churn Competition Log
## Kaggle Playground Series S6E3

Tracking all experiments, learnings, and results.

---


### Data Loading
**Date**: 2026-03-26T19:50:56.946171

**Results**:
- train_shape: (594194, 19)
- test_shape: (254655, 19)
- churn_rate: 0.2252
- null_count: 0

---

### Feature Engineering
**Date**: 2026-03-26T19:50:59.676458

Created 46 features using ratios, interactions, frequency encoding, groupby stats, and binning.

**Results**:
- n_features: 46
- feature_steps: 9

---

### Experiment Loop - 16 Strategies
**Date**: 2026-03-26T20:15:04.363751

Ran 16 experiments, 3 kept.

**Results**:
- total_experiments: 16
- kept: 3
- best_cv: 0.916312

**Learnings**:
- lgbm (LightGBM baseline with default params): 0.916156
- lgbm (LightGBM deeper trees): 0.916217
- lgbm (LightGBM low LR high reg): 0.916312

---

### Stacked Ensemble (3 models)
**Date**: 2026-03-26T20:24:21.354665

**Results**:
- Individual scores: {"lgbm": 0.9163124174452163, "lgbm_deep": 0.916216946891925, "catboost": 0.916128698660627}
- Ensemble score: 0.916469
- Submission: submissions/ensemble_v1.csv

**Learnings**:
- Best single model: LGBM low LR = 0.916312
- Ensemble of 3 models (2 LGBM variants + CatBoost)
- XGBoost excluded due to API crash (fixed for next iteration)

**Next Steps**:
- Include XGBoost in ensemble (API now fixed)
- Optuna tuning (50 trials per model)
- More feature engineering (target encoding, more interactions)
- Multi-seed averaging for variance reduction

---
