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

### CRITICAL: First LB Result - Overfitting Detected
**Date**: 2026-03-26T20:46:11.054948

**Results**:
- CV Score: 0.91647
- Public LB: 0.91380
- CV-LB Gap: 0.00267 (TOO HIGH)
- Top LB: 0.91762
- Gap to top: 0.00382

**Root Cause Analysis**:
1. Too many engineered features (46) - many groupby stats have near-zero importance but add noise
2. Frequency encoding may overfit to train distribution
3. Stacking meta-learner (logistic regression) may overfit OOF predictions
4. Need to check if target encoding or groupby stats leak information

**Strategy Change**:
1. Test MINIMAL feature set (original 19 + key ratios only) and compare LB
2. Use simple averaging or rank averaging instead of stacking
3. Increase model regularization significantly
4. Add XGBoost for model diversity (just fixed)
5. Focus on LB score, not CV score

---

### Iteration 2 Submissions
**Date**: 2026-03-26T22:30

**Submissions Today (2/5 used)**:
1. `ensemble_v1.csv` - Stacked 2xLGBM+CatBoost, 46 features, logistic meta → **LB: 0.91380**
2. `iter2_rank_avg_3models.csv` - Rank avg 3 LGBM, 46 features → LB: pending

**Key Finding from Iteration 2 experiments**:
- XGBoost now working: xgb_reg_minimal scored 0.91648 CV (best individual model!)
- Minimal features (21) perform EQUAL to full features (46) on CV
- This confirms: extra features (groupby stats, freq encoding) add noise, not signal
- CV-LB gap of 0.00267 is severe overfitting

**Score Tracker**:
| Submission | CV | LB | Gap |
|------------|----|----|-----|
| ensemble_v1 (stacked) | 0.91647 | 0.91380 | 0.00267 |
| rank_avg_3models | - | pending | - |
| Top LB (Chris Deotte) | - | 0.91762 | - |

---
