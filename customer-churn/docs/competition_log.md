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

### Iteration 3 Results (with ORIG_proba)
**Date**: 2026-03-27T01:11:18.134744

- lgbm_s42: 0.916498
- lgbm_s11: 0.916435
- lgbm_reg: 0.916605
- catboost: 0.916284
- logistic: 0.911472
- **Simple avg**: 0.916512
- **Rank avg**: 0.916482
- **Tree avg**: 0.916809

---

### Iteration 4 - Fixed Leakage + All Top Techniques
**Date**: 2026-03-27T03:11:56.519766

**Key changes**: All FE inside CV folds, 10-fold CV, digit features, bi-gram cats, target encoding

- lgbm_s42: 0.917807
- lgbm_s11: 0.917811
- lgbm_reg: 0.917545
- catboost: 0.916917
- logistic: 0.911730
- **Tree avg**: 0.917925
- **All avg**: 0.917524
- **Rank avg**: 0.917480

---

### Iteration 4 - Fixed Leakage + All Top Techniques
**Date**: 2026-03-27T03:17:44.560072

**Key changes**: All FE inside CV folds, 10-fold CV, digit features, bi-gram cats, target encoding

- lgbm_s42: 0.917794
- lgbm_s11: 0.917844
- lgbm_reg: 0.917560
- catboost: 0.916924
- logistic: 0.911730
- **Tree avg**: 0.917919
- **All avg**: 0.917517
- **Rank avg**: 0.917473

---

### Iteration 5 - Full Top-Notebook Reproduction
**Date**: 2026-03-27T09:19:59.841502

**New techniques**: Ridge stage1, XGB BlamerX params, distribution features, extended digit features, tri-grams, pseudo-labeling

- lgbm_3seed: 0.918023
- lgbm_reg: 0.917571
- lgbm_ridge: 0.917695
- xgb_blamerx: 0.917770
- catboost: 0.916870
- logistic: 0.912990
- lgbm_pseudo: 0.917647
- **Tree avg**: 0.917909
- **All avg**: 0.917618
- **Rank avg**: 0.917592

---

### Iteration 5 - Full Top-Notebook Reproduction
**Date**: 2026-03-27T09:25:40.139083

**New techniques**: Ridge stage1, XGB BlamerX params, distribution features, extended digit features, tri-grams, pseudo-labeling

- lgbm_3seed: 0.918007
- lgbm_reg: 0.917545
- lgbm_ridge: 0.917724
- xgb_blamerx: 0.917782
- catboost: 0.916866
- logistic: 0.912998
- lgbm_pseudo: 0.917678
- **Tree avg**: 0.917898
- **All avg**: 0.917606
- **Rank avg**: 0.917581

---

### Iteration 6 - BlamerX Reproduction
**Date**: 2026-03-27T15:51:14.562064

**CV**: 0.918790 (target: 0.91927)

**Approach**: 20-fold Ridge→XGB, N-gram TE, ORIG_proba, distribution features, digit features, exact BlamerX XGB params (lr=0.0063, colsample=0.32, reg_alpha=3.5)

---

### Adversarial Validation + Post-Processing
**Date**: 2026-03-27

**Adversarial Validation**: AUC = 0.511 → Train/test distributions are IDENTICAL.
The CV-LB gap (~0.0027) is NOT from distribution shift. It's from overfitting to fold structure.

**Iter6 LB**: 0.91603 (CV 0.91879, gap 0.00276)
**Gap to #1**: 0.00159

**Rank Calibration**: Generated blends in rank space (80/20, 70/30, 90/10).
Submitted 80/20 blend.

**Next**: Need novel approaches beyond GBDT. Researching cutting-edge tabular ML papers.

---

### Iteration 7 - 20-Seed + Hill Climbing
**Date**: 2026-03-27T19:18:21.309468

- 20-seed LGBM + 10-seed XGB + LogReg = 31 models
- Simple avg: 0.918209
- Top-10 avg: 0.918265
- **Hill climb**: 0.918296

---

### Iteration 9 - Improved BlamerX + RealMLP
**Date**: 2026-03-27T22:44:55.523067

- xgb: 0.918701
- lgbm: 0.918473
- realmlp: 0.913976
- XGB+LGBM avg: 0.918671
- **Deotte 3-model**: 0.918197

---

### Iteration 10 - Optuna + Pair TE + Pseudo-labels
**Date**: 2026-03-28T01:55:57.328049

**Optuna best params**: {'lr': 0.006612729138639732, 'max_depth': 5, 'min_child_weight': 3, 'subsample': 0.8807689432639139, 'colsample_bytree': 0.24473038620786253, 'reg_alpha': 9.133995846860973, 'reg_lambda': 2.0736445177905036, 'gamma': 0.3974313630683448}

- optuna_xgb: 0.915885
- blamerx_xgb: 0.915878
- pair_te_logreg: 0.811341
- pseudo_xgb: 0.915558

---

### Iteration 11 - Lean Features + 30-fold
**Date**: 2026-03-28T06:01:24.557037

**Key change**: Lean features (~80 vs 127), 30-fold CV

- xgb: 0.918090
- lgbm: 0.917948
- pairte: 0.894260
- XGB+LGBM: 0.918110
- Deotte 3-model: 0.916272

---

### Iteration 12 - Stacking + Conditional ORIG
**Date**: 2026-03-28T12:39:32.994702

- L0 xgb_bx: 0.918032
- L0 xgb_deep: 0.917901
- L0 lgbm_reg: 0.917936
- L0 lgbm_wide: 0.917729
- **L1 Ridge stack**: 0.918084
- **L1 LogReg stack**: 0.918053
- Simple avg: 0.918062

---
