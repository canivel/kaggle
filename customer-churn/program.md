# Customer Churn Competition - Autonomous Research Program

## Goal
Maximize AUC-ROC on the Predict Customer Churn competition (Playground Series S6E3).
Current top LB: ~0.91762. Target: top 10.

## Setup
1. Ensure data is at `data/train.csv`, `data/test.csv`
2. Ensure kaggle-agent framework is installed: `uv pip install -e ../kaggle-agent`
3. Create `experiments/` and `checkpoints/` directories
4. Initialize `experiments/results.tsv` with header
5. Confirm GPU is available (RTX 3080)

## Experiment Loop

### NEVER STOP. The human might be asleep. You are autonomous.

```
LOOP FOREVER:
  1. Read experiments/results.tsv to see current best score
  2. Choose next experiment:
     a. If < 3 experiments: run baselines (lgbm, xgb, catboost)
     b. If < 10 experiments: run strategy variants from library
     c. If < 20 experiments: run Optuna tuning for each model type
     d. If >= 20 experiments: build ensemble, try new features, iterate
  3. Run the experiment:
     - Train with 5-fold StratifiedKFold CV
     - Record: cv_score, cv_std, model_type, params
  4. Compare to best:
     - If improved: status = "kept", save checkpoint
     - If not improved: status = "discarded"
  5. Log to experiments/results.tsv
  6. If ensemble score > best single model by > 0.001:
     - Generate submission
     - status = "submitted"
  7. Continue to next experiment
```

### Timeout
- Each experiment: max 10 minutes
- If a run crashes: log "crashed", move on
- If stuck for 3+ experiments with no improvement: try radical changes

### What to Try (in order)
1. Baseline models (lgbm, xgb, catboost) with defaults
2. Feature engineering (interactions, ratios, groupby stats)
3. Hyperparameter tuning (Optuna, 50 trials each)
4. More feature engineering (target encoding, frequency encoding)
5. Model diversity (different seeds, different architectures)
6. Stacked ensemble (logistic regression meta-learner)
7. Weighted ensemble (optimize weights on OOF)
8. Original data blending (IBM Telco dataset)
9. Post-processing (calibration, rank averaging)
10. Multi-seed averaging (train same model with 5 different seeds)

### Results Format
Tab-separated file: `experiments/results.tsv`
```
experiment_id	timestamp	model_type	description	cv_score	cv_std	lb_score	status	duration_seconds	n_features	params	notes
0001	2026-03-26T...	lgbm	baseline	0.8432	0.0012	N/A	kept	45.2	19	{}	first run
```

### Constraints
- Max 5 Kaggle submissions per day
- Use GPU for XGBoost and CatBoost
- Never modify the evaluation metric or CV strategy mid-run
- Always use seed=42 for reproducibility (except diversity experiments)
- NEVER use litellm
