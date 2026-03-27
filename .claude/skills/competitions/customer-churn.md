---
name: customer-churn
description: Competition skill for Kaggle Playground Series S6E3 - Predict Customer Churn. Contains all competition knowledge, data schema, winning strategies, and submission requirements.
---

# Predict Customer Churn - Competition Skill
## Kaggle Playground Series - Season 6, Episode 3

### Quick Facts
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3
- **Type**: Binary Classification
- **Metric**: AUC-ROC
- **Deadline**: March 31, 2026 23:59 UTC
- **Submissions/day**: 5
- **Team size**: Max 3
- **External data**: Allowed (original IBM Telco dataset)
- **Top LB score**: ~0.91762 (Chris Deotte, as of 2026-03-26)

### Target
- Column: `Churn`
- Train values: "Yes" / "No" (encode to 1/0)
- Submission: float probability [0.0, 1.0]
- Distribution: 77.5% No, 22.5% Yes (imbalanced ~1:3.4)

### Dataset Schema (21 columns)
| Column | Type | Values | Churn Signal |
|--------|------|--------|-------------|
| id | int | unique identifier | - |
| gender | str | Female, Male | weak |
| SeniorCitizen | int | 0, 1 | strong (50% vs 19%) |
| Partner | str | Yes, No | moderate |
| Dependents | str | Yes, No | moderate |
| tenure | int | 1-72 months | **strongest** (r=-0.418) |
| PhoneService | str | Yes, No | weak |
| MultipleLines | str | Yes, No, No phone service | weak |
| InternetService | str | DSL, Fiber optic, No | **strong** (Fiber=41.5%) |
| OnlineSecurity | str | Yes, No, No internet service | strong |
| OnlineBackup | str | Yes, No, No internet service | moderate |
| DeviceProtection | str | Yes, No, No internet service | moderate |
| TechSupport | str | Yes, No, No internet service | strong |
| StreamingTV | str | Yes, No, No internet service | weak |
| StreamingMovies | str | Yes, No, No internet service | weak |
| Contract | str | Month-to-month, One year, Two year | **strongest** (M2M=42.1%) |
| PaperlessBilling | str | Yes, No | strong (31.9% vs 7.5%) |
| PaymentMethod | str | 4 methods | **strong** (E-check=48.9%) |
| MonthlyCharges | float | 18.25-118.75 | moderate (r=+0.273) |
| TotalCharges | float | 18.80-8684.80 | moderate (r=-0.218) |
| Churn | str | Yes, No | **TARGET** |

### Dataset Sizes
- Train: 594,194 rows x 21 columns (80.5 MB)
- Test: 254,655 rows x 20 columns (33.8 MB)
- Original IBM Telco: 7,043 rows (can blend for advantage)
- Zero null values

### Submission Format
```csv
id,Churn
594194,0.5
594195,0.3
...
```
- 254,655 rows
- Probabilities (float), NOT binary 0/1

### Winning Strategies (from top solutions & discussions)

#### Models That Win
1. **LightGBM** - Primary workhorse
2. **XGBoost** (GPU) - Strong diversity for ensemble
3. **CatBoost** (GPU) - Handles categoricals natively
4. **Massive ensembling** - Top solutions blend 30-70+ models
5. Neural nets (TabNet, simple FF) for diversity

#### Critical Feature Engineering
1. `TotalCharges / tenure` → average monthly spend
2. `MonthlyCharges / tenure` → charge rate
3. Service count (sum of all "Yes" service flags)
4. Contract × InternetService interaction
5. PaymentMethod × Contract interaction
6. tenure bins (0-12, 13-24, 25-48, 49-72)
7. SeniorCitizen × tenure interaction
8. Group-by stats: mean charges per Contract type, per InternetService
9. Frequency encoding for all categoricals
10. "No internet service" / "No phone service" → binary flags

#### Key Insights
- Leaderboard is EXTREMELY tight (0.00037 AUC separates 1st-20th)
- Every decimal matters - ensembling is critical
- Contract type + PaymentMethod + InternetService are the 3 strongest features
- Fiber optic + Month-to-month + Electronic check = highest churn combo
- Long tenure (>48 months) = very low churn regardless of other factors
- Original IBM data blending provides small but meaningful boost

### Project Layout
```
f:/kaggle/customer-churn/
├── config.yaml              # Competition config
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── run_baseline.py          # Main pipeline
├── experiments/
│   └── results.tsv          # Experiment log
├── checkpoints/             # Model checkpoints
└── submissions/             # Generated submissions
```

### Framework
Uses `kaggle-agent` framework at `f:/kaggle/kaggle-agent/`
- Config: `kaggle_agent.config`
- Data: `kaggle_agent.pipeline.data`
- Features: `kaggle_agent.pipeline.features`
- Models: `kaggle_agent.pipeline.models`
- Tuning: `kaggle_agent.pipeline.tuning`
- Ensemble: `kaggle_agent.ensemble.stacking`
- Tracking: `kaggle_agent.tracking.experiments`
- Submission: `kaggle_agent.pipeline.submission`

<!-- COMPETITION LEARNINGS START -->
### Discovered Insights (Auto-Updated)

#### Ensemble
- Logistic regression meta-learner on OOF may overfit. Try simpler averaging or rank averaging. (impact: medium, evidence: Stacking with logistic meta-learner: CV=0.91647 but LB=0.91380)

#### Feature
- Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_MonthlyCharges (impact: high, evidence: Feature importance from lgbm (exp 0001))
- Low-value features: InternetService_tenure_mean, InternetService_TotalCharges_std, Contract_TotalCharges_std, InternetService_tenure_std, Contract_tenure_mean (impact: low, evidence: Near-zero importance in lgbm)
- Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_MonthlyCharges (impact: high, evidence: Feature importance from lgbm (exp 0004))
- Low-value features: InternetService_tenure_mean, InternetService_TotalCharges_std, Contract_TotalCharges_std, InternetService_tenure_std, Contract_tenure_mean (impact: low, evidence: Near-zero importance in lgbm)
- Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_TotalCharges (impact: high, evidence: Feature importance from lgbm (exp 0006))
- Low-value features: InternetService_TotalCharges_std, InternetService_tenure_mean, InternetService_tenure_std, Contract_TotalCharges_std, Contract_tenure_mean (impact: low, evidence: Near-zero importance in lgbm)
- Groupby stats and frequency encoding may be causing overfitting. Consider dropping low-importance engineered features. (impact: high, evidence: 46 features, many groupby stats had near-zero importance but may add noise)

#### Model
- lgbm: more leaves improves capture of complex patterns (impact: high, evidence: CV improved to 0.916217)
- lgbm: low learning rate helps; more iterations with patience helps; stronger regularization reduces overfitting (impact: high, evidence: CV improved to 0.916312)

#### Strategy
- CRASH: XGBoost baseline with default params - XGBClassifier.fit() got an unexpected keyword argument 'callbacks' (impact: low, evidence: Experiment 0002 crashed)
- CRASH: XGBoost deeper trees - XGBClassifier.fit() got an unexpected keyword argument 'callbacks' (impact: low, evidence: Experiment 0009 crashed)
- CRASH: XGBoost low LR high reg - XGBClassifier.fit() got an unexpected keyword argument 'callbacks' (impact: low, evidence: Experiment 0010 crashed)
- CRASH: XGBoost wide shallow - XGBClassifier.fit() got an unexpected keyword argument 'callbacks' (impact: low, evidence: Experiment 0011 crashed)
- CRASH: XGBoost different seed - XGBClassifier.fit() got an unexpected keyword argument 'callbacks' (impact: low, evidence: Experiment 0015 crashed)
- CRITICAL: CV-LB gap of 0.00267 (CV=0.91647, LB=0.91380). Severe overfitting. (impact: high, evidence: First submission: CV=0.91647, Public LB=0.91380)
- Iter4 LB=0.91526 (CV=0.9176, gap=0.00234). Improved from 0.91380. In-fold FE helped reduce gap but still 0.00234 overfitting. Need: two-stage modeling, XGB diversity, pseudo-labels, more aggressive regularization. (impact: high, evidence: Sub1: LB=0.91380 (gap 0.00267). Sub4: LB=0.91526 (gap 0.00234). Top LB=0.91762.)

<!-- COMPETITION LEARNINGS END -->
