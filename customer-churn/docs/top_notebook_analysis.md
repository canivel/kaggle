# Top Notebook Analysis - Playground Series S6E3 (Customer Churn)

Competition: https://www.kaggle.com/competitions/playground-series-s6e3
Deadline: 2026-03-31 | Teams: 3530 | Metric: AUC-ROC

---

## 1. CV AUC 0.91930 - XGB+CB Blend (Artem, 93 votes) - HIGHEST CV

**Source**: `artemevstafyev/cv-auc-0-91930-xgb-cb-blend`

**What it actually does (LB notebook)**:
- Takes its own best submission (from a private `xgb-cb-best-cv-auc-0-91930-blend` notebook)
- Blends with AnthonyTherrien's blend submission
- Uses **rank-based calibration**: `rank_new = rankdata(pred2) * 0.99 + adjustment * 0.01`
- Applies `calibrate_rank_by_pred()` to map blended ranks back to probability space using pred1's distribution
- This is a **post-processing rank blending** technique, not a training notebook

**Key Technique - Rank Calibration**:
```python
rank_new = rankdata(pred2) * 0.99 + adjustment * 0.01
# Then calibrate: group by rank, assign mean pred from pred1, monotonically increasing
```

**Scores**: CV 0.91930 (from underlying model), LB likely higher via blending

---

## 2. Ridge -> XGB + N-gram (BlamerX, 101 votes) - CV 0.91927

**Source**: `blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv`

**Models**: Two-stage pipeline: Ridge (Stage 1) -> XGBoost (Stage 2)

**Feature Engineering** (VERY comprehensive):
1. **Frequency encoding** of numerical features (tenure, MonthlyCharges, TotalCharges)
2. **Arithmetic interactions**: charges_deviation, monthly_to_total_ratio, avg_monthly_charges
3. **Service counts**: service_count, has_internet, has_phone
4. **ORIG_proba features**: Target probability lookup from original IBM Telco dataset for ALL 19 columns
5. **Distribution features**: pctrank against churner/non-churner TotalCharges, z-score gaps, conditional percentile ranks by InternetService and Contract
6. **Quantile distance features**: Distance to q25/q50/q75 of churner vs non-churner TotalCharges
7. **Digit features** (35 features!): First/last/second digits of tenure/MonthlyCharges/TotalCharges, mod10/mod12/mod100, num_digits, is_multiple_10, fractional parts, deviation from round numbers
8. **Numericals as categories**: tenure/MonthlyCharges/TotalCharges cast to category
9. **Bi-gram composites**: All C(6,2)=15 pairs from top 6 categoricals (Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling)
10. **Tri-gram composites**: C(4,3)=4 triples from top 4 categoricals

**Target Encoding** (leak-free):
- **Outer**: 20-fold StratifiedKFold
- **Inner**: 5-fold for leak-free TE aggregation (std, min, max)
- sklearn TargetEncoder for mean (auto smooth, binary target)
- Separate TE for N-gram categoricals (mean only)

**Stage 1 - Ridge**:
- All numerical + TE features (standardized) + OHE categoricals
- Ridge(alpha=10.0)
- Ridge predictions clipped to [0,1]

**Stage 2 - XGBoost**:
- Ridge predictions added as feature `ridge_pred`
- Raw categoricals dropped (replaced by TE features)

**XGB Hyperparameters** (Optuna-optimized):
```python
n_estimators=50000, learning_rate=0.0063, max_depth=5,
subsample=0.81, colsample_bytree=0.32, min_child_weight=6,
reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790,
early_stopping_rounds=500, device='cuda', enable_categorical=True
```

**Key Insight**: colsample_bytree=0.32 is very aggressive (only 32% of features per tree), combined with very low learning_rate=0.0063. This makes XGBoost very regularized.

**CV Score**: 0.91927

---

## 3. Chris Deotte - 3xGPU Models (96 votes) - CV 0.9178

**Source**: `cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178`

**3 Diverse Models** (equal 1/3 weight blend):

### Model A: cuML Logistic Regression with Pair Target Encoding (CV 0.9160)
- Creates ALL C(n,2) feature pairs from the original 19 features
- Applies cuML TargetEncoder to each pair (5-fold inner CV, smooth=0)
- Converts TE outputs to logit space: z, z^2, z^3 (logit3 features)
- StandardScaler -> cuML LogisticRegression(C=0.5, L2, max_iter=4000)
- 5-fold outer CV

### Model B: XGBoost (CV 0.9166)
- Basic XGBoost with NO feature engineering
- Just enable_categorical=True on raw features
- Parameters:
```python
n_estimators=100000, learning_rate=0.1, max_depth=3,
min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
early_stopping_rounds=200, device='cuda'
```

### Model C: PyTorch Embedding MLP (CV 0.9164)
- Categorical embeddings (dim = 1.8 * card^0.25, clipped 4-64)
- Numeric: raw + categorical proxies (snap rare values to nearest frequent)
- Architecture: Embedding -> [512, 256] with LayerNorm + ReLU + Dropout(0.30)
- SmoothBCE loss (eps=0.02)
- AdamW(lr=2.5e-5, weight_decay=3e-4)
- Cosine annealing with 1 epoch warmup
- 10 epochs, batch_size=256, patience=10

### Ensemble: Simple average of 3 models -> CV 0.9178
**Key Insight**: Diversity beats individual strength. Three very different models (linear, tree, neural) each ~0.916 combine to 0.9178.

---

## 4. EDA + Baseline XGB (Sagar Nagpure, 129 votes) - CV 0.91808

**Source**: `datasciencegrad/s6e3-detail-eda-baseline-xgb-auc-0-91808`

**Models**: Single XGBoost

**Feature Engineering**:
1. Frequency encoding of numericals
2. Arithmetic interactions (charges_deviation, monthly_to_total_ratio, avg_monthly_charges)
3. Service counts (service_count, has_internet, has_phone)
4. ORIG_proba features from original dataset (19 features)
5. Numericals as categories

**Target Encoding**:
- 5-fold outer, 5-fold inner (leak-free)
- TE stats: std, min, max + sklearn TargetEncoder for mean
- Pseudo labels: threshold=0.995, retrain if AUC improves

**XGB Parameters**:
```python
n_estimators=50000, learning_rate=0.05, max_depth=6,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
reg_alpha=0.1, reg_lambda=1.0, gamma=0.05,
early_stopping_rounds=500, device='cuda', enable_categorical=True
```

**Pseudo-labeling**: Test samples with pred > 0.995 or pred < 0.005 added to training. Only kept if AUC improves on validation fold.

**CV Score**: 0.91808

---

## 5. GNN Starter with Hill Climbing (Chris Deotte, 88 votes) - CV 0.9155

**Source**: `cdeotte/gnn-starter-cv-0-9155-with-hill-climbing-demo`

**Model**: GraphSAGE (2-layer)

**Graph Construction**:
- KNN graph (K=8 neighbors) using cuML NearestNeighbors
- Features for graph: OHE of 16 categorical columns + StandardScaled 3 numerics * 3.0 multiplier

**Node Features** (25 total):
- 16 original categoricals (as embeddings)
- 3 numeric->categorical proxies (snap rare values to nearest frequent, min_count=25)
- 3 numeric rare-flag categoricals
- 3 raw numeric columns

**Architecture**: CatEmbed + 2x SAGEConv(hidden=128) with residual connections (0.5 weight)
- Dropout 0.20, LayerNorm, SmoothBCE(eps=0.01)
- Mini-batch training with FANOUTS=[6,4]
- BATCH_SIZE=8192, 5 epochs

**Key Insight**: GNN captures customer similarity graph structure. Complementary to tree/linear models.

**CV Score**: 0.9155

---

## 6. RealMLP PyTabKit (Vladimir Demidov, 66 votes)

**Source**: `yekenot/ps-s6-e3-realmlp-pytabkit`

**Model**: RealMLP_TD_Classifier from pytabkit

**Feature Engineering**:
- Arithmetic ratios: MonthlyCharges/TotalCharges, TotalCharges/tenure, Monthly_to_avg_ratio
- tenure^2, is_loyal_customer (tenure >= 24)
- Digit extraction: TotalCharges_d-3 (3rd decimal digit)
- TotalCharges mod100, mod1000, is_multiple_10
- KBinsDiscretizer: TotalCharges [4000, 500 bins], MonthlyCharges [200, 100 bins]
- Categorize all numericals
- Triple interaction: Contract_InternetService_PaymentMethod
- Target encoding on combo features

**RealMLP Parameters**:
```python
n_ens=8, n_epochs=3, batch_size=256, lr=0.075, wd=0.0236,
sq_mom=0.988, lr_sched='flat_anneal', first_layer_lr_factor=0.25,
embedding_size=6, max_one_hot_cat_size=18,
hidden_sizes=[512, 256, 128], act='silu', p_drop=0.05,
plr_hidden_1=16, plr_hidden_2=8, plr_sigma=2.33,
ls_eps=0.01, ls_eps_sched='sqrt_cos',
tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize']
```

**CV**: 20-fold StratifiedKFold

---

## 7. AnthonyTherrien Blend (68 votes)

**Source**: `anthonytherrien/predict-customer-churn-blend`

Simple weighted average blend:
- submission.csv (weight 2.7) + submission (1).csv (weight 0.1)
- Both from a private "vault" dataset
- Likely blending multiple model outputs

---

## 8. XGB+LGB Multi-Seed Ensemble (65 votes) - CV 0.91680

**Source**: `badalkrsharma/cv-0-91680-xgb-lgb-multi-seed-ensemble`

**Models**: XGBoost + LightGBM with multi-seed averaging

**Config** (from commented code):
- 10-fold CV, 10-fold inner TE, 3 seeds [11, 42, 99]
- Pseudo labels with threshold 0.998

**XGB Parameters**:
```python
n_estimators=50000, learning_rate=0.03, max_depth=6,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
reg_alpha=0.1, reg_lambda=1.0, gamma=0.05,
early_stopping_rounds=500, device='cuda'
```

**LGB Parameters**:
```python
n_estimators=50000, learning_rate=0.03, max_depth=7,
num_leaves=63, subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20
```

**Feature Engineering** (extensive, from commented code):
- Frequency encoding, rank encoding, log/sqrt/inverse transforms
- Arithmetic interactions (7 features)
- Service counts (6 features)
- Tenure bins, MonthlyCharges bins (40 quantile), TotalCharges bins (60 quantile)
- ORIG_proba from original dataset
- Cross pairs: (Contract, InternetService), (PaymentMethod, Contract), etc.
- Triple: Contract__InternetService__PaymentMethod
- Category counts and rare flags
- ISYES/ISNO/ISOTHER binary features for Yes/No columns

---

## 9. XGB+LGBM+CatBoost Ensemble (Rohan Rathod, 64 votes)

**Source**: `rohanrathod02/customer-churn-xgb-lgbm-catboost-ensemble`

**Models**: XGBoost + LightGBM + CatBoost + RealMLP (all via unified training function)

**Feature Engineering** (10 steps):
1. Frequency encoding (all columns)
2. Out-of-fold target encoding (all columns)
3. RobustScaler (numericals)
4. KBinsDiscretizer (10 bins, uniform)
5. OrdinalEncoder (categoricals)
6. All C(15,2)=105 categorical pair combinations
7. Interaction: tenure*MonthlyCharges, SeniorCitizen*MonthlyCharges
8. Polynomial features (degree=2): tenure, MonthlyCharges, TotalCharges
9. Ratio: TotalCharges/tenure
10. Aggregate features: mean/median/std of numericals grouped by Contract/InternetService/PaymentMethod

**CatBoost Parameters**:
```python
iterations=10000, learning_rate=0.02, depth=2, min_data_in_leaf=1,
auto_class_weights='Balanced', bootstrap_type='Bernoulli', subsample=0.9,
early_stopping_rounds=300
```

**LightGBM Parameters**:
```python
n_estimators=16000, learning_rate=0.02, num_leaves=20, max_depth=4,
min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
reg_alpha=0.1, reg_lambda=0.1, early_stopping=300
```

**RealMLP Parameters**:
```python
n_epochs=1, batch_size=256, n_ens=8, act='mish', embedding_size=8,
hidden_width=384, n_hidden_layers=4, lr=0.04, p_drop=0.073
```

---

## 10. Yusuf Murtaza Blending Notebook (top requested)

**Source**: `yusufmurtaza01/s6e3-blending`

**Technique**: Advanced weighted blending with positional weighting

**Inputs**:
- Artem's CV 0.91930 submission (base, weight 0.95)
- Own blend variants (weight 0.05 each, cascaded)
- AnthonyTherrien's blend

**Blending Method**:
1. First cascaded mixing: `df = df*0.95 + 0.05*a; df = df*0.95 + 0.05*b; ...`
2. Then positional sorting + weighting:
   - type_sort='asc/desc' with asc_weight=0.30, desc_weight=0.70
   - subwts (positional sub-weights): [+0.21, -0.03, -0.07, -0.11]
   - main weights: [0.40, 0.30, 0.20, 0.10]
3. For each row, sorts submissions by predicted probability, assigns positional weights

**Warning from author**: "Blending top public notebook outputs can cause significant drop on private LB"

---

## SUMMARY: KEY ACTIONABLE TECHNIQUES

### Models that work best:
1. **XGBoost** (GPU, enable_categorical) - most consistent performer
2. **LightGBM** - complements XGBoost well
3. **CatBoost** - especially with depth=2 and balanced class weights
4. **RealMLP** (pytabkit) - strong neural baseline
5. **GNN** (GraphSAGE) - unique diversity source
6. **Ridge/Logistic Regression** - as Stage 1 or diversity component

### Essential Feature Engineering:
1. **ORIG_proba** - Target probability lookup from original IBM Telco dataset (leak-free, huge gain)
2. **Target encoding** - Inner-fold leak-free TE with std/min/max stats + mean
3. **Arithmetic interactions** - charges_deviation, monthly_to_total_ratio, avg_monthly_charges
4. **Service counts** - sum of "Yes" services, has_internet, has_phone
5. **Frequency encoding** - value_counts(normalize=True) for numericals
6. **Digit features** - first/last digits, mod10/mod100, fractional parts, deviation from round numbers
7. **Distribution features** - percentile rank vs churner/non-churner distributions, z-score gaps
8. **N-gram categoricals** - bi-gram and tri-gram from top categoricals (Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling)
9. **Numericals as categories** - casting tenure/MonthlyCharges/TotalCharges to category type
10. **KBinsDiscretizer** - 4000 bins for TotalCharges, 200 bins for MonthlyCharges

### Best CV Strategies:
- 20-fold outer StratifiedKFold (BlamerX, Yekenot)
- 5-fold inner for leak-free target encoding
- Multi-seed ensembling (seeds [11, 42, 99])
- Pseudo-labeling with threshold 0.995-0.998

### Ensembling Techniques:
1. **Simple average** of diverse models (Chris Deotte: 3 models -> +0.0018 AUC gain)
2. **Weighted average** blending
3. **Rank-based calibration** (Artem: blend ranks then calibrate back to probabilities)
4. **Cascaded blending** (Yusuf: iterative 95/5 mixing)
5. **Two-stage stacking** (BlamerX: Ridge predictions as XGBoost feature)

### Special Techniques:
- **Pseudo-labeling**: Test predictions > 0.995 confidence added to training
- **Pair target encoding**: All C(n,2) feature pairs through TargetEncoder (Chris Deotte)
- **Logit3 features**: logit(x), logit(x)^2, logit(x)^3 for calibrated probabilities
- **Numeric snapping**: Rare numerical values snapped to nearest frequent value
- **SmoothBCE**: Label smoothing with eps=0.01-0.02 for neural models
- **Original dataset usage**: IBM Telco dataset for target probability lookups (not added to training rows)

### Hyperparameter Ranges that Work:
| Parameter | XGBoost | LightGBM | CatBoost |
|-----------|---------|----------|----------|
| n_estimators | 50000-100000 | 16000-50000 | 10000 |
| learning_rate | 0.003-0.1 | 0.02-0.03 | 0.02 |
| max_depth | 3-6 | 4-7 | 2 |
| subsample | 0.8-0.85 | 0.7-0.8 | 0.9 |
| colsample_bytree | 0.32-0.85 | 0.7-0.8 | - |
| early_stopping | 200-500 | 300 | 300 |
| reg_alpha | 0.1-3.5 | 0.1 | - |
| reg_lambda | 1.0-1.3 | 0.1-1.0 | - |
