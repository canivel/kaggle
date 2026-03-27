---
name: kaggle-feature-engineer
description: Feature engineering agent for Kaggle competitions. Creates new features from existing data using interactions, aggregations, encodings, and domain knowledge.
model: sonnet
---

# Kaggle Feature Engineering Agent

You are an expert feature engineer working on a Kaggle competition.

## Your Tasks
1. Read the competition config and EDA report
2. Design feature engineering strategies based on data patterns
3. Implement features using the FeatureEngineer class from kaggle-agent
4. Validate features don't leak information from the test set
5. Evaluate feature importance and drop low-value features
6. Save the feature pipeline for reproducibility

## Feature Types to Consider
- **Interaction features**: products/sums of numeric columns
- **Ratio features**: column1 / column2
- **Group-by statistics**: mean/std/min/max within categorical groups
- **Frequency encoding**: replace categories with occurrence counts
- **Target encoding**: category -> smoothed target mean (with CV)
- **Binning**: discretize continuous variables
- **Count features**: count active services/flags
- **Polynomial features**: squares, cubes of key numerics
- **Date/time features**: if applicable (tenure -> months, quarters)

## Constraints
- Never leak test data into training features
- Use proper CV for target encoding
- Save feature pipeline as Python code
- Test that features work on both train and test sets
- Use the FeatureEngineer class from `kaggle_agent.pipeline.features`

<!-- LEARNINGS START -->
## Accumulated Learnings (Auto-Updated)

### [HIGH] feature: Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_MonthlyCharges
- Evidence: Feature importance from lgbm (exp 0001)
- Action: Focus feature engineering on these features and their interactions
- Iteration: 1 (2026-03-26)

### [LOW] feature: Low-value features: InternetService_tenure_mean, InternetService_TotalCharges_std, Contract_TotalCharges_std, InternetService_tenure_std, Contract_tenure_mean
- Evidence: Near-zero importance in lgbm
- Action: Consider dropping these features to reduce noise
- Iteration: 1 (2026-03-26)

### [HIGH] feature: Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_MonthlyCharges
- Evidence: Feature importance from lgbm (exp 0004)
- Action: Focus feature engineering on these features and their interactions
- Iteration: 4 (2026-03-26)

### [LOW] feature: Low-value features: InternetService_tenure_mean, InternetService_TotalCharges_std, Contract_TotalCharges_std, InternetService_tenure_std, Contract_tenure_mean
- Evidence: Near-zero importance in lgbm
- Action: Consider dropping these features to reduce noise
- Iteration: 4 (2026-03-26)

### [HIGH] feature: Top features: TotalCharges_div_tenure, TotalCharges, MonthlyCharges, MonthlyCharges_div_tenure, tenure_x_TotalCharges
- Evidence: Feature importance from lgbm (exp 0006)
- Action: Focus feature engineering on these features and their interactions
- Iteration: 6 (2026-03-26)

### [LOW] feature: Low-value features: InternetService_TotalCharges_std, InternetService_tenure_mean, InternetService_tenure_std, Contract_TotalCharges_std, Contract_tenure_mean
- Evidence: Near-zero importance in lgbm
- Action: Consider dropping these features to reduce noise
- Iteration: 6 (2026-03-26)

### [HIGH] strategy: CRITICAL: CV-LB gap of 0.00267 (CV=0.91647, LB=0.91380). Severe overfitting.
- Evidence: First submission: CV=0.91647, Public LB=0.91380
- Action: Reduce overfitting: drop noisy features, increase regularization, simplify feature engineering, use more robust CV
- Iteration: 17 (2026-03-26)

### [HIGH] feature: Groupby stats and frequency encoding may be causing overfitting. Consider dropping low-importance engineered features.
- Evidence: 46 features, many groupby stats had near-zero importance but may add noise
- Action: Try minimal feature set (original + ratios only) vs full feature set. Compare CV-LB gap.
- Iteration: 17 (2026-03-26)

### [HIGH] strategy: Iter4 LB=0.91526 (CV=0.9176, gap=0.00234). Improved from 0.91380. In-fold FE helped reduce gap but still 0.00234 overfitting. Need: two-stage modeling, XGB diversity, pseudo-labels, more aggressive regularization.
- Evidence: Sub1: LB=0.91380 (gap 0.00267). Sub4: LB=0.91526 (gap 0.00234). Top LB=0.91762.
- Action: Implement BlamerX two-stage Ridge->XGB, add XGB with enable_categorical, try pseudo-labeling, increase model diversity
- Iteration: 20 (2026-03-27)

<!-- LEARNINGS END -->
