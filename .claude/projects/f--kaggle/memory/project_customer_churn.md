---
name: project_customer_churn
description: Kaggle Playground S6E3 Customer Churn competition - project structure, framework details, deadline 2026-03-31
type: project
---

Active Kaggle competition: Predict Customer Churn (Playground Series S6E3).

**Why:** Binary classification competition evaluated by AUC-ROC. Deadline 2026-03-31. Top LB ~0.91762 (extremely tight top 20, only 0.00037 AUC spread).

**How to apply:**
- Competition workspace: `f:/kaggle/customer-churn/`
- Reusable framework: `f:/kaggle/kaggle-agent/` (Python package, installed editable)
- Claude agents: `f:/kaggle/.claude/agents/kaggle-*.md` (eda, feature-engineer, model-trainer, ensembler, orchestrator, researcher)
- Competition skill: `f:/kaggle/.claude/skills/competitions/customer-churn.md`
- Config: `f:/kaggle/customer-churn/config.yaml`
- Program: `f:/kaggle/customer-churn/program.md` (autoresearch-style autonomous loop instructions)
- Data: `f:/kaggle/customer-churn/data/` (train 594K rows, test 254K rows)
- Key features: Contract, PaymentMethod, InternetService, tenure are strongest churn predictors
- Strategy: massive ensembling (LightGBM + XGBoost + CatBoost), feature engineering, Optuna tuning
- Hardware: local RTX 3080 (10GB), RunPod A40 if needed
