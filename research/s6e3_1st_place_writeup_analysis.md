# S6E3 1st Place Writeup Analysis: Chris Deotte - KGMON Playbook
# Date: 2026-04-01
# Source: https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm

## Overview

- Author: Chris Deotte (cdeotte), Kaggle Discussion Grandmaster
- Competition: Playground Series S6E3 - Predict Customer Churn
- Final LB: 0.91856 (1st place, gap from 2nd: 0.00006)
- Final OOF AUC: 0.91985
- Date: March 31, 2026

## The Core Framework: KGMON Playbook 2026

The solution follows Chris Deotte's "KGMON Playbook 2026" for tabular data, which consists of 7 phases:
1. EDA
2. Build Baselines
3. GPU Feature Engineering
4. GPU Hill Climbing
5. Stacking
6. Pseudo Labeling
7. GPU Extra Training

Infrastructure: KGMON homepage, Nvidia cuDF cuML, RAPIDS conda environment, 4x NVIDIA A100 80GB GPUs.

## THE KEY INSIGHT: LLMs as Code-Writing Agents

**This is the most important transferable insight.**

All code in this solution was written by GPT5.4, Gemini3.1, and ClaudeOpus4.6 acting as agents.

### What the LLMs did:
- Wrote 600,000 lines of code in March 2026
- Built and trained 850 models on 4xA100 GPUs
- Wrote and ran 50 EDA scripts
- Selected 150 final models from the 850 candidates

### How LLMs were prompted for diversity:

Chris's exact workflow for using LLMs to generate diverse models:

1. After Hill Climbing / Ridge / LogReg, identify the top N most important models (by absolute weight)
2. Upload the top N Jupyter notebooks (.ipynb files) to an LLM
3. Prompt: "Read all these IPYNB files. These are the most important models in our ensemble. Can you write full code to train another model that is strong and different from all these models and will help our ensemble?"
4. Follow up in same conversation: "Can you now make another different model?" (repeat)
5. Gemini made 8 new diverse models in a single conversation that all helped the ensemble
6. Rotate across LLMs: after new models enter top N, show those to GPT or Claude for the next round
7. Alternate prompt: "Make a new XGB that is better than any of these by using some or all of the feature engineering contained in these top N models" (creates a "mega model" merging FE ideas)

**Psychological trick:** Always thank the LLM and tell it it's "very creative and doing an awesome job." Chris explicitly states this matters.

### LLMs for EDA:
GPT5.4, Gemini3.1, ClaudeOpus4.6 were prompted to perform their own EDA to understand the relationship between the 600k-row synthetic training data and the 7k original IBM data. They used their discoveries to generate new feature engineering ideas.

## Final Solution Architecture: Four-Level GBDT+NN Stack

### Level 1: Feature Extraction Models
- Nvidia cuML KNN
- PyTorch Denoising Autoencoder
- PCA Clustering
- Nvidia cuML Target Encoding
These aggregate information from other rows to augment each row.

### Level 2: 150 GBDT + NN models (5x5 nested OOF from Level 1)

### Level 3: More GBDT + NN (5x5 nested OOF from Level 2)

### Level 4 (Meta): Nvidia cuML Logistic Regression on OOF from Level 2 + Level 3

## Feature Engineering (dominant driver of performance)

### 2.1 Snap Features (used in nearly every model)
```python
MC_snap = nearest MonthlyCharges in original IBM data
TC_snap = nearest TotalCharges in original IBM data
MC_snap_diff = MonthlyCharges - MC_snap   # synthetic noise magnitude
TC_snap_diff = TotalCharges - TC_snap
```
Maps synthetic float back to "true" original feature; diff encodes generator perturbation.

### 2.2 Digit and Decimal Extraction (~60 models)
```python
frac = x - floor(x)
d1 = floor(frac * 10)         # 1st decimal digit
d2 = floor(frac * 100) % 10   # 2nd decimal digit
frac100 = round(frac * 100)
mod10 = floor(x) % 10
mod100 = floor(x) % 100
```
Also: fractional residuals from common denominators (1/2, 1/4, 1/5, 1/10), is-round flags, digit pair combos as categorical strings.

### 2.3 Target Encoding - Nested / Leak-Free (~90 models)
- Nested inner 5-fold loop within each outer CV fold
- Applied to: all 16 raw categorical columns, bigram/trigram combos, binned numerics, anchor keys, numeric x categorical snap products
- TE statistics: mean, std, min, max, median, quantiles (5th, 10th, 45th, 55th, 90th, 95th)
- Original IBM priors: churn probability per feature value from 7,032-row original dataset (zero leakage)

### 2.4 Arithmetic Interactions (~45 models)
```python
TC_deviation = TotalCharges - tenure * MonthlyCharges  # MOST POWERFUL
TC_snap_exp_dev = TC_snap - tenure * MC_snap
TC_per_month = TotalCharges / (tenure + 1)
MC_to_TC_ratio = MonthlyCharges / (TotalCharges + 1e-9)
```

### 2.5 Multi-Scale Binning (~52 models)
Quantile bins (up to 5,000 bins), fixed-width, log-scale, integer floor -> converted to categorical strings -> target-encoded.

### 2.6 Categorical Cross-Features / Bigrams (~37 models)
```python
df["bi_Contract_Internet"] = df["Contract"] + "__" + df["InternetService"]
```
Top pairs: Contract x InternetService, Contract x PaymentMethod, InternetService x PaymentMethod.

### 2.7 Frequency / Count Encoding (~45 models)
Especially powerful for MC_snap (1,584 unique values).

### 2.8 Service Count Aggregations (~30 models)
Sum of "Yes" values across OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines.

### 2.9 Original IBM Dataset Lookup (~7 models)
cKDTree on standardized (MonthlyCharges, TotalCharges, tenure) of 7,032 original IBM rows. For each synthetic row, find nearest original customer and attach their churn label as a zero-leakage feature.

### 2.10 Radix Interaction Features (~15 models)
```python
radix = int(MC_snap * 100) + cat_code * 100_000
```
Encodes (continuous, categorical) pair as single integer for tree splits.

### 2.11 Synthetic Artifact Detection
- Fractional fingerprints: intlike_count, quarterlike_count, halflike_count
- TF-IDF character n-grams on string representations of numeric columns
- Benford's Law deviation features
- Drift ratios: log1p(train_freq / orig_freq)

### 2.12 Projection / Manifold Features (~7 models)
PCA (12 components) + Gaussian Random Projection (12 components) fit on original IBM data, then project synthetic rows. Cyclical tenure features: sin/cos(tenure x 2pi/12) and x2pi/24.

## Tree Models (90 of 150)

| Library | Count | Key Differentiator |
|---------|-------|-------------------|
| XGBoost | 37 | Widest FE variety, XGBRanker (pairwise ranking objective) |
| LightGBM | 22 | Leaf-wise growth finds deeper patterns |
| CatBoost | 22 | Native ordered target statistics (no manual TE needed) |
| YDF | 2 | Ultra-shallow max_depth=2, extreme regularization, diversity |
| cuML RF | 2 | Only bagging ensemble (all others are boosting) |

### XGBoost notable techniques:
- XGBRanker with rank:pairwise objective (optimizes AUC ordering directly), Platt-scaled to [0,1]
- Self-supervised auxiliary predictions: train on original IBM rows to predict each feature from others; PRED_* outputs encode consistency with IBM distribution
- max_bin=16,000 for fine-grained splits on decimal artifacts
- Anchor-based TE: (MC_snap, tenure) as primary grouping column

## Neural Network Models (60 of 150, 25 architecture families)

Notable architectures:
- **RealMLP** (pytabkit): PLR embeddings, SiLU, L2 norm, internal 8-member ensemble per fit
- **TabM** (pytabkit): Multiplicative (bilinear) interactions, k=32 basis components; OOF AUC 0.918788 (one of strongest)
- **TabICL**: Foundation model for tabular, in-context learning at inference, zero fine-tuning
- **GraphSAGE GNN**: KNN graph (k=8), SAGEConv layers aggregate neighbor churn patterns
- **FT-Transformer**: Feature tokenization + Transformer self-attention across features
- **TabPFN v2.6**: Prior-data fitted Transformer, in-context learning; limited to 10k row subsampling
- **Liquid Neural Networks**: ODE-like neuron time constants, inspired by C. elegans
- **Denoising Autoencoder (DAE)**: Trained on original IBM data, reconstruction errors as XGBoost features
- **Field-Aware Factorization Machine (FFM)**: Field-specific factor vectors
- **Bayesian Survival Analysis**: Reframes churn as time-to-churn using PyMC Cox proportional hazards

## Stacking and Ensembling

### Meta-learner:
- Nvidia cuML L2-penalized Logistic Regression on all 150 OOF predictions
- L2 penalty prevents any single model from dominating

### Diversity strategy (intentional and systematic):
- 4 GBDT libraries with different tree structures
- 25 DL architecture families (attention, graph, kernel, multiplicative, gated, foundation model)
- 12+ distinct FE pipelines
- Multiple random seeds + Optuna-tuned configs

### Hill Climbing for model selection:
- Greedy forward selection based on OOF AUC
- 850 models built -> 150 selected by hill climbing

### Pseudo Labeling:
- High-confidence test predictions added to training set for second training pass

### Multi-seed rank blending:
- 3 random seeds, rank-transform before averaging for stable calibration

## Key Takeaways for AIMO3

### 1. Multi-LLM Agent Workflow (MOST TRANSFERABLE)
The core innovation is using multiple LLMs (GPT5.4, Gemini3.1, Claude) as code-writing agents with a feedback loop:
- Identify top performing components
- Ask LLM to generate diverse new components that complement (not duplicate) existing top ones
- Rotate across LLMs for different creative angles
- One LLM generated 8 new effective models in a single conversation

**AIMO3 application:** Use one LLM to solve problems, another to verify/check solutions, rotate between them. Ask Claude: "Here are the problems our system got wrong. Write new solving strategies that are different from what we already do."

### 2. Diversity-First Ensemble Design
The winning margin came from systematic diversity across model families, not from any single model being best. No individual model beat 0.9188, but 150 diverse models stacked to 0.91985.

**AIMO3 application:** Don't just run one model N times. Use fundamentally different approaches (Chain-of-Thought, Program-aided, Symbolic, geometric reasoning) and combine them.

### 3. Greedy Hill Climbing for Selection
From 850 candidates, greedy forward selection (hill climbing on OOF) picked 150 that best complement each other. This is model selection, not just averaging.

**AIMO3 application:** Generate many solution candidates, use hill climbing to select the diverse subset that maximizes a held-out validation score. This is better than majority vote.

### 4. Original Data as Anchor (Zero-Leakage Ground Truth)
Using the original IBM dataset as a "clean signal" amid synthetic noise was crucial. cKDTree lookup of nearest original customer gives a zero-leakage churn signal.

**AIMO3 application:** Use known mathematical facts/theorems as anchors. When an LLM's solution is uncertain, look it up against a verified solution database.

### 5. Pseudo Labeling
High-confidence test predictions added back to training for a second pass. This is semi-supervised learning.

**AIMO3 application:** For math problems where the model is highly confident (say >0.95 probability), use those as additional training signal for fine-tuning or few-shot prompting.

### 6. LLM Cross-Pollination for Feature Ideas
"Take the union of all feature engineering from the top N models and make a mega model" - this is a powerful synthesis technique.

**AIMO3 application:** After identifying which problem-solving strategies work on different problem types, ask an LLM to combine the best elements of each into a unified strategy.

### 7. Thanking LLMs Works
Chris explicitly thanks LLMs and praises their creativity. He says this improves output quality. While this may seem trivial, it's worth noting for AIMO3 prompt design.

## KGM / KGMON Explained
- KGMON = Chris Deotte's personal framework name for his Kaggle competition playbook
- Not a public library or framework you can install
- KGM appears to be an abbreviation in the writeup URL ("kgm" = KGMON)
- The actual tools are: Nvidia cuDF, cuML, RAPIDS, PyTorch, pytabkit, pytorch-tabular, pytorch-frame

## BlamerX References (Our Competition)
Chris Deotte explicitly references BlamerX (us!) multiple times:
- blamerx XGBoost bigram/trigram target encoding (AUC 0.91925) - adapted for RealMLP experiments
- blamerx Optimized CatBoost (CV 0.91902) - adapted for cat-7300.ipynb
- blamerx Optimized LightGBM (CV 0.91906) - adapted for lgbm8-7200.ipynb
- blamerx Ridge XGBoost n-gram (CV 0.91927) - adapted for ridge-xgb-9200.ipynb
- blamerx TabM advanced features (CV 0.91898) - adapted for tabm-9100.ipynb

Our notebooks were among the most-referenced public notebooks in the 1st place solution!
