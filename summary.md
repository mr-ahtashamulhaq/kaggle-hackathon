# Kaggle PSS6E7 - Experimentation Summary

## Competition Overview

- **Goal:** Predict student health condition (fit / at-risk / unhealthy) from behavioral and physical data
- **Metric:** Balanced Accuracy (mean of per-class recalls)
- **Key Challenge:** 86% class imbalance toward "fit", synthetic MNAR missing values, train/test distribution shift in `water_intake`, `calorie_expenditure`, `bmi`

---

## Version History and Results

| Version | Key Change | OOF BA | Public LB |
|---|---|---|---|
| V1 | LightGBM baseline, 5-fold, wrong data paths | ~0.947 | 0.94667 |
| V2 | Fixed paths, feature engineering ratios | ~0.948 | 0.94769 |
| V3 | LightGBM + CatBoost blend, 50/50 | ~0.949 | 0.94944 |
| V4 | Prior-Correction instead of class_weight | ~0.949 | 0.94970 |
| V5 | Added cross-fitted binned target encoding | 0.9497 | 0.94987 |
| V6 | Added XGBoost to 3-way blend + Nelder-Mead optimizer | 0.9498 | 0.94983 (dropped) |
| V7 | HGBC + sklearn exact-value TargetEncoder, 7 folds | 0.9497 | 0.95017 |
| V8/V9 | Added seed averaging (3 seeds), removed early-stop validation waste | 0.9496 | 0.95032 |
| V10 | Added explicit NaN indicator features | 0.9498 | 0.95029 |

**Current standing: Rank ~320, Score 0.95032. Top LB: 0.95224.**

---

## What Definitively Worked

### 1. Prior-Correction (V4)
**Change:** Removed `class_weight='balanced'` from all models. Instead, after training, divide raw predicted probabilities by the class priors, then renormalize.

**Why it worked:** `class_weight='balanced'` distorts the tree-building geometry -- trees learn splits on a reweighted dataset, which can be suboptimal. Post-hoc prior correction adjusts the decision boundary without touching the training process.

**Gain:** +0.00026 OOF, consistent across all subsequent versions.

### 2. HGBC + Exact-Value Target Encoding (V7)
**Change:** Replaced the LightGBM/CatBoost/XGBoost ensemble entirely with a single `HistGradientBoostingClassifier` and replaced binned target encoding with sklearn's `TargetEncoder` (exact numeric values, multiclass, smooth='auto', inner cv=5).

**Why it worked:**
- HGBC is the 4th distinct tree-boosting library (alongside LGBM/CB/XGB). It has a different histogram strategy and splitting mechanics, providing genuine architectural diversity where LGBM/CB/XGB all plateau at ~0.949.
- Exact-value TargetEncoder creates a richer, smoother mapping from numeric feature value to target probability than percentile bins. Empirical Bayes shrinkage (smooth='auto') handles rare values robustly.
- Proven by Mark Susol's research trail (v0.7): +0.0009 OOF vs CatBoost baseline, no CV-to-LB haircut, 0.94937 to 0.95036 LB.

**Gain:** This was the single biggest jump, taking us from 0.94987 to 0.95017.

### 3. 7-Fold CV
**Why:** On 690k training rows, 7-fold gives each validation fold ~98k rows. This makes the OOF estimate far more stable. The community researcher with the most rigorous methodology also used 7-fold and showed CV tracks LB with mean gap of only -0.0001 across 13 models.

---

## What Did Not Work

### Blending LightGBM + CatBoost + XGBoost
**Why it failed:** LGBM, CB, XGB agree on ~99% of rows on this dataset. Blending them adds no real diversity. When we optimize blend weights (Nelder-Mead), we overfit to the train distribution's error-correlation structure. Because of the adversarial train/test shift (ROC-AUC ~0.65 for water_intake/calorie_expenditure/bmi), this structure shifts in test, and the meta-weights become wrong.

**Evidence:** Every blend attempt (V3, V5, V6) produced flat or negative improvements. Mark Susol's v0.5 (4-way blend, nested CV) showed -0.0002 nested OOF improvement.

### Numerical Ratio Features
Removed in V4. Features like `calorie_expenditure / step_count` directly encode the distribution shift in `calorie_expenditure` and `bmi`. Models trained on these ratios fail to generalize to the test set's different distribution.

### XGBoost as 3rd Ensemble Member (V6)
Adding XGBoost alongside HGBC dropped the LB from 0.95032 to 0.94983. XGBoost and our LGBM are too correlated; the 3-way blend added noise, not diversity.

### Binned Target Encoding (V5)
Creating 15 percentile bins and encoding mean target per bin is noisier than sklearn's exact-value TargetEncoder with Empirical Bayes shrinkage. The binned approach also fixed bin boundaries on the full dataset before folding, which caused subtle data leakage.

### NaN Indicator Features (V10)
Added `{col}_is_nan` binary features for all 7 numerics. Flat result (0.95029 vs 0.95032). HGBC already learns the MNAR direction natively -- the explicit indicator provides no additional signal that the implicit split direction doesn't already capture. The web search also confirmed: "several top competitors reported no significant improvement when adding is-missing flags."

### Seed Averaging (V8/V9)
Very marginal benefit. The fold variance (0.9475-0.9525) persisted even with 3-seed averaging. The variance is driven by fold-level data distribution differences (hard folds 4 and 7 consistently score ~0.9478), not by model randomness.

---

## The Ceiling Problem

The honest research community ceiling for this dataset is approximately **OOF 0.9503-0.9506** for tree-based models. The only approaches documented above this with honest nested CV are:

1. **RealMLP (Mark Susol v0.8):** PyTorch neural net with periodic numeric embeddings, NTK-parametrized layers, 16-way ensemble-in-one, EMA. OOF 0.95062, LB 0.95048. But nested CV improvement over HGBC-TE was only +0.0001 (below significance threshold).
2. **FT-Transformer (Masaya Kawamata):** CV 0.95063. A deep learning transformer architecture for tabular data.

The top LB scores (0.95224, 0.95212) are ~0.002 above the honest community ceiling. Given that binomial noise on the public split's minority classes is ±0.001-0.002, these scores are likely within the public LB noise band and may not survive the private split.

---

## Key Lessons

1. **CV is truth.** The public LB is 20% of test (~59k rows). One flipped minority prediction moves BA by ~0.0001. Chasing the LB means chasing noise.

2. **Do not pre-impute.** MNAR patterns in synthetic data encode target signal. Pre-imputation with median/mode destroys this. The HGBC (and XGBoost/LGBM natively) handle NaN without pre-imputation.

3. **Model diversity requires architectural diversity.** LGBM + CatBoost + XGBoost are not diverse enough. Real diversity requires a different algorithm family (HGBC, neural nets) or a different problem decomposition (OvR).

4. **Feature representation is the primary lever.** When all model-tuning/ensemble attempts plateau, the bottleneck is the feature representation -- not the model or its hyperparameters.

5. **The ensemble trap is real.** Under train/test shift, ensemblers overfit the training distribution's error-correlation structure. A single robust model transfers more cleanly.

---

## Next Candidate Approaches (Not Yet Tried)

1. **Optuna hyperparameter tuning for HGBC:** We have never tuned our HGBC parameters. Key params: `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_bins`. A 20-trial Optuna search with 3-fold CV could improve OOF toward 0.9502-0.9504.

2. **FT-Transformer via pytorch-tabular:** The highest confirmed single-model CV (0.95063). Requires `pip install pytorch_tabular` and GPU training. Architecturally different from all tree models -- genuine diversity. High implementation complexity, but the highest realistic ceiling.

3. **XGBoost One-vs-Rest (OvR):** 3 independent binary XGBoost classifiers (one per class) with per-class `scale_pos_weight`. From the community CV table: CV 0.95036, LB 0.95040 -- extremely stable gap of +0.00004. Documented as providing genuine diversity from standard multiclass GBDTs.
