# Explainable AI for Black-Box Classification: Telco Customer Churn
**Course project report** · Team: *(names)* · Date: *(date)*

Sections 1–2 are expanded in `problem_definition.md` and `dataset_documentation.md`. All numbers below come from the executed notebooks (`results/*.csv`, `visualizations/*.png`, seed 42).

## 1. Problem and objectives
Predict customer churn and explain the predictions. Research questions RQ1–RQ5 are listed in `problem_definition.md`.

## 2. Data
7,043 customers, 19 features, target `Churn` (26.5% positive). Test set: 1,409 customers. Limitations are in `dataset_documentation.md`.

## 3. Exploratory analysis (notebook 01)
| Analysis | Figure | Finding |
|---|---|---|
| Class balance | `churn_distribution.png` | 26.5% churn; always predicting "stay" scores 73.5% accuracy, so accuracy is unsuitable. |
| Churn rate by category | `eda_churn_by_category.png` | Higher churn for month-to-month contracts, fibre-optic internet, electronic-check payment, no online security / tech support; `gender` shows almost no difference. |
| Numeric distributions and outliers | `eda_numeric_distributions.png` | Churners have much shorter tenure and somewhat higher monthly charges; no erroneous outliers. |
| Correlation | `eda_correlation.png` | `tenure` and `TotalCharges` are strongly correlated, which will split attribution. |
| Missingness | notebook 01 | 11 blank `TotalCharges`, all with tenure 0. |

## 4. Preprocessing (notebook 01, `src/utils.py`)
Drop ID; coerce `TotalCharges`; median imputation; one-hot encoding (40 features); StandardScaler for Logistic Regression only (unnecessary for trees); stratified 80/20 split; imbalance handled by `class_weight`, not resampling. Every learned transformation sits inside a `Pipeline`, so it is fitted on training folds only.

## 5. Methods (notebooks 02, 04)
* **Baseline:** Logistic Regression (`max_iter=2000`, C=1).
* **Tree model:** Random Forest, 300 trees.
* **XAI:** SHAP `TreeExplainer` (exact, 600-customer test sample; additivity verified to ~1e-15) and LIME `LimeTabularExplainer` (5,000 perturbations; one-hot columns treated as categorical).

## 6. Experiments
Five configurations, 5-fold stratified CV on training data (`results/experiments_cv.csv`):

| ID | Change | CV ROC-AUC | CV PR-AUC | CV F1 | CV precision | CV recall |
|---|---|---|---|---|---|---|
| E1 | LR baseline | 0.846 | 0.662 | 0.593 | 0.653 | 0.544 |
| E2 | + balanced class weights | 0.846 | 0.660 | 0.628 | 0.517 | 0.801 |
| E3 | RF default | 0.822 | 0.615 | 0.542 | 0.626 | 0.478 |
| E4 | RF balanced, `min_samples_leaf=5` | 0.845 | 0.658 | 0.630 | 0.554 | 0.730 |
| E5 | RF tuned (random search, 15 iterations, ROC-AUC) | 0.849 | 0.665 | 0.572 | 0.676 | 0.497 |

E5 selected: 300 trees, `max_depth=6`, `min_samples_leaf=10`, `max_features=0.3`, no class weights.

## 7. Evaluation (notebook 03, test set, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| E1 LR baseline | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | 0.634 |
| E2 LR balanced | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 | 0.632 |
| E3 RF default | 0.783 | 0.612 | 0.497 | 0.549 | 0.821 | 0.616 |
| E4 RF balanced + regularised | 0.770 | 0.550 | 0.741 | 0.631 | 0.841 | 0.650 |
| E5 RF tuned | 0.803 | 0.671 | 0.508 | 0.578 | 0.845 | 0.659 |

Figures: `confusion_matrix.png`, `roc_pr_curves.png`, `metric_comparison.png`, `threshold_analysis.png`.

## 8. Analysis of the predictive models
* **Regularisation matters for the forest.** The default RF (E3) over-fits (lowest ROC-AUC, 0.821). Limiting depth/leaf size (E4, E5) raises ROC-AUC to 0.841–0.845.
* **The complex model barely beats the simple one.** E5 reaches ROC-AUC 0.845 vs 0.842 for the baseline, a difference smaller than the CV standard deviation (about 0.01). Logistic Regression is a very strong baseline on this tabular data; the black box brings little accuracy gain, which is itself a relevant finding for the interpretability trade-off.
* **Class weights move the operating point, not the ranking.** E2 and E4 raise recall (0.78, 0.74) at the cost of precision (0.50, 0.55) and accuracy, while ROC-AUC stays essentially unchanged. The right choice depends on the cost of a missed churner vs a wasted retention offer. Lowering E5's threshold to about 0.31 gives its best F1 (about 0.64) on the test set (a threshold chosen on the test set is optimistic; in practice choose it on validation data).
* **Failure cases (E5).** 184 churners are missed (false negatives), 190 are caught, and 93 are false alarms. Missed churners have a mean tenure of 26.5 months (caught: 6.7) and 24.5% are on one- or two-year contracts; they look like loyal customers. Every true and false positive is on a month-to-month contract, so the model effectively predicts churn only within that group, which explains customer 3 below.

## 9. Explanation analysis (notebook 04)
**RQ1 – Global (SHAP).** Top features by mean |SHAP|: `Contract_Month-to-month` (0.082), `tenure` (0.050), `InternetService_Fiber optic` (0.041), `OnlineSecurity_No` (0.041), `TechSupport_No` (0.027), then `TotalCharges`, `MonthlyCharges`, electronic-check payment. This matches the EDA. See `global_shap.png`, `global_shap_bar.png`, `shap_dependence.png` (use the beeswarm to state the *direction* of each effect).

**RQ2 – Local.** `local_shap_1..3.png`:
1. *Confident churner* (P=0.84, churned): a very short tenure, month-to-month contract, high charges, fibre-optic, no online security, no tech support and electronic-check payment all push toward churn.
2. *Confident loyal customer* (P=0.01, stayed): *(describe from `local_shap_2.png`)*.
3. *Missed churner* (P=0.05, churned): *(describe from `local_shap_3.png`; the model sees a customer unlike the typical churner and nothing in the data explains why they left.)*

Across 600 customers, the single most influential feature is month-to-month contract for 511, tenure for 73, and online security or fibre-optic for 16, so explanations are dominated by a few features but do vary by customer.

**RQ3 – SHAP vs LIME.** For the three customers Spearman rank correlation is 0.95, 0.84, 0.94, top-5 overlap is 4, 5, 4 of 5, and the signs of the top-5 features always agree. Over 50 random customers: mean Spearman 0.87 (min 0.74), mean top-5 overlap 4.1 of 5. The two methods agree well on this model, but not perfectly (`shap_vs_lime.png`, `shap_lime_agreement.png`). Magnitudes are not comparable (probability points vs surrogate weights).

**RQ4 – Stability.** LIME across 5 seeds for one customer: mean pairwise Spearman 0.93, and the same five features appear in every top-5. SHAP global rankings from RFs with 4 different seeds and an 80% subsample: pairwise Spearman 0.94–0.98, with the top-5 set unchanged (only the order of ranks 3–4 changes). Explanations are reasonably stable for this dataset and model, but this is a measurement for *this* setup, not a guarantee.

**RQ5 – Black box vs glass box, and causality.** The Spearman correlation between |LR coefficient| and RF mean |SHAP| is only 0.42: the two models agree on the main drivers but rank many mid-importance features differently, reflecting correlated one-hot service columns. This shows that explanations are model-specific. SHAP and LIME describe the model's use of features, not causes: customers may *choose* month-to-month contracts because they are already unsure about the service (reverse causality/confounding), `tenure`/`TotalCharges` share credit, and the data are observational. Acceptable wording: "the model associates month-to-month contracts with higher risk". Causal claims would require an A/B retention experiment.

## 10. Conclusions
1. A regularised RF and a balanced LR reach similar discrimination (ROC-AUC about 0.84–0.85); the tuned RF is best on ROC-AUC/PR-AUC, and balanced weights trade precision for recall.
2. SHAP shows that contract type, tenure, internet service type and missing security/support add-ons drive predictions, consistent with the EDA.
3. SHAP and LIME largely agree and are fairly stable here, which increases (but does not prove) trust.
4. The black box offers little predictive advantage over Logistic Regression; if explanation is the priority, the glass-box model is a serious alternative.
5. Explanations are associations; actions should be validated with experiments.

## 11. Limitations and future work
Single dataset snapshot; synthetic-style data; one split (CV used for model comparison, but the test set is small, about 374 positives); explanation stability measured on limited seeds/samples; correlated features not handled (e.g. grouping one-hot features); no fairness audit for `gender`/`SeniorCitizen`; no cost-sensitive threshold. Future work: XGBoost, SHAP interaction values, counterfactual explanations (e.g. DiCE), causal inference on retention offers, grouping correlated features before explaining.

## 12. Individual contributions
| Member | Contribution |
|---|---|
| | |
