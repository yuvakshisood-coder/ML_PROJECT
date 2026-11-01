# Presentation and Demo Outline (about 10–12 minutes)

| Slide | Content | Speaker |
|---|---|---|
| 1 | Title, team, the one-sentence goal | M4 |
| 2 | Problem: churn + why black boxes are a problem | M1 |
| 3 | Dataset (7,043 x 21, 26.5% churn) and 3 key EDA findings | M1 |
| 4 | Preprocessing pipeline; "fit on train only"; why no scaling for RF | M1 |
| 5 | Models and 5 experiments (E1–E5) | M2 |
| 6 | Results table + ROC/PR + confusion matrix; accuracy is not enough | M2 |
| 7 | SHAP global: beeswarm and bar | M3 |
| 8 | SHAP local: 3 customers (confident churner, loyal, missed churner) | M3 |
| 9 | LIME and SHAP vs LIME agreement | M4 |
| 10 | Stability: LIME seeds, RF seeds, subsample | M4 |
| 11 | Causal caveat: month-to-month example | M4 |
| 12 | Conclusion, limitations, future work | M4 |

## Live demo (2 min)
Open notebook 04, change the customer index in `cands`, rerun the waterfall + LIME cells, and narrate the explanation.

## Questions every member should be able to answer
1. Why is the preprocessor inside the Pipeline? (No leakage in CV.)
2. Why is accuracy misleading here? (73.5% from always predicting "stay".)
3. What does a SHAP value mean? What does the base value mean?
4. Why do SHAP and LIME differ in scale? (Probability points vs surrogate-model weights.)
5. Why can't we say month-to-month contracts *cause* churn?
6. Why did the model miss customer 3?
