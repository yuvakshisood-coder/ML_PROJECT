# 1. Problem Definition and Background

## Problem statement
A telecom operator loses revenue when customers leave (churn). Machine-learning models can flag at-risk customers, but the strongest models are black boxes, and a retention team cannot act on, or trust, a score without knowing *why* it was produced. This project builds churn classifiers and uses Explainable AI (XAI) to examine how they reach their predictions, and how far those explanations can be trusted.

## Objectives
1. Build a baseline (Logistic Regression) and a non-linear model (Random Forest) for churn prediction.
2. Evaluate with metrics suited to an imbalanced target (precision, recall, F1, ROC-AUC, PR-AUC).
3. Explain the Random Forest globally and locally with SHAP, and locally with LIME.
4. Assess the explanations: agreement between methods, stability, and the limits of causal interpretation.

## Research questions
* **RQ1** Which features does the model rely on most across customers?
* **RQ2** Why is a specific customer predicted to churn (or not)? Do explanations differ between customers?
* **RQ3** Do SHAP and LIME agree on feature attributions?
* **RQ4** Are explanations stable across LIME seeds, Random Forest seeds and training subsamples?
* **RQ5** Does the black box rank features like an interpretable model does, and can any explanation be read as a *cause* of churn?

## Motivation
Acquiring a customer typically costs more than retaining one, so targeted retention offers are valuable, but only if the targeting logic can be audited. Regulators and internal stakeholders increasingly expect explanations for automated decisions (e.g. the "right to explanation" debate around GDPR). Explanations are also easy to over-interpret, so measuring their reliability is part of responsible use.

## Background and short literature review
* **Churn prediction.** Churn is a standard supervised binary classification task on customer-account data; tree ensembles and logistic regression are common, strong baselines for tabular data.
* **Random Forest** (Breiman, 2001) averages many decorrelated decision trees; it captures non-linearities and interactions but is not directly interpretable.
* **LIME** (Ribeiro, Singh & Guestrin, KDD 2016) fits a sparse linear model on perturbed samples around one instance to approximate the black box locally.
* **SHAP** (Lundberg & Lee, NeurIPS 2017) attributes a prediction to features using Shapley values from cooperative game theory, giving additive, consistent attributions; **TreeSHAP** (Lundberg et al., Nature Machine Intelligence 2020) computes them exactly and efficiently for tree models.
* **Limitations of explanations.** Alvarez-Melis & Jaakkola (2018) show that explanations can be unstable under small input changes; Slack et al. (AIES 2020) show LIME/SHAP can be manipulated; Kumar et al. (ICML 2020) discuss conceptual problems of treating Shapley values as feature importance, especially with correlated features. Molnar, *Interpretable Machine Learning*, is a practical overview.
* **Causality.** Explanations describe model behaviour, not causal effects; causal claims need experimental or causal-inference designs (Pearl, 2009).

## References
1. Breiman, L. (2001). Random Forests. *Machine Learning* 45, 5–32.
2. Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier. *KDD*.
3. Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 56–67.
5. Alvarez-Melis, D., Jaakkola, T. (2018). On the Robustness of Interpretability Methods. *ICML Workshop on Human Interpretability*.
6. Slack, D. et al. (2020). Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods. *AIES*.
7. Kumar, I. E. et al. (2020). Problems with Shapley-value-based explanations as feature importance measures. *ICML*.
8. Molnar, C. *Interpretable Machine Learning* (online book).
9. Pearl, J. (2009). *Causality*, 2nd ed. Cambridge University Press.

*(Verify the citation details before submission.)*
