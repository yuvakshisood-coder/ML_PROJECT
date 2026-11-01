# Explainable AI for Black-Box Classification – Telco Customer Churn

Predict which telecom customers will churn with Logistic Regression and Random Forest, then open the black box with **SHAP** (primary) and **LIME** (comparison) and ask whether the explanations can be trusted.

> We don't just want to predict which customers will leave. We want to understand how and why our model arrives at those predictions, and whether those explanations can be trusted.

## Quick start

```bash
git clone <your-repo-url> && cd Explainable-AI-Customer-Churn
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Dataset: put WA_Fn-UseC_-Telco-Customer-Churn.csv in dataset/  (see dataset/README.md)
# If missing, src/utils.py tries to download it automatically with kagglehub.
jupyter lab
```

Run the notebooks **in order** (about 5 min total; notebook 02 hyper-parameter search is the slowest step):

| # | Notebook | Purpose | Deliverables |
|---|---|---|---|
| 01 | `01_Data_Preprocessing.ipynb` | Dataset documentation, 4 EDA analyses, preprocessing design | 2, 3, 4 |
| 02 | `02_Model_Training.ipynb` | Baseline + 4 further configurations, 5-fold CV, saves models | 5, 6 |
| 03 | `03_Model_Evaluation.ipynb` | Test metrics, confusion matrices, ROC/PR, threshold and failure analysis | 7, 8 |
| 04 | `04_Explainable_AI.ipynb` | SHAP global/local, LIME, SHAP vs LIME, stability, LR vs RF, causal caveat | 5, 6, 7, 8 |

Google Colab: upload the repo, run `!pip install -r requirements.txt`, and keep the folder layout.

## Repository layout

```
├── dataset/            raw CSV (+ dataset/README.md with documentation)
├── notebooks/          01–04, executed with outputs
├── src/utils.py        shared loading, cleaning, split, preprocessing, SHAP helper
├── visualizations/     all figures produced by the notebooks (PNG)
├── results/            metric tables (CSV)
├── models/             fitted pipelines (*.joblib, git-ignored; recreate with notebook 02)
├── docs/               problem definition, dataset documentation, final report, presentation outline
├── requirements.txt
└── README.md
```

## Reproducibility
* Global seed `SEED = 42` (`src/utils.py`) for the split, models, CV folds, SHAP sample and LIME.
* One shared 80/20 stratified split used by every notebook.
* All transformations (imputation, scaling, encoding) live in scikit-learn `Pipeline`s and are fitted on the training data only.
* Tested with Python 3.12, scikit-learn 1.8, SHAP 0.54, pandas 2.x. Minor numeric differences across library versions are expected.

## Headline results (test set, 1,409 customers)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| E1 LR baseline | 0.657 | 0.559 | 0.604 | 0.842 | 0.634 |
| E2 LR balanced | 0.504 | 0.783 | 0.614 | 0.841 | 0.632 |
| E3 RF default | 0.612 | 0.497 | 0.549 | 0.821 | 0.616 |
| E4 RF balanced + regularised | 0.550 | 0.741 | 0.631 | 0.841 | 0.650 |
| E5 RF tuned | 0.671 | 0.508 | 0.578 | 0.845 | 0.659 |

Top SHAP features for the tuned Random Forest: month-to-month contract, tenure, fibre-optic internet, no online security, no tech support. See `docs/final_report.md` for the full analysis and limitations.

## Team and work division

| Member | Responsibility | Main files |
|---|---|---|
| Member 1 | Dataset, EDA, preprocessing | notebook 01, `src/utils.py`, `docs/dataset_documentation.md` |
| Member 2 | Logistic Regression, Random Forest, evaluation | notebooks 02, 03 |
| Member 3 | SHAP global and local explanations | notebook 04 sections 1–2 |
| Member 4 | LIME, comparison, stability, report | notebook 04 sections 3–7, `docs/` |

Everyone should be able to explain the full pipeline (see `docs/presentation_outline.md`).

## Pushing to GitHub

```bash
git init && git add . && git commit -m "Initial commit: XAI churn project"
git branch -M main
git remote add origin https://github.com/<user>/Explainable-AI-Customer-Churn.git
git push -u origin main
```
The raw CSV and model files are git-ignored (Kaggle redistribution terms; file size). Remove the `dataset/*.csv` line from `.gitignore` if your course requires the CSV in the repo.

## Data source and licence
Telco Customer Churn, IBM sample data, via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Check the licence shown on the Kaggle page before redistributing.
