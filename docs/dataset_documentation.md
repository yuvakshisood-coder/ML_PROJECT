# 2. Dataset Documentation

| Item | Detail |
|---|---|
| Name | Telco Customer Churn |
| Source | https://www.kaggle.com/datasets/blastchar/telco-customer-churn (IBM sample dataset) |
| Usage terms | Per the Kaggle dataset page; used for academic coursework. Not redistributed in this repository. |
| Samples | 7,043 customers (no duplicate rows or IDs) |
| Columns | 21 (1 identifier, 19 features, 1 target) |
| Target | `Churn` – Yes (1,869 = 26.5%) / No (5,174 = 73.5%) |
| Train / test | 5,634 / 1,409, stratified 80/20, seed 42 |
| Encoded features | 40 after one-hot encoding |

## Features
* **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* **Account:** `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
* **Dropped:** `customerID`

## Data quality
* `TotalCharges` is text; 11 blank values, all for customers with `tenure = 0` (not yet billed). Converted to NaN and median-imputed within the pipeline (train statistics only).
* "No internet service" / "No phone service" are valid categories that duplicate information in `InternetService` / `PhoneService` (redundant, collinear columns).
* No other missing values; no impossible values found.

## Limitations
* **Single snapshot, observational:** no time dimension and no interventions, so no causal conclusions.
* **Synthetic origin:** an IBM sample dataset, not necessarily representative of real operators (the problem statement's Jio/Airtel/Vodafone framing is an analogy; charges in this dataset are in the dataset's currency, not rupees).
* **Moderate imbalance** (26.5% positives) and modest size (about 1,900 positives).
* **Correlated features** (tenure vs TotalCharges, service add-ons vs internet type) complicate attribution.
* **Sensitive attributes** (`gender`, `SeniorCitizen`) are present; fairness should be checked rather than assumed.
* No information on *why* customers left, on prices of competitors, or on service quality.
