# Motor Insurance **Fraud‑Detection** Platform

> End‑to‑end machine‑learning pipeline + Streamlit dashboard for classifying fraudulent motor‑insurance claims with both **supervised** and **unsupervised** techniques.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Quick Start](#quick-start)
4. [Repository Layout](#repository-layout)
5. [Usage Examples](#usage-examples)
6. [Streamlit Dashboard](#streamlit-dashboard)
7. [Extensibility](#extensibility)
8. [License](#license)

---

## Project Overview

Motor‑insurance fraud inflates premiums world‑wide and can cost carriers billions of USD annually.  This project provides an **academic‑grade yet production‑ready** code‑base that:

* Cleans & engineers claim‑level data (≈ 40 predictors)
* Trains multiple models (Logistic Regression → Balanced Random Forest → Isolation Forest)
* Benchmarks them via ROC‑AUC, PR‑AUC, lift, cumulative‑gain & reliability curves
* Explains predictions with SHAP
* Serves an interactive **Streamlit** dashboard for EDA, training and prediction (batch & single‑claim)

Although the reference dataset is an anonymised CSV (`insurance_claims.csv`), the pipeline generalises to any tabular claims file that contains a binary `fraud_reported` target.

---

## Key Features

| Category             | Highlights                                                                                                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Engineering** | • Robust missing‑value handling & outlier capping  <br>• Domain‑aware features (e.g. `is_rush_hour`)                                                                                               |
| **Model Zoo**        | • Classic supervised learners *(LR, DT, RF, GB, XGB, SVM)*  <br>• Ensemble stacking + isotonic calibration  <br>• Unsupervised detectors *(Isolation Forest, LOF, One‑Class SVM, K‑Means, DBSCAN)* |
| **Auto‑Tuning**      | Grid‑search for Random Forest hyper‑parameters with cross‑validated F1 objective                                                                                                                   |
| **Evaluation Suite** | Confusion heat‑maps, ROC, PR, lift, cumulative‑gain, calibration, probability densities                                                                                                            |
| **Explainability**   | Global & local SHAP (bar + force plots)                                                                                                                                                            |
| **Visual EDA**       | Class balance, correlation heat‑map, violin plots, t‑SNE projection                                                                                                                                |
| **Dashboard**        | Zero‑code UI built with Streamlit → upload data, train, explain, predict, download results                                                                                                         |
| **Modularity**       | Each stage lives in its own module (`preprocessing.py`, `evaluation.py`, …) for easy reuse                                                                                                         |

---

## Quick Start

### 1 · Clone & Setup

```bash
# clone the repo
$ git clone https://github.com/LieAbility-Insurance-LLC/Analysis.git
$ cd insurance‑fraud‑ml

# create virtual env (recommended)
$ python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies in one go
$ python setup_env.py        # installs pandas, scikit‑learn, imbalanced‑learn, xgboost, shap, …
```

### 2 · Prepare Data

Place `insurance_claims.csv` in the project root **or** point to your own CSV with identical column names (target column = `fraud_reported`, values `Y`/`N` or `1`/`0`).

### 3 · Command‑Line Pipeline

```bash
# trains models, saves plots → ./figures, best model → ./models
$ python main.py
```

Logs appear in the terminal; plots are written as `fig_000.png`, `fig_001.png`, …

### 4 · Launch the Dashboard

```bash
$ streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and explore 📊

---

## Repository Layout

```
.
├── app.py                # Streamlit dashboard (UI layer)
├── main.py               # End‑to‑end CLI workflow
├── setup_env.py          # One‑shot dependency installer
├── data_handling.py      # Safe loading + schema validation
├── preprocessing.py      # Cleaning, encoding, scaling
├── feature_engineering.py# Domain‑specific feature additions + top‑N selector
├── model_training.py     # Supervised & unsupervised model routines
├── evaluation.py         # Metrics, plots, SHAP helpers
├── models/               # ← auto‑saved .pkl models
├── figures/              # ← auto‑generated PNG plots
└── README.md             # You are here 😉
```

---

## Usage Examples

### Batch Scoring with Saved Model

```python
import joblib, pandas as pd
from preprocessing import preprocess_data
from feature_engineering import feature_engineering

# 1 · Load artefacts
model  = joblib.load("models/best_model.pkl")
preproc= joblib.load("models/preprocessor.pkl")

# 2 · Prepare new claims
new = pd.read_csv("new_claims.csv")
X = feature_engineering(preprocess_data(new))
X_enc = preproc.transform(X.drop(columns=["fraud_reported"], errors="ignore"))

# 3 · Predict probability of fraud
proba = model.predict_proba(X_enc)[:, 1]
print(pd.Series(proba, name="fraud_probability"))
```

### Extending with Additional Models

Add any scikit‑learn compatible estimator in **`model_training.py → models_sup`** and it will automatically inherit evaluation plots with no extra code.

---

## Streamlit Dashboard

<p align="center">
  <img width="740" alt="dashboard" src="https://github.com/your‑org/insurance‑fraud‑ml/raw/main/docs/dashboard_light.png">
</p>

* **Upload** or auto‑load sample data
* **EDA**: class balance, correlation, t‑SNE
* **Preprocess**: one‑click cleaning + feature engineering
* **Train**: optional hyper‑parameter tuning, SMOTE balancing
* **Predict**: batch CSV or single‑claim form
* **Explain**: SHAP bar / force plots rendered inline

---

## Extensibility

| Task                            | Where to Hook In                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Replace SMOTE with ADASYN       | `main.py → SMOTE` line · change sampler                                                                     |
| Log to MLflow / Weights\&Biases | Wrap training loop in `main.py` (or `app.py`) with your tracking calls                                      |
| Serve as REST API               | Save model & `preprocessor.pkl` then expose via FastAPI (`joblib.load` at startup)                          |
| New plots                       | Add matplotlib/Seaborn functions in `evaluation.py` – they auto‑save via the global `plt.show` monkey‑patch |

---

## License

Distributed under the **MIT License** – see [`LICENSE`](LICENSE) for details.
