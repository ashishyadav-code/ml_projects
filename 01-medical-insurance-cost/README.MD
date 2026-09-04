# 🏥 Medical Insurance Cost Prediction (Production-Grade ML Pipeline)

An end-to-end Machine Learning system that predicts individualized medical insurance premiums using patient demographic and health risk factors. Built from the ground up using modular OOP software engineering patterns, target skewness correction, and an interactive Streamlit UI.

Part of my **10-Project Pure Classical & Production ML Portfolio**.

---

## 🎯 The Problem

Insurance companies need accurate risk assessment models to price premiums fairly without undercharging high-risk policyholders or overcharging healthy ones.

In real-world health insurance data:
- Medical charges are **heavily right-skewed** (a few catastrophic illnesses cause extreme tail costs).
- Linear assumptions break down without proper transformation.
- Models must handle both continuous biometric factors (BMI, age) and categorical habits (smoking status, region).

This project solves this by treating ML not as an exploratory notebook script, but as a **reproducible, modular software system**.

---

## 🏗️ Architecture & Engineering Workflow

```text
01-medical-insurance-cost/
├── Artifacts/
│   ├── preprocessor.pkl             # Fitted ColumnTransformer pipeline
│   └── model.pkl                    # Best trained regression model
├── Data/
│   ├── Raw/insurance.csv            # Original raw dataset
│   └── Processed/                   # Train/Test splits (80/20)
├── Notebooks/
│   └── 01_EDA_and_Baselines.ipynb  # EDA, distributions, feature insights
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Clean train-test splitting
│   │   ├── data_transformation.py  # Skewness fix, scaling & one-hot encoding
│   │   └── model_trainer.py         # Multi-model benchmarking & artifact saving
│   ├── Pipeline/
│   │   ├── train_pipeline.py        # End-to-end orchestration
│   │   └── predict_pipeline.py      # Inference pipeline + reverse log transformation
│   ├── exception.py                 # Custom system traceback exception handling
│   ├── logger.py                    # Production rotating execution logger
│   └── utils.py                     # Metric evaluation & serialization helpers
├── app.py                           # Clean Streamlit web application
└── requirements.txt
```

---

## 💡 Key Engineering Highlights

### 1. Target Skewness Fix (`log1p` & `expm1`)
The raw target `charges` showed severe positive skewness ($> 1.5$), which harms regression models by letting outliers dominate the loss function.
- Applied `np.log1p(y)` during training to normalize the residual distribution.
- In `predict_pipeline.py`, predictions are seamlessly converted back to real dollars via `np.expm1(y_pred)`.

### 2. Zero Data Leakage Pipeline
- Preprocessing steps (imputation, `StandardScaler`, `OneHotEncoder`) are bundled inside `sklearn.compose.ColumnTransformer`.
- Fits **only** on the training set and transforms the test and future inference inputs.

### 3. Comprehensive Model Benchmarking
We benchmarked multiple algorithms under identical conditions:
- Linear Regression, Ridge, Lasso
- Random Forest Regressor
- XGBoost Regressor

The best performing model on the holdout test set was chosen and serialized into `Artifacts/model.pkl`.

### 4. Custom Inference Engine & UI
- `CustomData` class strictly type-checks and structures inputs into pandas format.
- `app.py` provides an intuitive Streamlit interface featuring risk indicators, live cost estimations, and breakdown insights.

---

## 📊 Dataset Overview

- **Source**: Medical Cost Personal Datasets (US Census Bureau / Medical Cost benchmarks)
- **Observations**: 1,338 records
- **Features**:
  - `age`: Age of primary beneficiary
  - `sex`: Insurance contractor gender (`male`, `female`)
  - `bmi`: Body mass index ($kg/m^2$)
  - `children`: Number of dependents covered
  - `smoker`: Smoking status (`yes`, `no`) — *highest feature importance*
  - `region`: Beneficiary's residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`)
- **Target**: `charges` (Individual medical costs billed by health insurance)

---

## 🚀 How to Run Locally

### 1. Clone the repository & navigate to project folder
```bash
git clone <your-repo-url>
cd 01-medical-insurance-cost
```

### 2. Activate your virtual environment
```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline (Optional)
To regenerate artifacts (`preprocessor.pkl`, `model.pkl`):
```bash
python -m src.components.model_trainer
```

### 5. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test interactive predictions.

---

## 🛠️ Tech Stack

- **Core ML & Math**: Python 3.10+, Scikit-Learn, XGBoost, NumPy, Pandas
- **Interface**: Streamlit
- **Dev & Ops**: OOP Pipeline pattern, Pickle serialization, Custom Logging & Exception handling
