# Baltimore-Salary-Prediction-Using-Support-Vector-Machine

A full-stack machine learning application predicting Baltimore City employee salaries.
This version includes **SVR (Support Vector Regression)**, Random Forest, and Linear Regression with a live model comparison panel.

---

## 📁 Project Structure

```
baltimore_salary_predictor_v2/
├── app.py                          # Flask app + mock prediction engine
├── requirements.txt                # Python dependencies
├── README.md
│
├── templates/
│   └── index.html                  # UI with model selector tabs + SVM info section
│
├── static/
│   ├── style.css                   # Dark glassmorphism stylesheet
│   └── script.js                   # Fetch API + model switching + compare panel
│
├── notebooks/
│   └── train_model.py              # Full ML pipeline: EDA + SVR + RF + LR
│
├── models/                         # Created after training
│   ├── salary_model.pkl            # Best model
│   ├── svr_rbf_kernel.pkl          # SVR pipeline
│   ├── random_forest.pkl           # RF pipeline
│   ├── linear_regression.pkl       # LR pipeline
│   ├── le_job.pkl
│   └── le_agency.pkl
│
└── data/                           # Created after training
    ├── baltimore_salaries_synthetic.csv
    └── eda_plots.png
```

---

## 🚀 Steps to Run

### Step 1 — Open in VS Code
```bash
code baltimore_salary_predictor_v2
```

### Step 2 — Create Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (Optional but Recommended) Train the ML Models
```bash
python notebooks/train_model.py
```
This will:
- Generate synthetic Baltimore salary data
- Run full EDA → save `data/eda_plots.png`
- Train **SVR (RBF)**, Random Forest, and Linear Regression
- Print MAE / RMSE / R² comparison table
- Save all 3 model pipelines to `/models/`
- Print the exact code snippet to plug real models into `app.py`

### Step 5 — Run the App
```bash
python app.py
```

### Step 6 — Open in Browser
**http://127.0.0.1:5000**

---

## 🤖 ML Models

| Model | Type | Key Params | Scaling Needed |
|-------|------|------------|----------------|
| **SVR (RBF)** | Support Vector Regression | C=100, γ=0.1, ε=0.1 | ✅ Yes |
| Random Forest | Ensemble | 200 trees, depth=12 | Optional |
| Linear Regression | Baseline | — | Optional |

### Why SVR?
- Handles **non-linear** salary relationships via RBF kernel
- Robust to outliers via **epsilon-insensitive loss**
- Excellent generalization on medium-sized datasets
- **Requires StandardScaler** — implemented via sklearn Pipeline

---

## 🔌 Upgrading to Real Models

After running `train_model.py`, replace `predict_salary_mock()` in `app.py`:

```python
import joblib, numpy as np

_models = {
    'svr':               joblib.load('models/svr_rbf_kernel.pkl'),
    'random_forest':     joblib.load('models/random_forest.pkl'),
    'linear_regression': joblib.load('models/linear_regression.pkl'),
}
_le_job    = joblib.load('models/le_job.pkl')
_le_agency = joblib.load('models/le_agency.pkl')

def predict_salary_mock(job_title, agency, years_exp, model_type='svr'):
    job_enc    = _le_job.transform([job_title])[0]
    agency_enc = _le_agency.transform([agency])[0]
    exp_tier   = min(4, int(years_exp // 5))
    seniority  = years_exp / 40.0
    X = np.array([[job_enc, agency_enc, years_exp, exp_tier, seniority]])
    pipe = _models.get(model_type, _models['svr'])
    return float(pipe.predict(X)[0])
```

---

## 🌐 Real Baltimore Dataset

1. Download: https://data.baltimorecity.gov/datasets/baltimore-city-employee-salaries
2. Save as `data/baltimore_salaries.csv`
3. In `train_model.py` replace data block with:
```python
df = pd.read_csv('data/baltimore_salaries.csv')
df = df.rename(columns={
    'JobTitle': 'job_title',
    'AgencyName': 'agency',
    'AnnualSalary': 'annual_salary'
})
df['years_experience'] = 5  # or derive from HireDate
```

---

## 🔗 API

### `POST /predict`
```json
// Request
{ "job_title": "Data Analyst", "agency": "IT Department",
  "years_experience": 5, "model_type": "svr" }

// Response
{ "predicted_salary": 84320.00, "confidence": 94.2,
  "model_label": "SVR · RBF Kernel", "feature_importance": [...] }
```

`model_type` accepts: `svr` | `random_forest` | `linear_regression`

### `GET /api/metadata`
Returns job titles, agencies, available models, and version.

