"""
Baltimore Salary Predictor v2 — ML Training Script with SVM
============================================================
Models trained:
  1. Linear Regression  (baseline)
  2. SVR — RBF Kernel   (Support Vector Regression)  ← PRIMARY
  3. Random Forest      (ensemble comparison)

Run:
    python notebooks/train_model.py

After training, paste the loader snippet printed at the end into app.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.ensemble          import RandomForestRegressor
from sklearn.linear_model      import LinearRegression
from sklearn.svm               import SVR
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA  (synthetic — swap with real Baltimore CSV)
# ══════════════════════════════════════════════════════════════════════════════
# Real dataset: https://data.baltimorecity.gov/datasets/baltimore-city-employee-salaries
# To use real data, replace this block with:
#     df = pd.read_csv('data/baltimore_salaries.csv')
#     df = df.rename(columns={'JobTitle':'job_title', 'AgencyName':'agency',
#                             'AnnualSalary':'annual_salary'})
#     df['years_experience'] = 0   # derive from HireDate if available

np.random.seed(42)
N = 1400

JOB_TITLES = [
    "Police Officer","Firefighter","Engineer","Data Analyst",
    "Budget Analyst","Public Health Nurse","Social Worker","IT Specialist",
    "Sanitation Worker","Administrative Officer","Project Manager","Urban Planner",
    "Legal Counsel","Financial Analyst","HR Specialist","Librarian",
    "Correctional Officer","Traffic Engineer","Inspector","Community Liaison"
]
AGENCIES = [
    "Police Department","Fire Department","Department of Transportation",
    "Health Department","Department of Finance","Public Works",
    "Department of Housing","IT Department","Human Resources",
    "Department of Education","City Council","Department of Recreation",
    "Office of the Mayor","Department of Social Services",
    "Baltimore City Courts","Planning Department",
    "Environmental Control Board","Department of Aging"
]
BASE = {
    "Police Officer":65000,"Firefighter":62000,"Engineer":85000,
    "Data Analyst":78000,"Budget Analyst":72000,"Public Health Nurse":70000,
    "Social Worker":55000,"IT Specialist":80000,"Sanitation Worker":48000,
    "Administrative Officer":55000,"Project Manager":90000,"Urban Planner":75000,
    "Legal Counsel":95000,"Financial Analyst":82000,"HR Specialist":65000,
    "Librarian":52000,"Correctional Officer":58000,"Traffic Engineer":76000,
    "Inspector":60000,"Community Liaison":50000
}

job_col = np.random.choice(JOB_TITLES, N)
agen_col = np.random.choice(AGENCIES, N)
exp_col  = np.random.randint(0, 40, N)

salaries = [
    max(30000, round(BASE.get(j, 60000) + y*1200 + np.random.normal(0, 3000), 2))
    for j, y in zip(job_col, exp_col)
]

df = pd.DataFrame({
    'job_title':        job_col,
    'agency':           agen_col,
    'years_experience': exp_col,
    'annual_salary':    salaries,
})
os.makedirs('data', exist_ok=True)
df.to_csv('data/baltimore_salaries_synthetic.csv', index=False)
print(f"✓ Dataset: {df.shape}  →  data/baltimore_salaries_synthetic.csv")
print(df.head(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Salary Statistics ──────────────────────────────────")
print(df['annual_salary'].describe().round(2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Baltimore Salary — EDA', fontsize=16, fontweight='bold', color='#1e293b')
plt.rcParams.update({'axes.facecolor':'#f8fafc','figure.facecolor':'#f8fafc'})

# 2a. Salary distribution
axes[0,0].hist(df['annual_salary'], bins=40, color='#6366f1', edgecolor='white', alpha=0.85)
axes[0,0].set_title('Salary Distribution');axes[0,0].set_xlabel('Annual Salary ($)')

# 2b. Experience vs Salary scatter
axes[0,1].scatter(df['years_experience'], df['annual_salary'], alpha=0.3, color='#22d3ee', s=14)
m,b = np.polyfit(df['years_experience'], df['annual_salary'], 1)
axes[0,1].plot(range(41), [m*x+b for x in range(41)], color='#f43f5e', lw=2)
axes[0,1].set_title('Experience vs Salary');axes[0,1].set_xlabel('Years Exp')

# 2c. Top-10 Job Titles
top10 = df.groupby('job_title')['annual_salary'].mean().nlargest(10)
axes[0,2].barh(top10.index, top10.values, color='#818cf8')
axes[0,2].set_title('Top 10 Titles by Avg Salary')

# 2d. Agency avg salary (top 8)
top_ag = df.groupby('agency')['annual_salary'].mean().nlargest(8)
axes[1,0].bar(range(len(top_ag)), top_ag.values, color='#34d399')
axes[1,0].set_xticks(range(len(top_ag)))
axes[1,0].set_xticklabels(top_ag.index, rotation=40, ha='right', fontsize=7)
axes[1,0].set_title('Top 8 Agencies by Avg Salary')

# 2e. Box plot by experience tier
df['exp_tier_label'] = pd.cut(df['years_experience'],
    bins=[-1,5,10,20,40], labels=['0-5','6-10','11-20','21-40'])
tier_data = [df[df['exp_tier_label']==t]['annual_salary'].values
             for t in ['0-5','6-10','11-20','21-40']]
axes[1,1].boxplot(tier_data, labels=['0-5','6-10','11-20','21-40'], patch_artist=True,
    boxprops=dict(facecolor='#818cf8', alpha=0.7))
axes[1,1].set_title('Salary by Experience Tier')

# 2f. Correlation heatmap
num_df = df[['years_experience','annual_salary']].copy()
num_df['job_enc']    = LabelEncoder().fit_transform(df['job_title'])
num_df['agency_enc'] = LabelEncoder().fit_transform(df['agency'])
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='Blues',
            ax=axes[1,2], linewidths=0.5)
axes[1,2].set_title('Feature Correlation')

plt.tight_layout()
plt.savefig('data/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ EDA saved → data/eda_plots.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
df_m = df.copy()

bins   = [-1, 2, 5, 10, 20, 40]
labels = [0, 1, 2, 3, 4]
df_m['exp_tier']       = pd.cut(df_m['years_experience'], bins=bins, labels=labels).astype(int)
df_m['seniority_score']= df_m['years_experience'] / 40.0

le_job    = LabelEncoder()
le_agency = LabelEncoder()
df_m['job_enc']    = le_job.fit_transform(df_m['job_title'])
df_m['agency_enc'] = le_agency.fit_transform(df_m['agency'])

FEATURES = ['job_enc', 'agency_enc', 'years_experience', 'exp_tier', 'seniority_score']
X = df_m[FEATURES].values
y = df_m['annual_salary'].values

print(f"\n✓ Features: {FEATURES}")
print(f"  X={X.shape}  y={y.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train={X_train.shape}  Test={X_test.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELS
# ══════════════════════════════════════════════════════════════════════════════
# SVR REQUIRES StandardScaler — use sklearn Pipeline to bundle scaler + model
models = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ]),

    # ── SVR (RBF Kernel) ── PRIMARY MODEL ──────────────────────────────────
    # C=100 : allows larger margin violations (less regularization)
    # gamma=0.1 : RBF kernel width — controls how far influence of one sample reaches
    # epsilon=0.1 : epsilon-insensitive tube — errors < ε are ignored
    'SVR (RBF Kernel)': Pipeline([
        ('scaler', StandardScaler()),          # CRITICAL for SVM
        ('model',  SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)),
    ]),

    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)),
    ]),
}

results = {}
print("\n── Model Evaluation ────────────────────────────────────")
print(f"{'Model':<28} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("─" * 60)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'pipeline': pipe}
    print(f"  {name:<26} ${mae:>8,.0f}  ${rmse:>8,.0f}  {r2:>7.4f}")

print("─" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 6. BEST MODEL & SAVE
# ══════════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]['R2'])
best_pipe = results[best_name]['pipeline']
print(f"\n✓ Best model: {best_name}  (R²={results[best_name]['R2']:.4f})")

# SVR Feature Importance via Permutation (since SVR has no native importances)
try:
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42)
    fi_df = pd.DataFrame({
        'Feature':    FEATURES,
        'Importance': perm.importances_mean,
    }).sort_values('Importance', ascending=False)
    print("\n── Permutation Feature Importance ──")
    print(fi_df.to_string(index=False))
except Exception as ex:
    print(f"  (Permutation importance skipped: {ex})")

# Also print Random Forest native importances
rf_pipe = results.get('Random Forest', {}).get('pipeline')
if rf_pipe:
    rf_model = rf_pipe.named_steps['model']
    fi_rf = pd.DataFrame({
        'Feature':    FEATURES,
        'RF_Importance': rf_model.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)
    print("\n── Random Forest Feature Importance ──")
    print(fi_rf.to_string(index=False))

os.makedirs('models', exist_ok=True)
joblib.dump(best_pipe,  'models/salary_model.pkl')
joblib.dump(le_job,     'models/le_job.pkl')
joblib.dump(le_agency,  'models/le_agency.pkl')

# Also save individual models for multi-model comparison endpoint
for model_key, result in results.items():
    slug = model_key.lower().replace(' ','_').replace('(','').replace(')','').replace('/','')
    joblib.dump(result['pipeline'], f'models/{slug}.pkl')

print("\n✓ Saved: models/salary_model.pkl  (best)")
print("✓ Saved: models/svr_rbf_kernel.pkl")
print("✓ Saved: models/random_forest.pkl")
print("✓ Saved: models/linear_regression.pkl")
print("✓ Saved: models/le_job.pkl  models/le_agency.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION SNIPPET FOR app.py
# ══════════════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════╗
║  Paste this into app.py to use the real trained models  ║
╚══════════════════════════════════════════════════════════╝

import joblib, numpy as np

# Load all three models + encoders
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
""")

print("Training complete! ✅")
