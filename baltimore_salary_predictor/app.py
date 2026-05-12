from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# ─── Data ─────────────────────────────────────────────────────────────────────
JOB_TITLES = [
    "Police Officer", "Firefighter", "Engineer", "Data Analyst",
    "Budget Analyst", "Public Health Nurse", "Social Worker",
    "IT Specialist", "Sanitation Worker", "Administrative Officer",
    "Project Manager", "Urban Planner", "Legal Counsel",
    "Financial Analyst", "HR Specialist", "Librarian",
    "Correctional Officer", "Traffic Engineer", "Inspector",
    "Community Liaison"
]

AGENCIES = [
    "Police Department", "Fire Department", "Department of Transportation",
    "Health Department", "Department of Finance", "Public Works",
    "Department of Housing", "IT Department", "Human Resources",
    "Department of Education", "City Council", "Department of Recreation",
    "Office of the Mayor", "Department of Social Services",
    "Baltimore City Courts", "Planning Department",
    "Environmental Control Board", "Department of Aging"
]

BASE_SALARY = {
    "Police Officer": 65000, "Firefighter": 62000, "Engineer": 85000,
    "Data Analyst": 78000, "Budget Analyst": 72000, "Public Health Nurse": 70000,
    "Social Worker": 55000, "IT Specialist": 80000, "Sanitation Worker": 48000,
    "Administrative Officer": 55000, "Project Manager": 90000,
    "Urban Planner": 75000, "Legal Counsel": 95000, "Financial Analyst": 82000,
    "HR Specialist": 65000, "Librarian": 52000, "Correctional Officer": 58000,
    "Traffic Engineer": 76000, "Inspector": 60000, "Community Liaison": 50000
}

AGENCY_MULTIPLIER = {
    "Police Department": 1.08, "Fire Department": 1.06,
    "Department of Transportation": 1.03, "Health Department": 1.02,
    "Department of Finance": 1.10, "Public Works": 0.98,
    "Department of Housing": 0.97, "IT Department": 1.12,
    "Human Resources": 1.00, "Department of Education": 0.99,
    "City Council": 1.15, "Department of Recreation": 0.94,
    "Office of the Mayor": 1.20, "Department of Social Services": 0.96,
    "Baltimore City Courts": 1.10, "Planning Department": 1.04,
    "Environmental Control Board": 1.01, "Department of Aging": 0.95
}

# ─── Mock Prediction (replace body with real model after training) ─────────────
def predict_salary_mock(job_title, agency, years_exp, model_type="svr"):
    """
    Mock prediction engine supporting multiple model types.
    After running train_model.py, replace this with the real model loader.

    model_type: 'svr' | 'random_forest' | 'linear_regression'
    """
    base = BASE_SALARY.get(job_title, 60000)
    agency_mult = AGENCY_MULTIPLIER.get(agency, 1.0)
    experience_bonus = years_exp * 1200

    # SVR introduces slightly different noise pattern (kernel smoothing simulation)
    if model_type == "svr":
        noise = np.random.normal(0, 1800)
        predicted = (base + experience_bonus) * agency_mult + noise
    elif model_type == "random_forest":
        noise = np.random.normal(0, 1500)
        predicted = (base + experience_bonus) * agency_mult + noise
    else:
        noise = np.random.normal(0, 2500)
        predicted = (base + experience_bonus) * agency_mult + noise

    return round(max(30000, predicted), 2)


# ─── XAI Feature Importance ───────────────────────────────────────────────────
FEATURE_IMPORTANCE_SVR = [
    {"feature": "Job Title",           "importance": 0.49, "icon": "briefcase"},
    {"feature": "Years of Experience", "importance": 0.30, "icon": "clock"},
    {"feature": "Agency / Department", "importance": 0.13, "icon": "building"},
    {"feature": "Seniority Score",     "importance": 0.05, "icon": "star"},
    {"feature": "Experience Tier",     "importance": 0.03, "icon": "layer-group"},
]

FEATURE_IMPORTANCE_RF = [
    {"feature": "Job Title",           "importance": 0.52, "icon": "briefcase"},
    {"feature": "Years of Experience", "importance": 0.28, "icon": "clock"},
    {"feature": "Agency / Department", "importance": 0.14, "icon": "building"},
    {"feature": "Seniority Score",     "importance": 0.04, "icon": "star"},
    {"feature": "Experience Tier",     "importance": 0.02, "icon": "layer-group"},
]

FEATURE_IMPORTANCE_LR = [
    {"feature": "Job Title",           "importance": 0.45, "icon": "briefcase"},
    {"feature": "Years of Experience", "importance": 0.32, "icon": "clock"},
    {"feature": "Agency / Department", "importance": 0.15, "icon": "building"},
    {"feature": "Seniority Score",     "importance": 0.05, "icon": "star"},
    {"feature": "Experience Tier",     "importance": 0.03, "icon": "layer-group"},
]

FI_MAP = {
    "svr":              FEATURE_IMPORTANCE_SVR,
    "random_forest":    FEATURE_IMPORTANCE_RF,
    "linear_regression":FEATURE_IMPORTANCE_LR,
}

MODEL_LABELS = {
    "svr":               "SVR · RBF Kernel",
    "random_forest":     "Random Forest",
    "linear_regression": "Linear Regression",
}

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', job_titles=JOB_TITLES, agencies=AGENCIES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json()
        job_title  = data.get('job_title', '')
        agency     = data.get('agency', '')
        years_exp  = float(data.get('years_experience', 0))
        model_type = data.get('model_type', 'svr')   # NEW: model selector

        if not job_title or not agency:
            return jsonify({'error': 'Missing required fields'}), 400

        salary = predict_salary_mock(job_title, agency, years_exp, model_type)

        # Confidence varies by model & experience range
        if model_type == "svr":
            confidence = round(random.uniform(91, 96), 1)
        elif model_type == "random_forest":
            confidence = round(random.uniform(93, 97), 1)
        else:
            confidence = round(random.uniform(84, 90), 1)

        return jsonify({
            'predicted_salary':   salary,
            'confidence':         confidence,
            'job_title':          job_title,
            'agency':             agency,
            'years_experience':   years_exp,
            'model_type':         model_type,
            'model_label':        MODEL_LABELS.get(model_type, model_type),
            'feature_importance': FI_MAP.get(model_type, FEATURE_IMPORTANCE_SVR),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metadata', methods=['GET'])
def metadata():
    return jsonify({
        'job_titles': JOB_TITLES,
        'agencies':   AGENCIES,
        'models':     list(MODEL_LABELS.keys()),
        'version':    '2.0.0',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
