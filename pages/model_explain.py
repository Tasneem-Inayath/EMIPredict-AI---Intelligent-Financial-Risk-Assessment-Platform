import gdown
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import zipfile
import os

# Page config
st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability & Insights")

st.markdown("""
This page shows **metrics and prediction behavior** for the models currently in Production.  
Metrics are pulled from the **Production model version** in MLflow (both version-level and run-level).
""")

# --- Download dataset ---
file_id = '1POXHpKuXsOHlrBaI2oKM44q46o-MVx7r'
url = f'https://drive.google.com/uc?id={file_id}&export=download'
output = 'data.csv'
gdown.download(url, output, quiet=False)

@st.cache_data
def load_data():
    return pd.read_csv(output)

df = load_data()
X = df.drop(columns=['emi_eligibility'], errors='ignore')


mlruns_zip = "mlruns.zip"


import zipfile
import os

if not os.path.exists("mlruns"):
    with zipfile.ZipFile("mlruns.zip", 'r') as zip_ref:
        zip_ref.extractall("mlruns")

mlflow.set_tracking_uri("mlruns")


# --- Load models ---
classifier, regressor = None, None
    # classifier = mlflow.pyfunc.load_model(
    #     "mlruns/958227794677363910/models/m-489da889889443a8af1433bbbad454b8/artifacts"
    # )
    # regressor = mlflow.pyfunc.load_model(
    #     "mlruns/958227794677363910/models/m-064cf115b7264f9ba051e9161f74b1e3/artifacts"
    # )

try:
    classifier =  mlflow.pyfunc.load_model(
        "mlruns/958227794677363910/models/m-489da889889443a8af1433bbbad454b8/artifacts"
    )
    st.success("✅ Classifier loaded successfully from MLflow Registry")
except Exception as e:
    st.error(f"❌ Could not load classifier: {e}")

try:
    regressor =  mlflow.pyfunc.load_model(
        "mlruns/884078883015768601/models/m-064cf115b7264f9ba051e9161f74b1e3/artifacts"
    )
    st.success("✅ Regressor loaded successfully from MLflow Registry")
except Exception as e:
    st.error(f"❌ Could not load regressor: {e}")

# --- Prepare input data ---
X = X.drop(columns=[
    'education', 'employment_type', 'emi_scenario', 'company_type',
    'house_type', 'marital_status', 'gender', 'emi_eligibility'
], errors='ignore')

X['salary_credit_interaction'] = X['monthly_salary'] * X['credit_score']
X['emi_gap'] = X['max_monthly_emi'] - X['current_emi_amount']
X['balance_emi_gap'] = X['bank_balance'] * X['emi_gap']

expected_bool_cols = [
    'education_High School', 'education_Post Graduate', 'education_Professional',
    'employment_type_private', 'employment_type_self-employed',
    'emi_scenario_Education Emi', 'emi_scenario_Home Appliances Emi',
    'emi_scenario_Personal Loan Emi', 'emi_scenario_Vehicle Emi',
    'company_type_MNC', 'company_type_Mid-size', 'company_type_Small', 'company_type_Startup',
    'house_type_Own', 'house_type_Rented',
    'marital_status_Single',
    'gender_FEMALE', 'gender_Female', 'gender_M', 'gender_MALE', 'gender_Male',
    'gender_female', 'gender_male'
]
for col in expected_bool_cols:
    if col not in X.columns:
        X[col] = False
X[expected_bool_cols] = X[expected_bool_cols].astype(bool)

# --- Align input schema manually ---
expected_input_cols = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure', 'max_monthly_emi',
    'total_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio',
    'emi_gap', 'credit_risk_score', 'employment_stability',
    'education_High School', 'education_Post Graduate', 'education_Professional',
    'employment_type_private', 'employment_type_self-employed',
    'emi_scenario_Education Emi', 'emi_scenario_Home Appliances Emi',
    'emi_scenario_Personal Loan Emi', 'emi_scenario_Vehicle Emi',
    'company_type_MNC', 'company_type_Mid-size', 'company_type_Small', 'company_type_Startup',
    'house_type_Own', 'house_type_Rented', 'marital_status_Single',
    'gender_FEMALE', 'gender_Female', 'gender_M', 'gender_MALE', 'gender_Male',
    'gender_female', 'gender_male', 'salary_credit_interaction', 'balance_emi_gap'
]
X = X.reindex(columns=expected_input_cols, fill_value=0)

# --- Metrics from MLflow ---
st.subheader("📊 Model Metrics from MLflow")
client = MlflowClient()

def show_metrics_for_run(run_id, label):
    try:
        run_data = client.get_run(run_id).data
        metrics = run_data.metrics

        st.markdown(f"### {label}")
        st.write("**Metrics:**", metrics if metrics else "No metrics logged")

        if metrics:
            fig, ax = plt.subplots()
            ax.bar(metrics.keys(), metrics.values()) #type: ignore
            ax.set_title(f"Logged Metrics ({label})")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Could not fetch metrics for {label}: {e}")

show_metrics_for_run("cfe97507737c468d96328c44d88ba10e", "Classifier")
show_metrics_for_run("c6efeb5a1e844a378d0b6051d1f52497", "Regressor")

# --- Prediction Visualizations ---
st.subheader("📈 Prediction Distributions")

if classifier is not None:
    try:
        preds = classifier.predict(X)
        fig, ax = plt.subplots()
        pd.Series(preds).astype(str).value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Classifier Prediction Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Could not generate classifier predictions: {e}")

if regressor is not None:
    try:
        preds = regressor.predict(X)
        fig2, ax2 = plt.subplots()
        pd.Series(preds).hist(bins=30, ax=ax2)
        ax2.set_title("Regressor Prediction Distribution")
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"❌ Could not generate regressor predictions: {e}")

# Footer
st.markdown("---")
st.caption("Model Explainability page | EMI Prediction App")