import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Page config
st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability & Insights")

st.markdown("""
This page shows **metrics and prediction behavior** for the models currently in Production.  
Metrics are pulled from the **Production model version** in MLflow (both version-level and run-level).
""")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("datasets/encoded_data.csv")

df = load_data()
X = df.drop(columns=['emi_eligibility'], errors='ignore')

# --- Load models from MLflow Registry (PyFunc flavor) ---
classifier, regressor = None, None
try:
    classifier = mlflow.pyfunc.load_model("models:/EMI_Classifier_XGBoost/Production")
    st.success("✅ Classifier loaded successfully from MLflow Registry")
except Exception as e:
    st.error(f"❌ Could not load classifier: {e}")

try:
    regressor = mlflow.pyfunc.load_model("models:/EMI_Regressor_XGBoost/Production")
    st.success("✅ Regressor loaded successfully from MLflow Registry")
except Exception as e:
    st.error(f"❌ Could not load regressor: {e}")

# --- Metrics from MLflow ---
st.subheader("📊 Model Metrics from MLflow")

client = MlflowClient()

def show_metrics_for_model(model_name, label):
    try:
        # Find the Production version
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if not prod_versions:
            st.warning(f"No Production version found for {label}")
            return

        prod_version = prod_versions[0]
        run_id = prod_version.run_id

        # Collect metrics from both model version and run
        version_metrics = {m.key: m.value for m in getattr(prod_version, "metrics", [])}
        run_data = client.get_run(run_id).data #type: ignore
        run_metrics = run_data.metrics
        params = run_data.params

        # Merge metrics (version-level takes priority if duplicate keys)
        metrics = {**run_metrics, **version_metrics}

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

# Show metrics for both models
show_metrics_for_model("EMI_Classifier_XGBoost", "Classifier")
show_metrics_for_model("EMI_Regressor_XGBoost", "Regressor")

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