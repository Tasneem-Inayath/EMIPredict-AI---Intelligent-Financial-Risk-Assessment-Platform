import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Model Training & Comparison", page_icon="🤖", layout="wide")

st.title("🤖 Model Training & Comparison")

st.markdown("""
This page summarizes the performance of different machine learning models 
for **EMI eligibility classification** and **maximum EMI regression**.
""")

# --- Classification Results ---
st.subheader("📊 Classification Models (Eligibility Prediction)")

classification_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Decision Tree",
        "HistGradientBoosting"
    ],
    "Accuracy": [0.83, 0.95, 0.98, 0.98, 0.96],
    "Macro F1": [0.67, 0.66, 0.93, 0.89, 0.74],
    "ROC-AUC": [0.943, 0.981, 0.998, 0.938, 0.993]
})

st.dataframe(classification_results, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=classification_results.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model", y="Score", hue="Metric", ax=ax
)
ax.set_title("Classification Model Performance")
plt.xticks(rotation=20)
st.pyplot(fig)

# --- Regression Results ---
st.subheader("📈 Regression Models (Max EMI Prediction)")

regression_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "XGBoost Regressor",
        "Decision Tree Regressor",
        "HistGradientBoosting Regressor"
    ],
    "RMSE": [0.0, 264.0, 350.0, 565.0],
    "MAE": [0.0, 142.0, 110.0, 358.0],
    "R² Score": [1.000, 0.999, 0.998, 0.994],
    "MAPE (%)": [0.0, 7.18, 4.19, 25.99]
})

st.dataframe(regression_results, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=regression_results.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model", y="Score", hue="Metric", ax=ax2
)
ax2.set_title("Regression Model Performance")
plt.xticks(rotation=20)
st.pyplot(fig2)

# --- Final Model Selection ---
st.subheader("✅ Final Model Selection")
st.markdown("""
- **Eligibility Classification** → **XGBoost Classifier**  
  (Best accuracy, F1, and ROC-AUC; handles minority *High_Risk* class better)

- **Max EMI Regression** → **XGBoost Regressor**  
  (Lowest RMSE, very high R², and reasonable MAPE)

⚠️ Note: Linear Regression showed a "perfect" score due to likely data leakage, so it is not considered.
""")

# Footer
st.markdown("---")
st.caption("Model Training & Comparison page | EMI Prediction App")