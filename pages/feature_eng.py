import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

st.title("⚙️ Feature Engineering")

st.markdown("""
This page highlights the **engineered features** already included in the dataset.  
These features are derived from raw applicant data to improve model performance and interpretability.
""")

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/cleaned_emi_data.csv")  # already includes derived features
    return df

df = load_data()

# --- Derived Features Dictionary ---
derived_features = {
    "total_expenses": "Sum of school fees, college fees, travel, groceries, and other expenses",
    "debt_to_income_ratio": "Current EMI ÷ Monthly Salary",
    "expense_to_income_ratio": "Total Expenses ÷ Monthly Salary",
    "emi_gap": "Max Monthly EMI – Current EMI",
    "credit_risk_score": "Inverse of credit score (850 – credit_score)",
    "employment_stability": "Years of employment × Existing loans",
    "salary_credit_interaction": "Monthly Salary × Credit Score",
    "balance_emi_gap": "Bank Balance × EMI Gap"
}

# --- Backfill missing engineered features if needed ---
if 'salary_credit_interaction' not in df.columns and {'monthly_salary','credit_score'}.issubset(df.columns):
    df['salary_credit_interaction'] = df['monthly_salary'] * df['credit_score']

if 'balance_emi_gap' not in df.columns and {'bank_balance','emi_gap'}.issubset(df.columns):
    df['balance_emi_gap'] = df['bank_balance'] * df['emi_gap']

st.subheader("📘 Engineered Features Explained")
st.table(pd.DataFrame(list(derived_features.items()), columns=["Feature", "Description"]))

# --- Show sample of engineered features ---
st.subheader("📊 Sample of Engineered Features in Dataset")
available_cols = [col for col in derived_features.keys() if col in df.columns]

if available_cols:
    st.dataframe(df[available_cols].head(10), use_container_width=True)
else:
    st.warning("No engineered features found in the dataset.")

# --- Visualize one engineered feature vs eligibility ---
if "emi_eligibility" in df.columns and available_cols:
    st.subheader("📈 Engineered Feature vs EMI Eligibility")
    selected_eng = st.selectbox("Select a derived feature:", available_cols)

    fig, ax = plt.subplots()
    sns.boxplot(x="emi_eligibility", y=selected_eng, data=df, palette="Set2", ax=ax)
    ax.set_title(f"{selected_eng} by EMI Eligibility")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Feature Engineering page | EMI Prediction App")