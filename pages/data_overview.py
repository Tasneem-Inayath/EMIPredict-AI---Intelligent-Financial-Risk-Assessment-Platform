import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview")

st.markdown("""
This page provides a quick overview of the EMI dataset:
- Dataset shape and column details
- Summary statistics
- Class balance for **EMI Eligibility**
""")

# --- Load your dataset ---
# Replace with your actual dataset path or data loading function
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/cleaned_emi_data.csv")  # adjust path
    return df

df = load_data()

# --- Dataset Shape ---
st.subheader("📐 Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# --- Column Info ---
st.subheader("🧾 Column Information")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})
st.dataframe(col_info, use_container_width=True)

# --- Summary Statistics ---
st.subheader("📊 Summary Statistics")
st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# --- Class Balance ---
if "emi_eligibility" in df.columns:
    st.subheader("⚖️ EMI Eligibility Distribution")
    class_counts = df["emi_eligibility"].value_counts()

    fig, ax = plt.subplots()
    class_counts.plot(kind="bar", color=["green", "orange", "red"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("EMI Eligibility Class Balance")
    st.pyplot(fig)

    st.write(class_counts)

# Footer
st.markdown("---")
st.caption("Data Overview page | EMI Prediction App")