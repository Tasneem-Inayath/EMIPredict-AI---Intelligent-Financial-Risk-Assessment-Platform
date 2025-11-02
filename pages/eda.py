# 📊 EDA - Exploratory Data Analysis Page

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# 🎨 Streamlit Page Config
st.set_page_config(page_title="EDA - EMI Prediction", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Gain insights into your dataset before model training.")

# 📥 Download dataset from Google Drive
file_id = '143OiDzUfZOaZMpRuGygu93FTZqJCByD8'
url = f'https://drive.google.com/uc?id={file_id}&export=download'
output = 'data.csv'
gdown.download(url, output, quiet=False)

# 🧼 Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv(output)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

df = load_data()

# ✅ Dataset Overview
st.subheader("📘 Dataset Overview")
st.dataframe(df.head())

# 🔍 Basic Info
st.subheader("🔍 Basic Information")
col1, col2 = st.columns(2)

with col1:
    st.write("**Shape:**", df.shape)
    st.write("**Data Types:**")
    st.write(df.dtypes)

with col2:
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

# 1️⃣ EMI Eligibility Distribution
if 'emi_eligibility' in df.columns:
    st.subheader("1️⃣ EMI Eligibility Distribution")
    st.bar_chart(df['emi_eligibility'].value_counts())
else:
    st.warning("Column 'emi_eligibility' not found.")

# 2️⃣ Financial Patterns
financial_cols = ['monthly_salary', 'bank_balance', 'credit_score']
if all(col in df.columns for col in financial_cols + ['emi_eligibility']):
    st.subheader("2️⃣ Average Financial Features by EMI Eligibility")
    avg_financial = df.groupby('emi_eligibility')[financial_cols].mean()
    st.dataframe(avg_financial)
else:
    st.warning("Missing financial columns for aggregation.")

# 3️⃣ Education vs EMI Eligibility
if 'education' in df.columns and 'emi_eligibility' in df.columns:
    st.subheader("3️⃣ Education vs EMI Eligibility (Proportions)")
    edu_emi = df.groupby('education')['emi_eligibility'].value_counts(normalize=True).unstack()
    st.bar_chart(edu_emi)
else:
    st.warning("Columns 'education' or 'emi_eligibility' not found.")

# 4️⃣ Employment Type vs EMI Eligibility
if 'employment_type' in df.columns and 'emi_eligibility' in df.columns:
    st.subheader("4️⃣ Employment Type vs EMI Eligibility (Proportions)")
    emp_emi = df.groupby('employment_type')['emi_eligibility'].value_counts(normalize=True).unstack()
    st.bar_chart(emp_emi)
else:
    st.warning("Columns 'employment_type' or 'emi_eligibility' not found.")

# 5️⃣ Salary Distribution (Boxplot)
if 'monthly_salary' in df.columns:
    st.subheader("5️⃣ Monthly Salary Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x='monthly_salary', data=df, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Column 'monthly_salary' not found.")

# 6️⃣ Correlation Matrix
st.subheader("6️⃣ Correlation Matrix of Financial Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 7️⃣ Age Distribution by EMI Eligibility
if 'age' in df.columns and 'emi_eligibility' in df.columns:
    st.subheader("7️⃣ Age Distribution by EMI Eligibility")
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_eligibility', y='age', data=df, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Columns 'age' or 'emi_eligibility' not found.")

# 8️⃣ Education vs EMI Eligibility (Countplot)
if 'education' in df.columns and 'emi_eligibility' in df.columns:
    st.subheader("8️⃣ Education vs EMI Eligibility (Countplot)")
    fig, ax = plt.subplots()
    sns.countplot(x='education', hue='emi_eligibility', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("Columns 'education' or 'emi_eligibility' not found.")

# 9️⃣ Employment Type vs EMI Eligibility (Countplot)
if 'employment_type' in df.columns and 'emi_eligibility' in df.columns:
    st.subheader("9️⃣ Employment Type vs EMI Eligibility (Countplot)")
    fig, ax = plt.subplots()
    sns.countplot(x='employment_type', hue='emi_eligibility', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("Columns 'employment_type' or 'emi_eligibility' not found.")

# 🔟 Expense Breakdown
expense_cols = ['school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
if all(col in df.columns for col in expense_cols + ['emi_eligibility']):
    st.subheader("🔟 Average Monthly Expenses by EMI Eligibility")
    expense_df = df[expense_cols].groupby(df['emi_eligibility']).mean()
    st.bar_chart(expense_df)
else:
    st.warning("Missing expense columns for breakdown.")

st.success("✅ EDA Completed Successfully!")