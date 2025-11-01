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
file_id = '143OiDzUfZOaZMpRuGygu93FTZqJCByD8'
url = f'https://drive.google.com/uc?id={file_id}&export=download'
output = 'data.csv'

gdown.download(url, output, quiet=False)

# 🧼 Load cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv(output)

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
st.subheader("1️⃣ EMI Eligibility Distribution")
st.bar_chart(df['emi_eligibility'].value_counts())

# 2️⃣ Financial Patterns
st.subheader("2️⃣ Average Financial Features by EMI Eligibility")
avg_financial = df.groupby('emi_eligibility')[['monthly_salary', 'bank_balance', 'credit_score']].mean()
st.dataframe(avg_financial)

# 3️⃣ Education vs EMI Eligibility
st.subheader("3️⃣ Education vs EMI Eligibility (Proportions)")
edu_emi = df.groupby('education')['emi_eligibility'].value_counts(normalize=True).unstack()
st.bar_chart(edu_emi)

# 4️⃣ Employment Type vs EMI Eligibility
st.subheader("4️⃣ Employment Type vs EMI Eligibility (Proportions)")
emp_emi = df.groupby('employment_type')['emi_eligibility'].value_counts(normalize=True).unstack()
st.bar_chart(emp_emi)

# 5️⃣ Salary Distribution (Boxplot)
st.subheader("5️⃣ Monthly Salary Distribution")
fig, ax = plt.subplots()
sns.boxplot(x='monthly_salary', data=df, ax=ax)
st.pyplot(fig)

# 6️⃣ Correlation Matrix
st.subheader("6️⃣ Correlation Matrix of Financial Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 7️⃣ Age Distribution by EMI Eligibility
st.subheader("7️⃣ Age Distribution by EMI Eligibility")
fig, ax = plt.subplots()
sns.boxplot(x='emi_eligibility', y='age', data=df, ax=ax)
st.pyplot(fig)

# 8️⃣ Education vs EMI Eligibility (Countplot)
st.subheader("8️⃣ Education vs EMI Eligibility (Countplot)")
fig, ax = plt.subplots()
sns.countplot(x='education', hue='emi_eligibility', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 9️⃣ Employment Type vs EMI Eligibility (Countplot)
st.subheader("9️⃣ Employment Type vs EMI Eligibility (Countplot)")
fig, ax = plt.subplots()
sns.countplot(x='employment_type', hue='emi_eligibility', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 🔟 Expense Breakdown
st.subheader("🔟 Average Monthly Expenses by EMI Eligibility")
expense_cols = ['school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
expense_df = df[expense_cols].groupby(df['emi_eligibility']).mean()
st.bar_chart(expense_df)

st.success("✅ EDA Completed Successfully!")
