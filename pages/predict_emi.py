import streamlit as st
import pandas as pd
import mlflow.pyfunc

st.set_page_config(page_title="EMI Predictor", page_icon="📊", layout="centered")
st.title("📊 EMI Eligibility & Max EMI Prediction")

with st.form("emi_form"):
    st.subheader("Applicant Details")

    # Raw numeric inputs
    age = st.number_input("Age", 18, 75, 30)
    monthly_salary = st.number_input("Monthly Salary (₹)", 0, 500000, 50000)
    years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
    monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
    family_size = st.selectbox("Family Size", [1, 2, 3, 4, 5])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
    existing_loans = st.selectbox("Existing Loans", [0, 1])

    school_fees = st.number_input("School Fees (₹)", 0, 100000, 0)
    college_fees = st.number_input("College Fees (₹)", 0, 100000, 0)
    travel_expenses = st.number_input("Travel Expenses (₹)", 0, 100000, 0)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 100000, 0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 100000, 0)

    current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 100000, 0)
    credit_score = st.number_input("Credit Score", 300, 850, 700)
    bank_balance = st.number_input("Bank Balance (₹)", 0, 1000000, 100000)
    emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 50000)
    requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 1000000, 250000)
    requested_tenure = st.slider("Requested Tenure (months)", 6, 60, 24)
    max_monthly_emi = st.number_input("Max Monthly EMI (₹)", 0, 100000, 20000)

    # Categorical inputs
    education = st.selectbox("Education", ['High School', 'Post Graduate', 'Professional'])
    employment_type = st.selectbox("Employment Type", ['private', 'self-employed'])
    emi_scenario = st.selectbox("EMI Scenario", ['Education Emi', 'Home Appliances Emi', 'Personal Loan Emi', 'Vehicle Emi'])
    company_type = st.selectbox("Company Type", ['MNC', 'Mid-size', 'Small', 'Startup'])
    house_type = st.selectbox("House Type", ['Own', 'Rented'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married'])
    gender = st.selectbox("Gender", ['Male', 'Female'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # --- Raw input dictionary ---
    input_dict = {
        'age': age,
        'monthly_salary': monthly_salary,
        'years_of_employment': years_of_employment,
        'monthly_rent': monthly_rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'emergency_fund': emergency_fund,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
        'max_monthly_emi': max_monthly_emi,
        'education': education,
        'employment_type': employment_type,
        'emi_scenario': emi_scenario,
        'company_type': company_type,
        'house_type': house_type,
        'marital_status': marital_status,
        'gender': gender
    }

    df = pd.DataFrame([input_dict])

    # --- Derived features ---
    df['total_expenses'] = df[['school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']].sum(axis=1)
    df['debt_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary'].replace(0, 1)
    df['expense_to_income_ratio'] = df['total_expenses'] / df['monthly_salary'].replace(0, 1)
    df['emi_gap'] = df['max_monthly_emi'] - df['current_emi_amount']
    df['credit_risk_score'] = 850 - df['credit_score']
    df['employment_stability'] = df['years_of_employment'] * df['existing_loans']
    df['salary_credit_interaction'] = df['monthly_salary'] * df['credit_score']
    df['balance_emi_gap'] = df['bank_balance'] * df['emi_gap']
    

    # --- One-hot encoding (drop_first=True) ---
    df_encoded = pd.get_dummies(df, columns=[
        'education', 'employment_type', 'emi_scenario', 'company_type',
        'house_type', 'marital_status', 'gender'
    ], drop_first=True)
    # --- Ensure all expected encoded columns exist ---
    # --- Ensure all expected encoded columns exist ---
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
        if col not in df_encoded.columns:
            df_encoded[col] = False
    df_encoded[expected_bool_cols] = df_encoded[expected_bool_cols].astype(bool)
    # --- Type casting for numeric columns ---
    numeric_cols = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
        'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
        'other_monthly_expenses', 'current_emi_amount', 'credit_score',
        'bank_balance', 'emergency_fund', 'requested_amount', 'max_monthly_emi',
        'total_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio',
        'emi_gap', 'credit_risk_score', 'employment_stability',
        'salary_credit_interaction', 'balance_emi_gap'
    ]
    int_cols = ['family_size', 'dependents', 'existing_loans', 'requested_tenure']

    df_encoded[numeric_cols] = df_encoded[numeric_cols].astype(float)
    df_encoded[int_cols] = df_encoded[int_cols].astype(int)
    # --- Tenure-based EMI estimate (for user insight only) ---
    annual_interest_rate = 0.10  # 10% assumed interest
    monthly_rate = annual_interest_rate / 12
    P = df['requested_amount']
    N = df['requested_tenure']
    R = monthly_rate

    # EMI formula
    df['tenure_based_emi'] = (P * R * (1 + R) ** N) / ((1 + R) ** N - 1)

    # --- Load models ---
    classifier = mlflow.pyfunc.load_model("models:/EMI_Classifier_XGBoost/Production")
    regressor = mlflow.pyfunc.load_model("models:/EMI_Regressor_XGBoost/Production")

    # --- Predict ---
    eligibility = classifier.predict(df_encoded)[0]
    max_emi = regressor.predict(df_encoded)[0]

    label_map = {
        'Eligible': '✅ Eligible',
        'High_Risk': '⚠️ High Risk',
        'Not_Eligible': '❌ Not Eligible',
        0: '✅ Eligible',
        1: '⚠️ High Risk',
        2: '❌ Not Eligible'
    }

    st.success(f"**Predicted Eligibility:** {label_map.get(eligibility, str(eligibility))}")
    st.info(f"**Predicted Max EMI:** ₹{max_emi:,.2f}")
    st.info(f"**Estimated EMI for Requested Loan:** ₹{df['tenure_based_emi'].iloc[0]:,.2f}")