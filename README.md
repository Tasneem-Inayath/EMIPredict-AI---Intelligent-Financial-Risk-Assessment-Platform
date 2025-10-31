# 🚀 EMIPredict AI – Intelligent Financial Risk Assessment Platform

## 📘 Overview

**EMIPredict AI** is a robust, user-centric financial risk assessment platform built using **Machine Learning**, **Streamlit**, and **MLflow**. It predicts EMI (Equated Monthly Instalment) payment risk and empowers financial institutions to analyze borrowers’ repayment behavior with transparency and precision.

Designed with production-readiness in mind, the platform ensures schema-safe predictions, automated feature engineering, and clear output interpretation. It integrates MLflow for seamless experiment tracking and model lifecycle management, enabling reproducible and traceable deployments.

## 🧠 Key Features

- 📊 **Interactive Dashboard**  
  Built with Streamlit, the dashboard offers an intuitive interface for users to input borrower details and view risk predictions in real time.

- 🧩 **MLflow Integration**  
  All models are tracked via MLflow, with registry-based deployment from the Production stage to ensure reliability and traceability.

- 💡 **Supports Classification & Regression**  
  The platform handles both EMI risk classification and EMI amount regression, with modular pipelines for each.

- 🧹 **Automated Preprocessing & Feature Engineering**  
  Categorical encoding, type casting, and derived column calculations are handled internally to minimize user effort and prevent schema mismatches.

- 🔍 **Model Evaluation & Metrics Visualization**  
  Includes precision, recall, F1-score, ROC-AUC, and SHAP-based explainability for transparent decision-making.

- ☁️ **Cloud-Ready Architecture**  
  Designed for easy deployment on cloud platforms with minimal configuration.

## 💼 Skills Gained

- Python  
- Streamlit  
- Machine Learning (Classification + Regression)  
- Data Analysis  
- MLflow Experiment Tracking  
- Feature Engineering  
- Data Preprocessing  
- FinTech & Banking Domain Knowledge  

## 📁 Project Files

You can access all project datasets and artifacts here 👇

- 🔗 [EMIPredict AI Project Folder for Datasets](https://drive.google.com/drive/folders/1foIXvQO0Af8YTK3VLs2Evny9bpCnGWp4?usp=drive_link)  
- 🔗 [MLflow Folder](https://drive.google.com/drive/folders/1tNVJMXQgmC7yLVGk6Qt41YFOpwi_0P8k?usp=drive_link)  
- 🔗 [Pickle Files](https://drive.google.com/drive/folders/1YDUYd5ujrJI7ih2ngzyUP6_L6upre4On?usp=drive_link)  

## ⚙️ Tech Stack

| Category        | Tools/Frameworks         |
|----------------|--------------------------|
| Programming     | Python                   |
| Dashboard       | Streamlit                |
| ML Tracking     | MLflow                   |
| Data Handling   | Pandas, NumPy            |
| ML Models       | Scikit-learn             |
| Visualization   | Matplotlib, Seaborn      |

## 🧩 How to Run Locally

1️⃣ **Clone the repository**

```bash
git clone https://github.com/your-username/EMIPredict-AI.git
```

2️⃣ **Navigate to the folder**

```bash
cd EMIPredict-AI
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit app**

```bash
streamlit run app.py
```

## 🧾 MLflow Setup (Optional)

Start MLflow Tracking UI:

```bash
mlflow ui
```

Open the URL displayed in your terminal (usually http://127.0.0.1:5000).

## 🧠 Behind the Scenes
This project reflects a deployment-first mindset with strong MLflow integration:
- MLflow Model Registry: Models are versioned and deployed directly from the Production stage, ensuring traceability and reliability across environments.
- Experiment Tracking: Every training run is logged with parameters, metrics, and artifacts, making it easy to compare model performance and reproduce results.
- Artifact Logging: Preprocessing pipelines, trained models, and evaluation plots are stored as MLflow artifacts for seamless access and auditability.
- Modular Pipelines: Classification and regression workflows are modular, with MLflow tracking embedded at each stage—from data preprocessing to model evaluation.
- Streamlit Integration: The app dynamically loads models from the MLflow registry, eliminating the need for local pickle files and enabling cloud-ready deployment.
All preprocessing steps are internally managed to prevent schema mismatches:
- Categorical features are manually encoded for type safety.
- Derived columns (e.g., loan-to-income ratio) are calculated automatically to reduce user input burden.
- The app handles imbalanced classes and provides clear, interpretable outputs for financial decision-makers.

## 👩‍💻 Author

**Tasneem Firdhosh**  
🎓 MCA 
📫 Contact: tasneemfirdhosh@gmail.com  

## 🌟 Acknowledgments

Special thanks to mentors and online communities that helped shape this project.

---

