# Customer Churn Intelligence Platform

An end-to-end machine learning application that predicts customer churn risk, explains important prediction factors using SHAP, and provides actionable customer retention recommendations.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers at risk of leaving can help businesses prioritize customer engagement and retention efforts.

This project uses the IBM Telco Customer Churn dataset to build a machine learning pipeline and deploy it through a FastAPI backend and an interactive Streamlit dashboard.

**The system provides predictions and decision support; it does not guarantee that a customer will churn.**

## Key Features

* **Churn Prediction:** Predicts churn probability using a tuned Random Forest classifier.
* **Model Evaluation:** Compares multiple classification algorithms using accuracy, precision, recall, F1-score, and ROC-AUC.
* **Hyperparameter Tuning:** Uses GridSearchCV with stratified cross-validation.
* **Explainable AI:** Uses SHAP to identify factors associated with higher or lower predicted churn risk.
* **Risk Segmentation:** Classifies customers into Low, Medium, High, and Critical risk levels.
* **Retention Recommendations:** Generates suggested retention actions based on customer attributes and predicted risk.
* **REST API:** FastAPI endpoints for health checks, individual predictions, and batch CSV predictions.
* **Interactive Dashboard:** Streamlit interface for customer predictions and batch processing.
* **Automated Tests:** Tests API health, prediction responses, and invalid input handling.

## Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Explainability:** SHAP
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Testing:** Pytest, HTTPX
* **Data Visualization:** Matplotlib

## Dataset

The project uses the IBM Telco Customer Churn dataset.

* **Records:** 7,043 customers
* **Original columns:** 21
* **Selected model features:** 10
* **Target:** `Churn`

The selected input features are:

1. `SeniorCitizen`
2. `Dependents`
3. `tenure`
4. `InternetService`
5. `OnlineSecurity`
6. `TechSupport`
7. `Contract`
8. `PaymentMethod`
9. `MonthlyCharges`
10. `TotalCharges`

The target is encoded as `No = 0` and `Yes = 1`. The dataset is split into training and testing sets using stratified sampling.

## Model Performance

The selected Random Forest model was tuned using GridSearchCV with F1-score as the selection metric.

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |      75.51% |
| Precision |      52.70% |
| Recall    |      75.67% |
| F1-score  |      62.13% |
| ROC-AUC   |      84.03% |

The model emphasizes churn recall, helping identify a larger share of customers who actually churned in the test set. This also produces false positives, so predictions should support—not replace—business judgment.

## Explainability with SHAP

SHAP helps explain how model features influence individual predictions.

Important features in the global explanation included:

* Contract type
* Customer tenure
* Internet service type
* Online security
* Technical support
* Total charges
* Payment method

These are model associations, not proof of causation.

## Application Architecture

```text
Telco Customer Dataset
          |
          v
Data Cleaning and EDA
          |
          v
Feature Selection
          |
          v
Preprocessing Pipeline
          |
          v
Random Forest Model
          |
          v
FastAPI Prediction Service
          |
          v
Streamlit Dashboard
          |
          +--> Churn Probability
          +--> Risk Classification
          +--> SHAP Explanations
          +--> Retention Recommendations
          +--> Batch CSV Predictions
```

## Project Structure

```text
customer_churn_project/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model1.pkl
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── eda_feature_model.ipynb
│   └── modeling.ipynb
├── src/
│   ├── explain.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── recommendations.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### 5. Start the Streamlit dashboard

Open a second terminal, activate the same virtual environment, navigate to the project root, and run:

```bash
streamlit run dashboard/app.py
```

Dashboard:

http://localhost:8501

Keep the FastAPI server running while using the dashboard.

### 6. Run automated tests

```bash
python -m pytest tests/test_api.py -v
```

## API Endpoints

| Method | Endpoint         | Purpose                                   |
| ------ | ---------------- | ----------------------------------------- |
| GET    | `/health`        | Check API and model status                |
| POST   | `/predict`       | Predict churn for one customer            |
| POST   | `/batch-predict` | Predict churn for customers in a CSV file |

See `/docs` for request schemas and interactive API testing.

## Business Impact

The platform demonstrates how machine learning can support customer retention by:

* Identifying customers who may need proactive engagement.
* Highlighting factors associated with churn risk.
* Suggesting relevant retention actions.
* Supporting analysis of multiple customers through batch predictions.

The potential business impact depends on model performance, retention campaign effectiveness, and operational implementation.

# Customer Churn Intelligence Platform

An end-to-end machine learning application that predicts customer churn risk, explains important prediction factors using SHAP, and provides actionable customer retention recommendations.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers at risk of leaving can help businesses prioritize customer engagement and retention efforts.

This project uses the IBM Telco Customer Churn dataset to build a machine learning pipeline and deploy it through a FastAPI backend and an interactive Streamlit dashboard.

**The system provides predictions and decision support; it does not guarantee that a customer will churn.**

## Key Features

* **Churn Prediction:** Predicts churn probability using a tuned Random Forest classifier.
* **Model Evaluation:** Compares multiple classification algorithms using accuracy, precision, recall, F1-score, and ROC-AUC.
* **Hyperparameter Tuning:** Uses GridSearchCV with stratified cross-validation.
* **Explainable AI:** Uses SHAP to identify factors associated with higher or lower predicted churn risk.
* **Risk Segmentation:** Classifies customers into Low, Medium, High, and Critical risk levels.
* **Retention Recommendations:** Generates suggested retention actions based on customer attributes and predicted risk.
* **REST API:** FastAPI endpoints for health checks, individual predictions, and batch CSV predictions.
* **Interactive Dashboard:** Streamlit interface for customer predictions and batch processing.
* **Automated Tests:** Tests API health, prediction responses, and invalid input handling.

## Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Explainability:** SHAP
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Testing:** Pytest, HTTPX
* **Data Visualization:** Matplotlib

## Dataset

The project uses the IBM Telco Customer Churn dataset.

* **Records:** 7,043 customers
* **Original columns:** 21
* **Selected model features:** 10
* **Target:** `Churn`

The selected input features are:

1. `SeniorCitizen`
2. `Dependents`
3. `tenure`
4. `InternetService`
5. `OnlineSecurity`
6. `TechSupport`
7. `Contract`
8. `PaymentMethod`
9. `MonthlyCharges`
10. `TotalCharges`

The target is encoded as `No = 0` and `Yes = 1`. The dataset is split into training and testing sets using stratified sampling.

## Model Performance

The selected Random Forest model was tuned using GridSearchCV with F1-score as the selection metric.

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |      75.51% |
| Precision |      52.70% |
| Recall    |      75.67% |
| F1-score  |      62.13% |
| ROC-AUC   |      84.03% |

The model emphasizes churn recall, helping identify a larger share of customers who actually churned in the test set. This also produces false positives, so predictions should support—not replace—business judgment.

## Explainability with SHAP

SHAP helps explain how model features influence individual predictions.

Important features in the global explanation included:

* Contract type
* Customer tenure
* Internet service type
* Online security
* Technical support
* Total charges
* Payment method

These are model associations, not proof of causation.

## Application Architecture

```text
Telco Customer Dataset
          |
          v
Data Cleaning and EDA
          |
          v
Feature Selection
          |
          v
Preprocessing Pipeline
          |
          v
Random Forest Model
          |
          v
FastAPI Prediction Service
          |
          v
Streamlit Dashboard
          |
          +--> Churn Probability
          +--> Risk Classification
          +--> SHAP Explanations
          +--> Retention Recommendations
          +--> Batch CSV Predictions
```

## Project Structure

```text
customer_churn_project/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model1.pkl
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── eda_feature_model.ipynb
│   └── modeling.ipynb
├── src/
│   ├── explain.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── recommendations.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

### 5. Start the Streamlit dashboard

Open a second terminal, activate the same virtual environment, navigate to the project root, and run:

```bash
streamlit run dashboard/app.py
```

Dashboard:

http://localhost:8501

Keep the FastAPI server running while using the dashboard.

### 6. Run automated tests

```bash
python -m pytest tests/test_api.py -v
```

## API Endpoints

| Method | Endpoint         | Purpose                                   |
| ------ | ---------------- | ----------------------------------------- |
| GET    | `/health`        | Check API and model status                |
| POST   | `/predict`       | Predict churn for one customer            |
| POST   | `/batch-predict` | Predict churn for customers in a CSV file |

See `/docs` for request schemas and interactive API testing.

## Business Impact

The platform demonstrates how machine learning can support customer retention by:

* Identifying customers who may need proactive engagement.
* Highlighting factors associated with churn risk.
* Suggesting relevant retention actions.
* Supporting analysis of multiple customers through batch predictions.

The potential business impact depends on model performance, retention campaign effectiveness, and operational implementation.

## Future Improvements

* Data drift and prediction monitoring.
* Additional automated tests and input validation.
* Docker-based deployment.
* Cloud deployment and production logging.
* Monitoring retention campaign outcomes.


