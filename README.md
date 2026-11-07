# 📊 Telco Customer Churn Prediction Dashboard

An interactive Machine Learning application that predicts customer churn in the telecommunications industry. Built with Python and Streamlit, the dashboard combines three classification models with exploratory data analysis, model evaluation, customer-level predictions, and data drift analysis.

<p align="center">
  <a href="https://psebtasktelcocustomerchurnpredictionapp-mhnb66quufw4nyhngar7vf.streamlit.app/">
    <strong>🚀 Launch Live Application</strong>
  </a>
</p>

---

## 🌐 Live Demo

**Try the deployed application:**  
[Telco Customer Churn Prediction Dashboard](https://psebtasktelcocustomerchurnpredictionapp-mhnb66quufw4nyhngar7vf.streamlit.app/)

Explore customer churn patterns, test individual machine learning models, generate predictions, and compare model performance through an interactive dashboard.

---

## 📌 Project Overview

Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to discontinue their services can help businesses understand customer behavior and develop better retention strategies.

This project uses historical telecommunications customer data to build and evaluate machine learning classification models that predict whether a customer is likely to leave the service.

The application provides an interactive interface for exploring the dataset, testing trained models, understanding their performance, and examining potential changes in feature distributions.

## ✨ Key Features

### 1. Dashboard Overview
- Total customer statistics.
- Churned and retained customer counts.
- Overall churn percentage.
- Interactive visualizations of churn distribution.
- Customer segmentation by contract and internet service.
- Customer tenure analysis.

### 2. Dataset Explorer
- Interactive dataset preview.
- Dataset dimensions and column information.
- Missing-value analysis.
- Duplicate record detection.
- Numerical and categorical feature exploration.
- Interactive charts for customer segmentation.
- Dataset CSV download functionality.

### 3. Customer Churn Prediction
- Predict churn for individual customers.
- Select between three trained ML models.
- Interactive customer input form.
- Automatic categorical feature encoding.
- Churn and retention probability visualization.
- Interactive churn-risk gauge.

### 4. Individual Model Analysis
Evaluate each classifier independently using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Feature importance and model coefficients

### 5. Model Comparison
- Compare all three models.
- View repeated stratified 5-fold cross-validation results.
- Compare classification performance metrics.
- Identify the strongest-performing baseline model.

### 6. Data Drift Monitoring
- Compare training and test feature distributions.
- Population Stability Index (PSI).
- Kolmogorov-Smirnov (KS) test for numerical features.
- Chi-square test for binary features.
- Feature-level drift reporting.
- Interactive visualization of features with the highest PSI values.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble of decision trees |

The models use preprocessing pipelines to handle numerical features and prepare data for classification.

## 📈 Model Performance

The following results represent the mean scores obtained through Repeated Stratified 5-Fold Cross-Validation.

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 0.8029 | 0.7262 | 0.7886 |
| Precision | 0.6516 | 0.4834 | 0.6314 |
| Recall | 0.5484 | 0.5051 | 0.4815 |
| F1-Score | **0.5954** | 0.4920 | 0.5474 |
| ROC-AUC | **0.8455** | 0.6553 | 0.8234 |

### Best Performing Baseline

**Logistic Regression** achieved the highest mean F1-score and ROC-AUC among the evaluated models.

These results make it the strongest baseline model in this experiment. The results are evaluation metrics, not a guarantee of future prediction accuracy.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| Joblib | Model serialization and loading |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| SciPy | Statistical drift tests |
| Streamlit | Interactive web application |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Application deployment |

---

## 📂 Repository Structure

```text
PSEB_TASK_Telco_Customer_Churn_Prediction_App/
│
├── artifacts/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   └── model_metadata.joblib
│
├── drift_reports/
│
├── streamlit_app.py
├── notebook.ipynb
├── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

### File Description

- `streamlit_app.py` — Main Streamlit application and dashboard.
- `notebook.ipynb` — Data preprocessing, model development, and experimentation.
- `Telco-Customer-Churn.csv` — Customer churn dataset.
- `artifacts/` — Serialized trained machine learning models and metadata.
- `drift_reports/` — Directory for drift-related reports.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

---

## ⚙️ Installation and Setup

Follow these steps to run the application locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Khubaib-Muhammad/PSEB_TASK_Telco_Customer_Churn_Prediction_App.git
```

### 2. Navigate to the Project Directory

```bash
cd PSEB_TASK_Telco_Customer_Churn_Prediction_App
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

For reproducibility, use Python 3.12 and the compatible Scikit-learn version specified in the project requirements.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 📊 Dataset Information

The project uses a telecommunications customer churn dataset containing customer demographics, account information, subscribed services, billing details, and churn status.

**Important features include:**

- Customer demographics: Gender, Senior Citizen, Partner, Dependents.
- Account information: Tenure, Contract, Payment Method.
- Service information: Internet Service, Phone Service, Online Security, Tech Support, and other subscribed services.
- Billing information: Monthly Charges and Total Charges.
- Target variable: Churn.

The target variable indicates whether a customer has discontinued the service.

## 🔍 Data Preprocessing

The preprocessing workflow includes:

- Handling missing and inconsistent values.
- Converting `TotalCharges` into a numerical feature.
- Encoding binary categorical variables.
- One-hot encoding multi-category features.
- Removing the non-predictive customer identifier.
- Aligning prediction inputs with the trained model's feature columns.
- Preparing data for model training and evaluation.

## 📉 Model Evaluation and Drift Analysis

The project uses multiple classification metrics to assess model performance rather than relying solely on accuracy.

The drift-monitoring module compares feature distributions between training and test subsets using statistical tests and PSI.

**Note:** The current drift-monitoring implementation is an illustrative train/test distribution comparison. It does not represent continuous monitoring of newly arriving production customer data.

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

Live application:

[Open Telco Customer Churn Dashboard](https://psebtasktelcocustomerchurnpredictionapp-mhnb66quufw4nyhngar7vf.streamlit.app/)

---

## 🔮 Future Improvements

Potential enhancements include:

- Integration of newly collected customer data.
- Continuous production data drift monitoring.
- Model retraining and automated evaluation.
- Advanced explainability using SHAP.
- Improved churn-risk segmentation.
- Customer retention recommendation strategies.
- Additional machine learning and ensemble models.

---

## ⚠️ Disclaimer

This project is developed for educational, analytical, and demonstration purposes. Predictions are generated using trained machine learning models and should not be treated as guaranteed outcomes or as a substitute for business validation.

---

## 👨‍💻 Author

**Khubaib Muhammad**

GitHub: [@Khubaib-Muhammad](https://github.com/Khubaib-Muhammad)

---

<p align="center">
  <strong>Built with Python, Machine Learning, and Streamlit.</strong>
</p>
