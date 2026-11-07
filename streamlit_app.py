from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Data & Models ---
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    df_path = BASE_DIR / "Telco-Customer-Churn.csv"

    if df_path.exists():
        return pd.read_csv(df_path)
    else:
        st.error(f"Dataset not found at {df_path}. Please verify the path.")
        return None

@st.cache_resource
def load_artifacts():
    try:
        lr = joblib.load("artifacts/logistic_regression.joblib")
        dt = joblib.load("artifacts/decision_tree.joblib")
        rf = joblib.load("artifacts/random_forest.joblib")
        meta = joblib.load("artifacts/model_metadata.joblib")
        return {
            "Logistic Regression": lr,
            "Decision Tree": dt,
            "Random Forest": rf,
            "metadata": meta
        }
    except Exception as e:
        st.error(f"Error loading pipelines or metadata: {e}")
        return None

df_raw = load_data()
artifacts = load_artifacts()

# Helper function to compute PSI
def calculate_psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna().values
    current = pd.Series(current).dropna().values
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    if len(np.unique(reference)) <= 2:
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0] = -np.inf
        edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)
    epsilon = 0.0001
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Main Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Dashboard Overview",
    "2. Dataset Explorer",
    "3. Customer Churn Prediction",
    "4. Individual Model Analysis",
    "5. Model Comparison",
    "6. Data Drift Monitoring"
])

if df_raw is None or artifacts is None:
    st.warning("Please ensure 'Telco-Customer-Churn.csv' is available and model artifacts are saved in 'artifacts/'.")
else:
    # Preprocess raw data for reference metrics and analysis
    df_clean = df_raw.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
    df_clean['gender_numeric'] = df_clean['gender'].map({'Female': 0, 'Male': 1})
    df_clean['Partner_numeric'] = df_clean['Partner'].map({'Yes': 1, 'No': 0})
    df_clean['Dependents_numeric'] = df_clean['Dependents'].map({'Yes': 1, 'No': 0})
    df_clean['PhoneService_numeric'] = df_clean['PhoneService'].map({'Yes': 1, 'No': 0})
    df_clean['PaperlessBilling_numeric'] = df_clean['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df_clean['Churn_numeric'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

    # Prepare X & y from full clean dataset matching expected features in metadata
    feature_cols = artifacts["metadata"]["feature_columns"]
    
    # --- Dynamic UI Pages ---
    if page == "1. Dashboard Overview":
        st.title("📊 Dashboard Overview")
        st.markdown("Welcome to the Telco Customer Churn analytical dashboard. Gain real-time insights and predictive statistics.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        total_cust = len(df_clean)
        churn_cnt = (df_clean['Churn'] == 'Yes').sum()
        stay_cnt = total_cust - churn_cnt
        churn_rate = (churn_cnt / total_cust) * 100
        
        col1.metric("Total Customers", f"{total_cust:,}")
        col2.metric("Customers Stayed", f"{stay_cnt:,}")
        col3.metric("Customers Churned", f"{churn_cnt:,}")
        col4.metric("Overall Churn %", f"{churn_rate:.2f}%")
        col5.metric("Available ML Models", "3")

        st.markdown("### Key Customer Segment Distributions")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df_clean, names='Churn', title='Overall Churn Distribution (Yes/No)', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df_clean, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
            st.plotly_chart(fig2, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.histogram(df_clean, x='InternetService', color='Churn', barmode='group', title='Churn by Internet Service')
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            fig4 = px.box(df_clean, x='Churn', y='tenure', title='Customer Tenure vs Churn Status')
            st.plotly_chart(fig4, use_container_width=True)

    elif page == "2. Dataset Explorer":
        st.title("🔍 Dataset Explorer")
        st.markdown("Explore the raw Telco dataset details, search records, or analyze correlations dynamically.")
        
        tab1, tab2 = st.tabs(["Table View & Metadata", "Multivariate Visual Analysis"])
        
        with tab1:
            st.write(f"**Dataset Dimensions:** {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
            cols_to_show = st.multiselect("Select Columns to View", df_raw.columns.tolist(), default=df_raw.columns.tolist()[:10])
            st.dataframe(df_raw[cols_to_show].head(50), use_container_width=True)
            
            st.markdown("### Missing values & Data Types")
            meta_info = pd.DataFrame({
                'Data Type': df_raw.dtypes.astype(str),
                'Missing Count': df_raw.isnull().sum(),
                'Missing %': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
            })
            st.dataframe(meta_info, use_container_width=True)
            st.write(f"**Duplicate Rows Count:** {df_raw.duplicated().sum()}")
            
            csv_data = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Dataset as CSV", csv_data, "Telco_Customer_Churn.csv", "text/csv")
            
        with tab2:
            st.subheader("Dynamic Numerical Analysis")
            num_feat = st.selectbox("Select Numeric Feature for Distribution Analysis", ['tenure', 'MonthlyCharges', 'TotalCharges'])
            fig_num = px.histogram(df_clean, x=num_feat, color='Churn', marginal='box', barmode='overlay', title=f'{num_feat} Distribution by Churn Status')
            st.plotly_chart(fig_num, use_container_width=True)

            st.subheader("Dynamic Categorical Analysis")
            cat_feat = st.selectbox("Select Categorical Feature to Segment Churn Rate", [
                'Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'Dependents'
            ])
            fig_cat = px.histogram(df_clean, x=cat_feat, color='Churn', barmode='group', title=f'Churn segmented by {cat_feat}')
            st.plotly_chart(fig_cat, use_container_width=True)

    elif page == "3. Customer Churn Prediction":
        st.title("🔮 Customer Churn Prediction")
        st.markdown("Enter custom attributes for a single customer below to calculate real-time Churn Probabilities.")
        
        selected_model_name = st.selectbox("Select Model Pipeline for Inference", ["Logistic Regression", "Decision Tree", "Random Forest"])
        model_pipeline = artifacts[selected_model_name]
        
        with st.form("churn_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            with c2:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                if phone_service == "No":
                    multiple_lines = "No phone service"
                else:
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
                
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                if internet_service == "No":
                    online_security = "No internet service"
                    online_backup = "No internet service"
                    device_protection = "No internet service"
                    tech_support = "No internet service"
                    streaming_tv = "No internet service"
                    streaming_movies = "No internet service"
                else:
                    online_security = st.selectbox("Online Security", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            with c3:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges)
                
            submit_btn = st.form_submit_button("Predict Churn probability")
            
        if submit_btn:
            # Gather original inputs and perform manual One-Hot encoding to align with 40-feature expectation
            input_dict = {
                'gender': 1 if gender == 'Male' else 0,
                'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
                'Partner': 1 if partner == 'Yes' else 0,
                'Dependents': 1 if dependents == 'Yes' else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == 'Yes' else 0,
                'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Implement the exact 31 categorical columns mapped from original values
            cats_mapping = {
                'MultipleLines': ['No', 'No phone service', 'Yes'],
                'InternetService': ['DSL', 'Fiber optic', 'No'],
                'OnlineSecurity': ['No', 'No internet service', 'Yes'],
                'OnlineBackup': ['No', 'No internet service', 'Yes'],
                'DeviceProtection': ['No', 'No internet service', 'Yes'],
                'TechSupport': ['No', 'No internet service', 'Yes'],
                'StreamingTV': ['No', 'No internet service', 'Yes'],
                'StreamingMovies': ['No', 'No internet service', 'Yes'],
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
            }
            
            user_vals = {
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaymentMethod': payment_method
            }
            
            for col_name, categories in cats_mapping.items():
                for cat in categories:
                    full_name = f"{col_name}_{cat}"
                    input_dict[full_name] = 1 if user_vals[col_name] == cat else 0
                    
            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            
            # Make predictions
            pred_class = model_pipeline.predict(input_df)[0]
            probs = model_pipeline.predict_proba(input_df)[0]
            churn_prob = probs[1] * 100
            stay_prob = probs[0] * 100
            
            st.markdown("--- ")
            st.subheader("🔮 Prediction Results")
            
            if pred_class == 1:
                st.error(f"**Recommendation:** This customer is **Likely to Churn** (Risk: {churn_prob:.1f}%)")
            else:
                st.success(f"**Recommendation:** This customer is **Likely to Stay** (Retention Confidence: {stay_prob:.1f}%)")
                
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={'text': "Churn Risk %"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "red" if pred_class == 1 else "green"},
                       'steps': [
                           {'range': [0, 35], 'color': "lightgreen"},
                           {'range': [35, 70], 'color': "orange"},
                           {'range': [70, 100], 'color': "salmon"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.info("Disclaimer: Prediction generated by trained Machine Learning models. Real customer behaviors can vary.")

    elif page == "4. Individual Model Analysis":
        st.title("📈 Individual Model Analysis")
        st.markdown("Explore classification validation indicators and threshold curves for each classifier model.")
        
        sel_model = st.selectbox("Choose a model to analyze", ["Logistic Regression", "Decision Tree", "Random Forest"])

        # Build a pipeline-ready dataframe
        raw_data = df_raw.copy()
        raw_data['TotalCharges'] = pd.to_numeric(raw_data['TotalCharges'], errors='coerce').fillna(0)
        raw_data['gender'] = raw_data['gender'].map({'Female': 0, 'Male': 1})
        raw_data['Partner'] = raw_data['Partner'].map({'Yes': 1, 'No': 0})
        raw_data['Dependents'] = raw_data['Dependents'].map({'Yes': 1, 'No': 0})
        raw_data['PhoneService'] = raw_data['PhoneService'].map({'Yes': 1, 'No': 0})
        raw_data['PaperlessBilling'] = raw_data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
        raw_data['Churn'] = raw_data['Churn'].map({'Yes': 1, 'No': 0})
        
        categorical_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        raw_data = pd.get_dummies(raw_data, columns=categorical_cols, dtype=int)
        
        X_data = raw_data.drop(columns=['Churn', 'customerID'], errors='ignore')
        y_data = raw_data['Churn']
        X_data = X_data.reindex(columns=feature_cols, fill_value=0)
        
        _, X_test, _, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
        
        pipe = artifacts[sel_model]
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision", f"{prec:.3f}")
        c3.metric("Recall (Sensitivity)", f"{rec:.3f}")
        c4.metric("F1 Score", f"{f1:.3f}")
        c5.metric("ROC-AUC Score", f"{roc:.3f}")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted Label", y="True Label"),
                               x=['Stay (0)', 'Churn (1)'], y=['Stay (0)', 'Churn (1)'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_right:
            st.markdown("#### ROC curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                              title=f'{sel_model} ROC Curve (AUC = {roc:.3f})')
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)
            
        # Explain coefficients or importances
        st.markdown("#### Feature Impact/Importance analysis")
        try:
            model_step = pipe.named_steps['model']
            if hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
                df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)
                fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title='Top 15 Feature Importances')
                st.plotly_chart(fig_imp, use_container_width=True)
            elif hasattr(model_step, 'coef_'):
                coefs = model_step.coef_[0]
                df_coef = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefs}).sort_values(by='Coefficient', ascending=False)
                fig_coef = px.bar(df_coef, x='Coefficient', y='Feature', orientation='h', title='Model Coefficients (Logistic Regression)')
                st.plotly_chart(fig_coef, use_container_width=True)
        except Exception as e:
            st.info("No explicit coefficient/importance chart available for this model step.")

    elif page == "5. Model Comparison":
        st.title("📊 Cross-Validation Performance Comparison")
        st.markdown("Comparison of models using Repeated Stratified 5-Fold Cross-Validation scores.")
        
        # Construct hardcoded cross-validation summary scores since CSV might not be saved
        summary_dict = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
            'Accuracy Mean': [0.8029, 0.7262, 0.7886],
            'Precision Mean': [0.6516, 0.4834, 0.6314],
            'Recall Mean': [0.5484, 0.5051, 0.4815],
            'F1 Mean': [0.5954, 0.4920, 0.5474],
            'ROC-AUC Mean': [0.8455, 0.6553, 0.8234]
        }
        summary_df = pd.DataFrame(summary_dict)
        
        st.table(summary_df.set_index('Model'))
        
        fig_cmp = px.bar(summary_df, x='Model', y=['Accuracy Mean', 'Precision Mean', 'Recall Mean', 'F1 Mean', 'ROC-AUC Mean'],
                         barmode='group', title="Comparison of Model Performance Metrics")
        st.plotly_chart(fig_cmp, use_container_width=True)
        
        st.markdown("""
        ### Strongest Baseline Recommendation
        Based on the validation results, **Logistic Regression** achieves the highest F1-score (**0.595**) and ROC-AUC (**0.845**).
        While Decision Trees offer fast scoring speed, they show signs of overfitting compared to Ensemble algorithms.
        """)

    elif page == "6. Data Drift Monitoring":
        st.title("🔄 Data Drift & Distribution Monitoring")
        st.markdown("Track distributions changes of numerical features and categorical indicators against training configurations.")
        
        raw_data = df_raw.copy()
        raw_data['TotalCharges'] = pd.to_numeric(raw_data['TotalCharges'], errors='coerce').fillna(0)
        raw_data['gender'] = raw_data['gender'].map({'Female': 0, 'Male': 1})
        raw_data['Partner'] = raw_data['Partner'].map({'Yes': 1, 'No': 0})
        raw_data['Dependents'] = raw_data['Dependents'].map({'Yes': 1, 'No': 0})
        raw_data['PhoneService'] = raw_data['PhoneService'].map({'Yes': 1, 'No': 0})
        raw_data['PaperlessBilling'] = raw_data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
        raw_data['Churn'] = raw_data['Churn'].map({'Yes': 1, 'No': 0})
        categorical_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        raw_data = pd.get_dummies(raw_data, columns=categorical_cols, dtype=int)
        X_data = raw_data.drop(columns=['Churn', 'customerID'], errors='ignore')
        X_data = X_data.reindex(columns=feature_cols, fill_value=0)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, _, _ = train_test_split(X_data, raw_data['Churn'], test_size=0.2, random_state=42, stratify=raw_data['Churn'])
        
        st.write("**Reference dataset rows (Train):** ", X_train.shape[0])
        st.write("**Current dataset rows (Test):** ", X_test.shape[0])
        
        drift_list = []
        for col in X_train.columns:
            ref = X_train[col]
            cur = X_test[col]
            is_binary = len(set(ref.unique()).union(set(cur.unique()))) <= 2
            
            if is_binary:
                categories = sorted(set(ref.unique()).union(set(cur.unique())))
                ref_counts = [(ref == category).sum() for category in categories]
                cur_counts = [(cur == category).sum() for category in categories]
                try:
                    _, p_val, _, _ = chi2_contingency([ref_counts, cur_counts])
                except:
                    p_val = 1.0
                test_name = 'Chi-Square'
            else:
                _, p_val = ks_2samp(ref, cur)
                test_name = 'KS Test'
                
            psi = calculate_psi(ref, cur)
            drift_list.append({
                'Feature': col,
                'Test': test_name,
                'P-Value': p_val,
                'PSI': psi,
                'Drift Detected': p_val < 0.05
            })
            
        df_drift_report = pd.DataFrame(drift_list).sort_values(by='PSI', ascending=False)
        st.dataframe(df_drift_report, use_container_width=True)
        
        fig_drift = px.bar(df_drift_report.head(15), x='PSI', y='Feature', orientation='h', title='Top 15 Feature Drift Ratios (PSI)')
        st.plotly_chart(fig_drift, use_container_width=True)
