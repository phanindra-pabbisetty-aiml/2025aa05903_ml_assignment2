import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

# Set page config
st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

# 1. Title
st.title("✨ Wine Quality Prediction App")
st.markdown("Upload your test data and select a model to evaluate performance.")

# 2. Load Preprocessing Objects
@st.cache_resource
def load_preprocessors():
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    return scaler, le

scaler, le = load_preprocessors()

# 3. Sidebar - Model Selection
st.sidebar.header("Configuration")
model_options = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'K-Nearest Neighbors': 'k-nearest_neighbors.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest': 'random_forest.joblib',
    'XGBoost': 'xgboost.joblib'
}
selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

# 4. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file containing wine features and quality", type="csv")

if uploaded_file is not None:
    # Read Data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview (First 5 Rows)")
    st.dataframe(df.head())

    if st.button("Run Evaluation"):
        # 5. Preprocessing
        try:
            # Assuming the last column is 'quality'
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Transform data
            X_scaled = scaler.transform(X)
            y_encoded = le.transform(y)

            # 6. Load Model and Predict
            model_path = model_options[selected_model_name]
            model = joblib.load(model_path)
            
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)

            # 7. Calculate Metrics
            acc = accuracy_score(y_encoded, y_pred)
            # Multi-class AUC
            auc = roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='weighted')
            prec = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_encoded, y_pred)

            # Display Metrics
            st.write(f"### Performance Metrics: {selected_model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1-Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # 8. Confusion Matrix Visualization
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_encoded, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {selected_model_name}')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")