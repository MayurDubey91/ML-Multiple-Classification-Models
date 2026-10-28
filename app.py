import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("🔬 Breast Cancer Classification")
st.markdown("**BITS Pilani WILP - ML Assignment 2** | Multiple Classification Models")
st.markdown("---")

# Load models
@st.cache_resource
def load_all():
    models = {}
    paths = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "SVM": "model/svm.joblib"
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load("model/scaler.joblib") if os.path.exists("model/scaler.joblib") else None
    return models, scaler

models, scaler = load_all()

if not models:
    st.error("Models not found in the 'model/' folder. Please ensure the repository contains the trained models.")
    st.stop()

# Comparison Table
st.subheader("📊 Model Performance Comparison")
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"],
    "Accuracy": [0.9825, 0.9474, 0.9561, 0.9298, 0.9825],
    "AUC": [0.9954, 0.9901, 0.9788, 0.9868, 0.9950],
    "Precision": [0.9861, 0.9583, 0.9589, 0.9444, 0.9861],
    "Recall": [0.9861, 0.9583, 0.9722, 0.9444, 0.9861],
    "F1": [0.9861, 0.9583, 0.9655, 0.9444, 0.9861],
    "MCC": [0.9623, 0.8869, 0.9054, 0.8492, 0.9623]
})
st.dataframe(comparison.style.highlight_max(axis=0, subset=["Accuracy", "AUC", "F1", "MCC"]), use_container_width=True)

st.markdown("---")

# Sidebar
st.sidebar.header("Select Model")
model_name = st.sidebar.selectbox("Classifier", list(models.keys()))
model = models[model_name]

st.subheader(f"Prediction with **{model_name}**")

# Feature names
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

option = st.radio("Input method", ["Use sample from test set", "Upload CSV"])

if option == "Use sample from test set":
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
        idx = st.slider("Select sample", 0, len(test_df)-1, 0)
        row = test_df.iloc[idx]
        actual = "Benign" if row['target'] == 1 else "Malignant"
        st.info(f"Actual label: **{actual}**")
        X = row[features].values.reshape(1, -1)
    else:
        st.warning("test_data.csv not found.")
        st.stop()
else:
    uploaded = st.file_uploader("Upload CSV (30 features, no target)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.iloc[0, :30].values.reshape(1, -1)
    else:
        st.info("Please upload a CSV file.")
        st.stop()

# Scale if required
if model_name != "Decision Tree" and scaler is not None:
    X_input = scaler.transform(X)
else:
    X_input = X

if st.button("Predict", type="primary"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    label = "Benign" if pred == 1 else "Malignant"
    conf = proba[pred] * 100

    c1, c2 = st.columns(2)
    c1.metric("Prediction", label)
    c2.metric("Confidence", f"{conf:.1f}%")
    st.progress(float(proba[1]))
    st.caption("Bar shows probability of Benign")

st.markdown("---")
st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic) | 30 features | Binary classification")
