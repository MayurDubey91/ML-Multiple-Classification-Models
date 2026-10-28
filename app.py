import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("🔬 Breast Cancer Classification")
st.markdown("**BITS Pilani WILP - ML Assignment 2** | Multiple Classification Models")
st.markdown("---")

@st.cache_resource
def get_models_and_data():
    # Try to load existing models
    models = {}
    paths = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "SVM": "model/svm.joblib"
    }
    scaler = None
    models_exist = all(os.path.exists(p) for p in paths.values()) and os.path.exists("model/scaler.joblib")
    
    if models_exist:
        for name, path in paths.items():
            models[name] = joblib.load(path)
        scaler = joblib.load("model/scaler.joblib")
        st.sidebar.success("Loaded pre-trained models")
    else:
        # Train models if not present
        st.sidebar.info("Training models (first run)...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42).fit(X_train, y_train)
        models["KNN"] = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
        models["Naive Bayes"] = GaussianNB().fit(X_train_scaled, y_train)
        models["SVM"] = SVC(probability=True, random_state=42).fit(X_train_scaled, y_train)
        st.sidebar.success("Models trained successfully")
    
    return models, scaler

models, scaler = get_models_and_data()

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

features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

option = st.radio("Input method", ["Use sample values", "Upload CSV"])

if option == "Use sample values":
    # Default sample (malignant-like)
    default_vals = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    X = np.array(default_vals).reshape(1, -1)
    st.info("Using a sample feature vector (you can modify the code for more options)")
else:
    uploaded = st.file_uploader("Upload CSV (30 features)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.iloc[0, :30].values.reshape(1, -1)
    else:
        st.info("Please upload a CSV or choose sample values.")
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
