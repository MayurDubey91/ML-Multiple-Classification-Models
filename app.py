import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Default Classifier", layout="wide")

st.title("Credit Card Default Classification")
st.markdown("**BITS Pilani WILP — ML Assignment 2**")
st.caption(
    "Dataset: UCI Default of Credit Card Clients (30,000 clients, 23 features). "
    "Target: will the client default next month?"
)

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET = "default payment next month"
NEEDS_SCALING = {"Logistic Regression", "KNN", "Naive Bayes", "SVM"}

COMPARISON = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"],
    "Accuracy": [0.8077, 0.8167, 0.8065, 0.7525, 0.8077],
    "AUC": [0.7076, 0.7508, 0.7325, 0.7249, 0.7044],
    "Precision": [0.6868, 0.6697, 0.6153, 0.4515, 0.6876],
    "Recall": [0.2396, 0.3376, 0.3338, 0.5539, 0.2389],
    "F1": [0.3553, 0.4489, 0.4328, 0.4975, 0.3546],
    "MCC": [0.3244, 0.3828, 0.3507, 0.3386, 0.3242],
})


@st.cache_resource
def load_models():
    paths = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "SVM": "model/svm.joblib",
    }
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load("model/scaler.joblib") if os.path.exists("model/scaler.joblib") else None
    return models, scaler


models, scaler = load_models()
if not models or scaler is None:
    st.error("Trained models were not found in the model/ folder.")
    st.stop()

st.subheader("Model comparison (held-out test set)")
st.dataframe(
    COMPARISON.style.highlight_max(axis=0, subset=["Accuracy", "AUC", "F1", "MCC"]),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "About 22% of clients default. A model that always predicts 'no default' already gets ~78% accuracy. "
    "Decision Tree is the winner on accuracy, AUC, and MCC."
)

st.sidebar.header("Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(models.keys()))
st.sidebar.markdown(
    "**Class labels**  \n"
    "0 = no default  \n"
    "1 = default next month"
)

st.subheader(f"Predict with {model_name}")
option = st.radio(
    "How should the app get the client data?",
    ["Use a client from the test set", "Upload a CSV"],
    horizontal=True,
)

if option == "Use a client from the test set":
    if not os.path.exists("test_data.csv"):
        st.error("test_data.csv is missing from the repository.")
        st.stop()
    test_df = pd.read_csv("test_data.csv")
    idx = st.slider("Test-set row", 0, len(test_df) - 1, 0)
    row = test_df.iloc[idx]
    actual = "Default" if int(row[TARGET]) == 1 else "No default"
    st.info(f"Actual label of this client: **{actual}**")
    with st.expander("Show the 23 feature values"):
        st.dataframe(row[FEATURES].to_frame("value"), use_container_width=True)
    X = row[FEATURES].to_numpy(dtype=float).reshape(1, -1)
else:
    uploaded = st.file_uploader(
        "CSV with the 23 feature columns (header names should match the dataset). Target column is optional.",
        type="csv",
    )
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    uploaded_df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in uploaded_df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()
    X = uploaded_df.loc[:, FEATURES].iloc[[0]].to_numpy(dtype=float)
    st.write(f"Using the first row of the uploaded file ({len(uploaded_df)} rows found).")

X_input = scaler.transform(X) if model_name in NEEDS_SCALING else X

if st.button("Predict", type="primary"):
    model = models[model_name]
    pred = int(model.predict(X_input)[0])
    proba = model.predict_proba(X_input)[0]
    label = "Default" if pred == 1 else "No default"
    c1, c2 = st.columns(2)
    c1.metric("Prediction", label)
    c2.metric("P(default)", f"{proba[1] * 100:.1f}%")
    st.progress(float(proba[1]))
    st.caption("The bar is the predicted probability of default (class 1).")
