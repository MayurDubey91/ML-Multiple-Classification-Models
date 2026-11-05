import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def prepare(frame):
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        return None, missing
    data = frame[FEATURES + [TARGET]].dropna()
    return data, []


def score_model(name, model, frame):
    X = frame[FEATURES]
    y = frame[TARGET].astype(int).to_numpy()
    X_input = scaler.transform(X) if name in NEEDS_SCALING else X
    pred = model.predict(X_input)
    score = model.predict_proba(X_input)[:, 1]
    metrics = {
        "Accuracy": round(float(accuracy_score(y, pred)), 4),
        "AUC": round(float(roc_auc_score(y, score)), 4),
        "Precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y, pred)), 4),
    }
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: no default", "Actual: default"],
        columns=["Predicted: no default", "Predicted: default"],
    )
    report_dict = classification_report(
        y,
        pred,
        target_names=["no default", "default"],
        zero_division=0,
        output_dict=True,
    )
    report = pd.DataFrame(report_dict).T.round(4)
    report.index.name = "class"
    return metrics, cm_df, report


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
    "Decision Tree is the winner on accuracy, AUC, and MCC. These numbers were calculated on the saved test split. "
    "The section below recalculates them from the saved models, without training again."
)

st.sidebar.header("Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(models.keys()))
st.sidebar.markdown(
    "**Class labels**  \n"
    "0 = no default  \n"
    "1 = default next month"
)

st.subheader("Evaluate on test data")
st.write(
    "Upload a CSV that has the 23 feature columns and the target "
    f"`{TARGET}`. Or use the test file already in the repository."
)
option = st.radio(
    "Which test file?",
    ["Use test_data.csv from the repository", "Upload a test CSV"],
    horizontal=True,
)

if option == "Use test_data.csv from the repository":
    if not os.path.exists("test_data.csv"):
        st.error("test_data.csv is missing from the repository.")
        st.stop()
    eval_df, missing = prepare(pd.read_csv("test_data.csv"))
else:
    uploaded = st.file_uploader(
        "Test CSV with the 23 features and the target column.",
        type="csv",
    )
    if uploaded is None:
        st.info("Upload a CSV to continue. The included test_data.csv is a valid file.")
        st.stop()
    eval_df, missing = prepare(pd.read_csv(uploaded))

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

st.write(f"Rows used for evaluation: {len(eval_df)}")
with st.spinner(f"Scoring {model_name} on the test file..."):
    metrics, cm_df, report = score_model(model_name, models[model_name], eval_df)

st.markdown(f"**Metrics for {model_name}**")
st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)
with c2:
    st.markdown("**Classification report**")
    st.dataframe(report, use_container_width=True)

st.subheader(f"Predict one client with {model_name}")
idx = st.slider("Row in the file above", 0, len(eval_df) - 1, 0)
row = eval_df.iloc[idx]
actual = "Default" if int(row[TARGET]) == 1 else "No default"
st.info(f"Actual label of this client: **{actual}**")
X = row[FEATURES].to_frame().T
X_input = scaler.transform(X) if model_name in NEEDS_SCALING else X
pred = int(models[model_name].predict(X_input)[0])
proba = models[model_name].predict_proba(X_input)[0]
p1, p2 = st.columns(2)
p1.metric("Prediction", "Default" if pred == 1 else "No default")
p2.metric("P(default)", f"{proba[1] * 100:.1f}%")
st.progress(float(proba[1]))
