from pathlib import Path

import joblib
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

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Model Explorer", layout="wide")
st.title("Breast Cancer Classification Model Explorer")
st.write("Upload labeled test data, select a trained model, and view its results.")

uploaded_file = st.file_uploader(
    "Upload test data as a CSV file (including the target column)",
    type="csv",
)
selected_name = st.selectbox("Choose a model", list(MODEL_FILES))

if uploaded_file is None:
    st.info("Upload test_data.csv to display results.")
    st.stop()

data = pd.read_csv(uploaded_file)

if "target" not in data.columns:
    st.error("The CSV must include a target column.")
    st.stop()

X = data.drop(columns=["target"])
y = data["target"]

model_path = MODEL_DIR / MODEL_FILES[selected_name]

try:
    model = joblib.load(model_path)
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }

    st.subheader(f"Results: {selected_name}")
    st.dataframe(pd.DataFrame([metrics]).round(4), use_container_width=True)

    st.subheader("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y, predictions, labels=[0, 1]),
            index=["Actual malignant", "Actual benign"],
            columns=["Predicted malignant", "Predicted benign"],
        )
    )

    st.subheader("Classification Report")
    st.text(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            zero_division=0,
        )
    )

except Exception as error:
    st.exception(error)