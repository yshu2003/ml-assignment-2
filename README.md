# ml-assignment-2
# Breast Cancer Classification Using Machine Learning

## Problem Statement

This project uses machine learning to classify breast tumors as malignant or benign using measurements of cell nuclei. Five classification models are trained on the same dataset and compared using standard evaluation metrics. The Streamlit app lets users select a saved model and evaluate a labeled test CSV without retraining the model.

## Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

It contains 569 records and 30 numerical input features. The target column is encoded as `0` for malignant and `1` for benign. The dataset has no missing values. The data was divided into stratified training and test sets, with 80% used for training and 20% held out for testing.

## Project Links

- **GitHub repository:** https://github.com/yshu2003/ml-assignment-2
- **Live Streamlit app:** https://yshu2003-ml-assignment-2-app-2ro40b.streamlit.app/

## Models and Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Precision, recall, and F1 use benign (`1`) as the positive class.

## Model Performance Observations

- **Logistic Regression:** It achieved 0.9825 accuracy and 0.9954 AUC. Its confusion matrix showed two incorrect predictions.
- **Decision Tree:** It achieved 0.9123 accuracy and made ten incorrect predictions, the most among the five models.
- **KNN:** It achieved 0.9561 accuracy and made five incorrect predictions.
- **Gaussian Naive Bayes:** It achieved 0.9386 accuracy and made seven incorrect predictions. Its AUC was 0.9878.
- **Support Vector Machine:** It achieved 0.9825 accuracy and made two incorrect predictions, matching Logistic Regression on accuracy and MCC.
- **Overall winner:** Logistic Regression and Support Vector Machine performed best on accuracy and MCC. Logistic Regression had a slightly higher AUC in this run.

## Streamlit App

The app accepts a labeled CSV file containing the required input features and the `target` column. Users can select one of the five saved models and view accuracy, AUC, precision, recall, F1, MCC, a confusion matrix, and a classification report. The app loads the saved model and evaluates the uploaded data without retraining.

## Run Locally

Install the dependencies and start the app with:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Files

- `app.py` — Streamlit application
- `requirements.txt` — required Python packages
- `test_data.csv` — held-out test data, including the target column
- `Model/` — saved trained models for all five classifiers
- `ASSIGNMENT.ipynb` — notebook containing the model implementation and evaluation
