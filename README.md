# Breast Cancer Classification - Multiple ML Models

## a. Problem Statement
The goal of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is malignant or benign based on diagnostic features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass. This is a binary classification problem aimed at assisting early detection.

## b. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository / scikit-learn
- **Number of Features**: 30 (numeric features computed from cell nucleus characteristics)
- **Number of Instances**: 569
- **Type**: Binary Classification (0 = Malignant, 1 = Benign)
- **Class Distribution**: Approximately 63% Benign, 37% Malignant
- **Features include**: mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## c. GitHub Repository Link
https://github.com/MayurDubey91/ML-Multiple-Classification-Models

## d. Models Used

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree          | 0.9474   | 0.9901 | 0.9583    | 0.9583 | 0.9583 | 0.8869 |
| KNN                    | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes            | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Excellent performance. Highest accuracy and MCC. Works very well on this dataset after scaling. |
| Decision Tree          | Good but slightly lower than linear models. Interpretable. Used pre-pruning (max_depth=5, min_samples_leaf=10) to avoid overfitting. |
| KNN                    | Solid performance. Benefits from feature scaling. Sensitive to the choice of k (used k=5). |
| Naive Bayes            | Lowest performance among the five. Gaussian assumption is reasonable but feature independence does not fully hold. |
| Support Vector Machine | Tied with Logistic Regression for best accuracy. Very strong on this dataset with RBF kernel (default). |
| Overall, Winner for your dataset? | Logistic Regression and SVM (tied). Both achieved 98.25% accuracy and the highest MCC. |

---
**Note**: All models were trained on the same train-test split (80-20, stratified, random_state=42). Features were standardized for Logistic Regression, KNN, Naive Bayes, and SVM.
