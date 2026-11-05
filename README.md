# Credit Card Default Prediction - Multiple ML Models

## a. Problem Statement
Banks need to know, before the next month, which credit-card customers are likely to miss a payment. This project builds and compares five classification models that predict whether a client will default on their credit-card payment next month, using their credit limit, demographics, recent repayment status, bill amounts, and previous payments.

## b. Dataset Description
- **Dataset Name**: Default of Credit Card Clients
- **Source**: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (dataset id 350). Training used the supplied workbook `default of credit card clients.xls`.
- **Number of Features**: 23 (ID column was dropped; it is only a row number)
- **Number of Instances**: 30,000
- **Type**: Binary classification
- **Target**: `default payment next month` (1 = default, 0 = no default)
- **Class balance**: about 22.12% default and 77.88% no default. The classes are not equal, so accuracy alone can look high even for a weak model.
- **Features**:
  - `LIMIT_BAL`: credit limit
  - `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`: client background
  - `PAY_0`, `PAY_2` to `PAY_6`: repayment status for the last six months
  - `BILL_AMT1` to `BILL_AMT6`: bill statement amounts
  - `PAY_AMT1` to `PAY_AMT6`: amounts paid in previous months
- **Split used**: 80% train (24,000) and 20% test (6,000), stratified, `random_state=42`
- **Scaling**: StandardScaler fitted on the training set only, then applied to Logistic Regression, KNN, Naive Bayes, and SVM. The Decision Tree was trained on the original values.

## c. GitHub Repository Link
https://github.com/MayurDubey91/ML-Multiple-Classification-Models

## d. Models Used

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.8077   | 0.7076 | 0.6868    | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree          | 0.8167   | 0.7508 | 0.6697    | 0.3376 | 0.4489 | 0.3828 |
| KNN                    | 0.8065   | 0.7325 | 0.6153    | 0.3338 | 0.4328 | 0.3507 |
| Naive Bayes            | 0.7525   | 0.7249 | 0.4515    | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.8077   | 0.7044 | 0.6876    | 0.2389 | 0.3546 | 0.3242 |

Positive class for Precision, Recall, and F1 is default (1). AUC is ROC-AUC. SVM is a linear SVM with probability calibration so that AUC and probabilities can be computed.

### Observations

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Accuracy is 0.8077, only a little above a model that always says "no default" (about 0.7788). Precision is high (0.6868) but recall is low (0.2396), so it misses most real defaulters. |
| Decision Tree          | Best overall model on this split. Highest accuracy (0.8167), highest AUC (0.7508), and highest MCC (0.3828). Depth was limited (`max_depth=6`, `min_samples_leaf=50`) so it does not memorize rare rows. |
| KNN                    | Close to the linear models on accuracy (0.8065). AUC 0.7325 is better than Logistic Regression and SVM. Needs scaled features. `k=15` was used so one odd neighbour does not dominate. |
| Naive Bayes            | Lowest accuracy (0.7525) because it raises more false alarms. It has the best recall (0.5539) and the best F1 (0.4975), so it catches more defaulters than the other models, but many of its "default" predictions are wrong. |
| Support Vector Machine | Almost the same as Logistic Regression (accuracy 0.8077, AUC 0.7044, MCC 0.3242). A linear decision boundary does not separate defaulters much better than logistic regression on these 23 features. |
| Overall, Winner for your dataset? | Decision Tree. It has the best accuracy, the best AUC, and the best MCC. If the bank cares more about catching defaulters than about false alarms, Naive Bayes is the stronger recall/F1 choice. |

---
**Note**: A dummy classifier that always predicts "no default" would score about 77.88% accuracy and 0 recall. Accuracy near 81% is only a small gain. AUC and MCC are the fairer comparison on this imbalanced dataset.
