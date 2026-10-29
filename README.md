# Heart Disease Classification — ML Assignment 2

**BITS Pilani | Work Integrated Learning Programmes Division | M.Tech (AIML)**  
**Course:** Machine Learning | **Assignment:** 2 | **Marks:** 15

---

## Problem Statement

Cardiovascular disease is one of the leading causes of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. This project builds and compares five classical machine learning classification models to predict the presence or absence of heart disease in a patient based on clinical measurements.

The task is a **binary classification problem**:
- **Class 0** — No heart disease
- **Class 1** — Heart disease present

---

## Dataset Description

| Property | Value |
|---|---|
| **Name** | Heart Disease Dataset (UCI Cleveland) |
| **Source** | [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Rows (Instances)** | 303 |
| **Features** | 13 |
| **Target** | `target` (0 = No Disease, 1 = Disease) |
| **Class Balance** | 165 Disease (54.5%), 138 No Disease (45.5%) |

### Features

| # | Feature | Description | Type |
|---|---|---|---|
| 1 | `age` | Age in years | Continuous |
| 2 | `sex` | Sex (1 = male, 0 = female) | Categorical |
| 3 | `cp` | Chest pain type (0–3) | Categorical |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Continuous |
| 5 | `chol` | Serum cholesterol (mg/dl) | Continuous |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Categorical |
| 7 | `restecg` | Resting ECG results (0–2) | Categorical |
| 8 | `thalach` | Maximum heart rate achieved | Continuous |
| 9 | `exang` | Exercise-induced angina (1 = yes) | Categorical |
| 10 | `oldpeak` | ST depression induced by exercise | Continuous |
| 11 | `slope` | Slope of the peak exercise ST segment | Categorical |
| 12 | `ca` | Number of major vessels coloured by fluoroscopy (0–4) | Continuous |
| 13 | `thal` | Thalassemia type (0–3) | Categorical |

**Preprocessing:**
- Missing values (`?`) replaced with column mode
- All features scaled using `StandardScaler` (fit on train, transform on test)
- Train/Test split: 80% / 20%, stratified, `random_state=42`

---

## GitHub Repository Link

> **[https://github.com/YOUR_USERNAME/heart-disease-classifier](https://github.com/YOUR_USERNAME/heart-disease-classifier)**

*(Replace `YOUR_USERNAME` with your actual GitHub username before submission)*

---

## Repository Structure

```
heart-disease-classifier/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test split (20%) for Streamlit upload
├── model_metrics.csv       # Saved evaluation results for all 5 models
├── confusion_matrices.png  # All confusion matrices (saved figure)
├── data/
│   └── heart.csv           # Full dataset
└── model/
    ├── scaler.joblib        # Fitted StandardScaler
    ├── lr_model.joblib      # Logistic Regression
    ├── dt_model.joblib      # Decision Tree
    ├── knn_model.joblib     # K-Nearest Neighbor
    ├── nb_model.joblib      # Naive Bayes (Gaussian)
    └── svm_model.joblib     # Support Vector Machine
```

---

## Models Used

All five models were trained on the same dataset using identical train/test splits and preprocessing.

| # | Model |
|---|---|
| 1 | Logistic Regression |
| 2 | Decision Tree Classifier |
| 3 | K-Nearest Neighbor Classifier |
| 4 | Naive Bayes (Gaussian) |
| 5 | Support Vector Machine |

---

## Model Comparison Table

> *Results on the held-out 20% test set (61 samples). Run `train_models.py` to regenerate.*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8033 | 0.8690 | 0.7692 | 0.9091 | 0.8333 | 0.6098 |
| Decision Tree | 0.7049 | 0.6975 | 0.7027 | 0.7879 | 0.7429 | 0.4029 |
| K-Nearest Neighbor | 0.8033 | 0.8631 | 0.7692 | 0.9091 | 0.8333 | 0.6098 |
| Naive Bayes | 0.8197 | 0.8755 | 0.7895 | 0.9091 | 0.8451 | 0.6410 |
| Support Vector Machine | **0.8197** | **0.8831** | 0.7750 | **0.9394** | **0.8493** | **0.6481** |

*Test set: 20% stratified split (61 samples). Best values shown in bold.*

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved Accuracy = 0.8033 and AUC = 0.8690. As a linear model it performs well on this dataset because many features (e.g., `cp`, `thalach`, `oldpeak`) have roughly linear relationships with the log-odds of disease. Strong recall (0.9091) means it correctly identifies most disease cases, which is critical in medical diagnosis. |
| Decision Tree | Lowest performer with Accuracy = 0.7049 and AUC = 0.6975. Without pruning, the tree overfits training data and generalises poorly. The low MCC (0.4029) confirms it struggles to distinguish both classes reliably. Would benefit from max_depth pruning or ensemble methods (Random Forest). |
| K-Nearest Neighbor | Matched Logistic Regression exactly (Accuracy = 0.8033, AUC = 0.8631) with k=5 after StandardScaler. KNN benefits greatly from scaling here. Its identical metric values to LR suggest both models find the same decision boundary on this dataset. |
| Naive Bayes | Achieved Accuracy = 0.8197 and the highest recall (0.9091) tied with LR and KNN. Gaussian NB works well here because most features are approximately Gaussian after scaling. Despite the naive independence assumption, it ranks among the top performers — a common observation in medical datasets. |
| Support Vector Machine | Best overall: Accuracy = 0.8197, highest AUC = 0.8831, best F1 = 0.8493 and MCC = 0.6481. The RBF kernel captures non-linear feature interactions, and the max-margin objective finds a robust decision boundary. High recall (0.9394) means it correctly identifies the most disease cases, which is the most important property for a heart disease screening tool. |
| **Overall Winner** | **Support Vector Machine** — best AUC (0.8831), F1 (0.8493), Recall (0.9394), and MCC (0.6481). Its ability to minimise false negatives (missed disease cases) makes it the most clinically useful classifier on this dataset. |

---

## Streamlit Application

The live app allows anyone to upload the test CSV and evaluate any of the 5 trained models without retraining.

**Features:**
- CSV upload (test data with target column)
- Model selection dropdown
- 6 evaluation metrics displayed as cards (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap
- Classification report table
- All-models accuracy comparison bar chart

**Live App:** [https://YOUR_APP.streamlit.app](https://YOUR_APP.streamlit.app)  
*(Replace with your deployed Streamlit URL)*

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/heart-disease-classifier.git
cd heart-disease-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates model/*.joblib and test_data.csv)
python train_models.py

# 4. Launch Streamlit app
streamlit run app.py
```

---

## Dependencies

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
```

---

*Submitted for BITS Pilani WILP M.Tech AIML — Machine Learning — Assignment 2*
