# 🫀 Heart Disease Risk Predictor

> An end-to-end Machine Learning web application that predicts heart disease risk based on 13 clinical parameters — classifying patients as **LOW**, **MODERATE**, or **HIGH** risk.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25+-brightgreen)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.8674-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Input Parameters](#-input-parameters)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project builds a complete machine learning pipeline — from raw data preprocessing to a deployed interactive Flask web application — for predicting heart disease risk. Given 13 clinical attributes (e.g., age, chest pain type, cholesterol), the model classifies a patient's heart disease risk into one of three categories: **Low**, **Moderate**, or **High**.

The model was trained on the **UCI Cleveland Heart Disease Dataset** (303 records) and achieves **85%+ accuracy** with an **AUC-ROC of 0.8674** using a Tuned Random Forest classifier.

---

## 🖥️ Demo

![App Demo](screenshots/app_demo.png)

To run the app locally, follow the [Getting Started](#-getting-started) section below.

---

## ✨ Features

- 🔬 Full ML pipeline: data cleaning → EDA → model training → hyperparameter tuning → deployment
- 📊 Comparison of multiple classifiers (Logistic Regression, Random Forest, Gradient Boosting)
- ⚙️ Hyperparameter tuning using **GridSearchCV** (36 combinations evaluated)
- ✅ **10-Fold Cross Validation** for robust evaluation
- 🌐 Interactive Flask web app with a clean form-based UI
- 📈 Visualizations: correlation heatmap, ROC curves, confusion matrices, feature importance

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository — Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **File:** `heart_cleveland_upload.csv`
- **Records:** 303 patients
- **Features:** 13 clinical attributes + 1 target variable
- **Target:** Binary classification (presence/absence of heart disease), mapped to risk tiers

---

## 🔬 ML Pipeline

```
Raw Data
   │
   ▼
Data Cleaning & Preprocessing
   │  - Handle missing values
   │  - Feature scaling (StandardScaler)
   │
   ▼
Exploratory Data Analysis (EDA)
   │  - Correlation heatmap
   │  - Feature importance plot
   │
   ▼
Model Training & Comparison
   │  - Logistic Regression
   │  - Random Forest
   │  - Gradient Boosting
   │
   ▼
Hyperparameter Tuning
   │  - GridSearchCV (36 combinations)
   │  - 10-Fold Cross Validation
   │
   ▼
Best Model: Tuned Random Forest
   │  - Accuracy: 85%+  |  AUC: 0.8674
   │
   ▼
Serialization → heart_disease_model.pkl + scaler.pkl
   │
   ▼
Flask Web App Deployment
```

---

## 📊 Model Performance

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | ~90% | — |
| Random Forest | 85%+ | 0.87 |
| Gradient Boosting | 85%+ | 0.87 |
| **Tuned Random Forest** ✅ | **85%+** | **0.8674** |

> The Tuned Random Forest was selected as the final model based on cross-validated AUC and generalization performance.

**Evaluation Artifacts:**
- `roc_curve_comparison.png` — ROC curves for all models
- `confusion_matrix_*.png` — Confusion matrices per model
- `feature_importance.png` — Top features by importance

---

## 💡 Key Findings

Top 5 clinical features influencing heart disease prediction:

| Rank | Feature | Description |
|---|---|---|
| 1 | `thalach` | Maximum Heart Rate Achieved |
| 2 | `cp` | Chest Pain Type |
| 3 | `oldpeak` | ST Depression Induced by Exercise |
| 4 | `ca` | Number of Major Vessels (0–3) |
| 5 | `age` | Patient Age |

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| ML & Modeling | Scikit-learn, Random Forest, GridSearchCV |
| Web Framework | Flask 3.1 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Notebook | Jupyter Notebook |
| IDE | VS Code |

---

## 📂 Project Structure

```
Heart_disease_risk_prediction/
│
├── app.py                              # Flask web application
├── heart_disease_predictor.ipynb       # Full ML pipeline notebook
├── heart_cleveland_upload.csv          # Dataset
├── heart_disease_model.pkl             # Trained model (serialized)
├── scaler.pkl                          # Feature scaler (serialized)
│
├── screenshots/
│   └── app_demo.png                    # App screenshot
│
├── confusion_matrix_Logistic_Regression.png
├── confusion_matrix_Random_Forest.png
├── confusion_matrix_Gradient_Boosting.png
├── confusion_matrix_Tuned_Random_Forest.png
├── roc_curve_comparison.png
├── correlation_heatmap.png
├── feature_importance.png
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/nandkishor-ux/Heart_disease_risk_prediction.git
cd Heart_disease_risk_prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib

# 4. Run the Flask app
python app.py

# 5. Open in your browser
# http://127.0.0.1:5000
```

---

## 🧾 Input Parameters

The web app accepts the following 13 clinical inputs:

| Parameter | Description | Type |
|---|---|---|
| `age` | Age of patient | Numeric |
| `sex` | Sex (1 = male, 0 = female) | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina (1 = yes) | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment (0–2) | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) | Numeric |
| `thal` | Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |

---

## ⚠️ Disclaimer

> This tool is built for **educational and research purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

---

## 🙋 Author

**Nandkishor** — [GitHub](https://github.com/nandkishor-ux)

If you found this project helpful, consider giving it a ⭐!
