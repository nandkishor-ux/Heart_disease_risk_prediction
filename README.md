# 🫀 Heart Disease Risk Predictor
> An end-to-end Machine Learning web application that predicts 
> heart disease risk based on 13 clinical parameters.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC--ROC-0.8674-orange)

---

## 📸 Demo
![App Demo](screenshots/app_demo.png)

---

## 🎯 Project Overview
Built a complete ML pipeline — from raw data cleaning to a 
deployed Flask web app — that classifies patients as 
LOW / MODERATE / HIGH risk for heart disease.

---

## 📊 Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | ~90% | - |
| Random Forest | 85%+ | 0.87 |
| Gradient Boosting | 85%+ | 0.87 |
| **Tuned Random Forest** | **85%+** | **0.8674** |

---

## 🔬 What I Did
- Cleaned and preprocessed UCI Cleveland dataset (303 records)
- Performed EDA — correlation heatmap, feature importance
- Trained and compared 3 ML models
- Hyperparameter tuning with GridSearchCV (36 combinations)
- 10-Fold Cross Validation | Mean AUC: 0.8674
- Deployed as interactive Flask web app

---

## 💡 Key Findings
Top 5 features influencing heart disease:
1. **thalach** — Maximum Heart Rate
2. **cp** — Chest Pain Type  
3. **oldpeak** — ST Depression
4. **ca** — Number of Major Vessels
5. **age**

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| ML | Scikit-learn, Random Forest, GridSearchCV |
| Web | Flask |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Joblib, VS Code |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/nandkishor-ux/heart-disease-app.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open browser
http://127.0.0.1:5000
```

---

## ⚠️ Disclaimer
This tool is for educational purposes only 
and is not a medical diagnosis.
