# 🏦 CS116 Project: Loan Approval Prediction 

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_S4E10-blue)](https://www.kaggle.com/competitions/playground-series-s4e10)

> This repository contains the full implementation of a machine learning pipeline developed for the Kaggle competition **Loan Approval Prediction (Playground Series - Season 4, Episode 10)**. Our goal is to predict whether a loan will be approved based on applicant information and credit history, leveraging state-of-the-art ML models and careful feature engineering.

---
# Team Information
| No.    | Student ID      | Full Name in Vietnamese        | Position   | Github                                       | Email                   |
| ------ |:---------------:| ------------------------------:|-----------:|---------------------------------------------:|-------------------------:
| 1      | 23521570        | Huynh Viet Tien                |Leader      |[SharkTien](https://encr.pw/SCu2w)            |23521570@gm.uit.edu.vn   |
| 2      | 23521143        | Nguyen Cong Phat               |Member      |[paht2005](https://github.com/paht2005)       |23521143@gm.uit.edu.vn   |
| 3      | 23520123        | Nguyen Minh Bao                |Member      |[baominh5xx2](https://github.com/baominh5xx2) |23520123@gm.uit.edu.vn   |        
| 4      | 23520133        | Pham Phu Bao                   |Member      |[itsdabao](https://github.com/itsdabao)       |23520133@gm.uit.edu.vn   |

## 📖 Table of Contents

- [✨ Features](#-features)
- [🗂️ Repository Structure](#️-repository-structure)
- [🚀 Pipeline Overview](#-pipeline-overview)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training & Selection](#2-model-training-&-selection)
  - [3. Feature Engineering & Selection](#3-feature-engineering-&-selection)
  - [4. Evaluation & Ensemble](#4-evaluation-&-ensemble)
- [⚙️ Installation](#️-installation)
- [🎯 Usage](#-usage)
- [📈 Results](#-results)
- [📌 Conclusion](#-conclusion)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- Full notebook implementation for EDA, preprocessing, training, evaluation.
- Comparison between multiple models: **XGBoost, LightGBM, CatBoost**, and an **Ensemble Model**.
- Robust data preprocessing including handling missing values, label encoding, and outlier removal.
- Strategic feature engineering and selection based on correlation and model interpretability.
- Uses **F1 Macro Score** as primary evaluation metric, alongside Accuracy and AUC.
- Performance evaluated using a single 80/20 train–test split.

---

## 🗂️ Repository Structure

```
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── test.csv
│ ├── df_train_preprocessed.csv
│ ├── df_test_preprocessed.csv
│ └── sample_submission.csv
├── catboost_info/
├── EDA/
├── features-engineering/
├── Preprocessing-data/
├── stage1/
├── CS116.ipynb # Jupyter notebook for testing
├── CS116P-final.ipynb # Main Jupyter notebook 
├── requirements.txt # Python Dependencies
└── README.md # Project overview
└── LICENSE
```

---

## 🚀 Pipeline Overview

### 1. Data Preprocessing
This phase focuses on transforming raw data into a clean and usable format for model training.
- **Missing Values:** Checked and confirmed no nulls in original dataset.
- **Outlier Removal:** Applied visual and statistical methods to remove anomalies.
- **Encoding:** Used Label and One-Hot Encoding for categorical variables.
- **Scaling:** Applied MinMaxScaler to ensure all features are within the same range [0,1].
= **Correlation Matrix:** Used to identify multicollinearity before feature selection.

### 2. Model Training & Selection
- **Models Used:** Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, CatBoost.
- **Validation:** 5-Fold Cross-Validation used to ensure robust performance.
- **Hyperparameter Tuning:** GridSearchCV (for small models), Optuna (for gradient boosting models).
### 3. Feature Engineering & Selection
This crucial step involved refining the feature set to enhance model performance.
- **Feature Creation:** Engineered domain-specific features:
  - ``affordability_score`` = income - loan * (1 + interest)
  - ``available_funds_ratio``, etc.
- **Feature Importance:** Assessed via XGBoost, Random Forest, and ANOVA.
- Dropped Low-impact Features: e.g., ``cb_person_cred_hist_length``, ``cb_person_default_on_file_encoded``

### 4. Evaluation & Ensemble
The final phase involved evaluating the models and combining them for improved predictive power.
- **Metrics:**
  - **F1 Macro Score** (primary)
  - Accuracy
  - AUC
- **Best Individual Model:** XGBoost with F1 = 0.8928, ACC = 0.9518
- Ensemble Techniques:
  - Soft Voting (3 models, weighted 0.4/0.3/0.3)
  - Stacking (used but gave slightly lower score)
  - Top-performing ensemble: **Soft Voting** **(XGB + LGBM + CatBoost)** with **F1 = 0.894**, **Accuracy = 0.9524**, **AUC = 0.96307**

---

## ⚙️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/paht2005/Loan-Approval-Prediction-CourseProjectgit
   cd Loan-Approval-Prediction-CourseProject

   ```
2. **(optional) Create environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook CS116-final.ipynb
   ```
## 📈 Results

Our rigorous evaluation demonstrated significant performance across all models, with the **Ensemble Model** achieving the highest scores.

| Model    | F1 Score     | ACC        | AUC   |
| ------ |:---------------:| ------------------------------:|-----------:|
| LogisticReg      | 0.7233        | 0.8173              | -      |
| RandomForest     | 0.8877        | 0.9500              | -      |
| LightGBM      | 0.8832       | 0.9450                | 0.9571      |
| Catboost     | 0.8880       | 0.9482              |0.9582      |
| XGBoost      | 0.8928       | 0.9518             |0.9558      |    
| **Ensemble**     | **0.894 **       | **0.9524 **                  |**0.9631**     |

> Ensemble improved **stability and robustness**, not just raw accuracy.

**Key Highlights:**
- The **Ensemble Model** achieved the **best overall performance**, demonstrating the power of combining diverse models.
- **XGBoost** showed strong individual performance, especially in F1 Score and Accuracy.
- **CatBoost** and **LightGBM** also performed competitively, contributing valuable insights to the ensemble.
## 📌 Conclusion
This project successfully developed a robust ML pipeline to predict loan approvals using real-world-inspired data. Through meticulous preprocessing, strong tree-based models, and strategic feature engineering, we achieved:
- High **F1 Macro Score = 0.894**, indicating excellent balance between precision and recall.
- Strong **Accuracy = 95.24%** and **AUC = 0.9631**.
- **Ensemble Learning** proved beneficial for improving overall model reliability and generalization.

The project demonstrates the power of **interpretable models**, **careful tuning**, and **model ensembling** in building production-level ML systems for financial applications.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

