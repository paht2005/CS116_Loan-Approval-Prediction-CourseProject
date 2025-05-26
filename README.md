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
├── CS116Project_LoanApprovalPrediciton.ipynb # Main Jupyter notebook
├── requirements.txt # Python Dependencies
└── README.md # Project overview
└── LICENSE
```

---

## 🚀 Pipeline Overview

### 1. Data Preprocessing
This phase focuses on transforming raw data into a clean and usable format for model training.
- **Handling Missing Values:** Missing data points were addressed using appropriate strategies, specifically **median for numerical features**, to ensure data completeness.
- **Outlier Removal:** Identified and mitigated the impact of outliers in numerical columns using **drop/fill in the blanks** methods to improve model robustness and accuracy.
- **Feature Encoding:** Categorical features were converted into numerical representations using **Label Encoding and One-Hot Encoding**, making them suitable for machine learning algorithms.
- **Correlation Matrix:** Generated a correlation matrix to understand relationships between features and identify potential multicollinearity.

### 2. Model Training & Selection
We explored and compared the performance of several powerful gradient boosting models:
- **Models Used:** Applied and compared **XGBoost, LightGBM, CatBoost**, and an **Ensemble Model** which combines **3 instances of XGBoost, 3 of LightGBM, and 3 of CatBoost**.
- **Hyperparameter Tuning:** Optimized model performance using **Optuna** for efficient hyperparameter search.
- **Validation Strategy:** Employed **K-Fold Cross-Validation** during hyperparameter tuning to ensure robust model selection and generalization performance.
- **Train-Test Split:** Models were trained and evaluated using an 80/20 train-test split to assess generalization performance.
### 3. Feature Engineering & Selection
This crucial step involved refining the feature set to enhance model performance.
- **Feature Augmentation/Combination:** New features were engineered by **combining existing features** to capture more complex relationships within the data.
- **Feature Selection:** Low-impact features were identified and removed based on:
  - **Correlation Matrix:** Features highly correlated with each other or with low correlation to the target variable were considered for removal.
  - **Random Forest Feature Importance:** Utilized feature importance scores from Random Forest models to rank and select the most impactful features.
  - **Model Interpretability:** Features that did not contribute significantly to model understanding or performance were dropped.
- **Final Dropped Features:**
  - ``cb_person_cred_hist_length``
  - ``cb_person_default_on_file_encoded``


### 4. Evaluation & Ensemble
The final phase involved evaluating the models and combining them for improved predictive power.
- **Individual Model Evaluation:** Each model's performance was rigorously evaluated both before and after feature selection using key metrics.
- **Metrics:**
  - **F1 Macro Score:** The primary evaluation metric, chosen for its ability to balance precision and recall across imbalanced classes.
  - **Accuracy:** Overall correctness of predictions.
  - **AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between classes.
- **Ensemble Model:** A custom ensemble approach was implemented to combine the strengths of individual models, aiming for superior overall performance.

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
| LightGBM      | 0.8722        | 0.9357                |0.95714      |
| Catboost     | 0.8633        | 0.9285               |0.95822      |
| XGBoost      | 0.8792        | 0.9407                |0.9558      |    
| **Ensemble**     | 0.8931        | 0.9530                   |0.96307     |

**Key Highlights:**
- The **Ensemble Model** achieved the **best overall performance**, demonstrating the power of combining diverse models.
- **XGBoost** showed strong individual performance, especially in F1 Score and Accuracy.
- **CatBoost** and **LightGBM** also performed competitively, contributing valuable insights to the ensemble.
## 📌 Conclusion
After performing extensive data preprocessing, model comparisons, feature engineering, and ensemble modeling, we found that our approach led to robust and accurate predictions for loan approval. The **Ensemble Model** consistently outperformed individual models, achieving an impressive **F1 Macro Score of 0.8931**, **Accuracy of 0.9530**, and **AUC of 0.96307**. This confirms that **careful feature selection and strategic model ensembling** can have significant positive effects on model performance in real-world lending scenarios.

## 🤝 Contributing
Contributions, improvements, or issue reports are welcome. Please open a pull request or submit an issue!

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

