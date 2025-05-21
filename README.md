# 🏦 CS116 Project: Loan Approval Prediction 

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_S4E10-blue)](https://www.kaggle.com/competitions/playground-series-s4e10)

> This repository contains the full implementation of a machine learning pipeline developed for the Kaggle competition **Loan Approval Prediction (Playground Series - Season 4, Episode 10)**. Our goal is to predict whether a loan will be approved based on applicant information and credit history, leveraging state-of-the-art ML models and careful feature engineering.

---
# Team Information
| No.    | Student ID      | Full Name in Vietnamese        | Position   | Github                                       | Email                   |
| ------ |:---------------:| ------------------------------:|-----------:|---------------------------------------------:|-------------------------:
| 1      | 23521570        | Huỳnh Việt Tiến                |Leader      |[SharkTien](https://encr.pw/SCu2w)            |23521570@gm.uit.edu.vn   |
| 2      | 23521143        | Nguyễn Công Phát               |Member      |[paht2005](https://github.com/paht2005)       |23521143@gm.uit.edu.vn   |
| 3      | 23520123        | Nguyễn Minh Bảo                |Member      |      |23520123@gm.uit.edu.vn   |        
| 4      | 23520133        | Phạm Phú Bảo                   |Member      |[itsdabao](https://github.com/itsdabao)       |23520133@gm.uit.edu.vn   |

## 📖 Table of Contents

- [✨ Features](#-features)
- [🗂️ Repository Structure](#️-repository-structure)
- [🚀 Pipeline Overview](#-pipeline-overview)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Evaluation](#4-evaluation)
- [⚙️ Installation](#️-installation)
- [🎯 Usage](#-usage)
- [📈 Results](#-results)
- [📌 Conclusion](#-conclusion)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

- Full notebook implementation for EDA, preprocessing, training, evaluation.
- Comparison between multiple models: **XGBoost, LightGBM, CatBoost**.
- Automatic encoding and feature dropping based on correlation and performance.
- Uses **F1 Macro Score** as primary evaluation metric.
- Performance was evaluated using a single 80/20 train–test split instead of cross-validation.

---

## 🗂️ Repository Structure

```
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── CS116Project_LoanApprovalPrediciton.ipynb # Main Jupyter notebook
├── requirements.txt # Python Dependencies
└── README.md # Project overview
└── LICENSE
```

---

## 🚀 Pipeline Overview

### 1. Data Preprocessing

- Handling missing values (median, mode).
- Label encoding for categorical features.
- Outlier removal on numeric columns.
- Correlation matrix for feature insights.

### 2. Model Training

- Applied and compared:
  - **XGBoost**
  - **LightGBM**
  - **CatBoost**
- Used train_test_split (80/20) for model evaluation and comparison.
- Metrics:
  - Accuracy
  - F1 Macro

### 3. Feature Selection

- Dropped low-value features based on:
  - Correlation matrix
  - Model interpretability
- Final dropped features:
  - `cb_person_cred_hist_length`
  - `cb_person_default_on_file_encoded`

### 4. Evaluation

- Evaluated each model before and after feature selection.
- Highlighted performance boosts post feature removal.

---

## ⚙️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/paht2005/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction
   ```
2. **(optional) Create environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Open the notebook:**
   ```bash
   jupyter notebook CS116Project_LoanApprovalPrediciton.ipynb
   ```
## 📈 Results
```
| Model    | All Features | After Dropping 2 Features |
| -------- | ------------ | ------------------------- |
| XGBoost  | 94.90%       | 95.83%                    |
| LightGBM | 94.67%       | 94.83%                    |
| CatBoost | 94.70%       | 94.87%                    |
```
## 📌 Conclusion
After performing model comparisons and testing with feature elimination, we found that removing ``cb_person_cred_hist_length`` and ``cb_person_default_on_file_encoded`` led to consistent performance improvements across all models.

Among them, **XGBoost** achieved the **best result** with:
- **F1 Macro Score: 95.83%**
- **After dropping 2 low-impact features**

This confirms that **careful feature selection** can have significant positive effects on model performanc

## 🤝 Contributing
Contributions, improvements, or issue reports are welcome. Please open a pull request or submit an issue!

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

