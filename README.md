<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>CS116 - Machine Learning with Python</b></h1>

# CS116 Project: Loan Approval Prediction 

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground_S4E10-blue)](https://www.kaggle.com/competitions/playground-series-s4e10)

> This repository contains the full implementation of a machine learning pipeline developed for the Kaggle competition **Loan Approval Prediction (Playground Series - Season 4, Episode 10)**. Our goal is to predict whether a loan will be approved based on applicant information and credit history, leveraging state-of-the-art ML models and careful feature engineering.

---

##  Team Information
| No. | Student ID  | Full Name (Vietnamese) | Position | Github | Email |
|----:|:-----------:|------------------------|----------|--------|-------|
| 1 | 23521570 | Huynh Viet Tien | Leader | [SharkTien](https://encr.pw/SCu2w) | 23521570@gm.uit.edu.vn |
| 2 | 23521143 | Nguyen Cong Phat | Member | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |
| 3 | 23520123 | Nguyen Minh Bao | Member | [baominh5xx2](https://github.com/baominh5xx2) | 23520123@gm.uit.edu.vn |
| 4 | 23520133 | Pham Phu Bao | Member | [itsdabao](https://github.com/itsdabao) | 23520133@gm.uit.edu.vn |

---

##  Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

##  Features
- Full notebook implementation (`CS116_sources_code.ipynb`) with EDA, preprocessing, model training, tuning, and evaluation.
- Extensive **EDA**: outlier analysis, categorical feature distributions, correlation heatmaps.
- Advanced **Feature Engineering**:
  - Created domain-specific features such as `affordability_score` and `available_funds_ratio`:contentReference[oaicite:3]{index=3}.
  - Feature importance analysis using XGBoost + ANOVA to rank features.
- Comparison between multiple models: **Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost**.
- Final **Soft Voting Ensemble (XGB + LGBM + CatBoost)** improved stability and achieved best performance.
- Evaluation metrics: **F1 Macro Score (primary)**, Accuracy, and AUC.

---

##  Repository Structure

```
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── catboost_info/
├── CS116_sources_code.ipynb # Main source notebook
├── CS116_report.pdf # Full project report
├── CS116_Slide.pdf # Presentation slides
├── requirements.txt # Python dependencies
├── LICENSE
└── README.md
```

---

##  Pipeline Overview

### 1. Data Preprocessing
- **Missing Values:** No missing data detected.
- **Outlier Handling:** Removed unrealistic values (e.g., `person_age < 18` or > 80, `person_emp_length = 123`), capped `loan_percent_income > 0.6`, median-imputation for extreme interest rates.
- **Encoding:**  
  - One-hot encoding for nominal features (`home_ownership`, `loan_intent`).  
  - Ordinal encoding for `loan_grade`.  
  - Binary encoding for `default_on_file`.  
- **Scaling:** Applied **MinMaxScaler** on continuous features.

### 2. Feature Engineering
- Created features such as:
  - `affordability_score = income - loan*(1+interest)`
  - `available_funds_ratio = (income - loan) / income`
- Used **XGBoost feature importance** + **ANOVA** to select high-impact features.

### 3. Model Training & Tuning
- Baseline: Logistic Regression (F1 ≈ 0.56).  
- Tried **Random Forest, LightGBM, CatBoost, XGBoost** with **GridSearchCV** and **Optuna** for hyperparameter tuning.  
- 5-fold cross-validation to ensure robustness.

### 4. Ensemble Methods
- **Soft Voting** (XGB + LGBM + CatBoost, weights 0.4/0.3/0.3).  
- **Stacking** also tested but performed slightly worse.  
- Final chosen model: **Soft Voting Ensemble**.

---

##  Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/paht2005/Loan-Approval-Prediction-CourseProjectgit
   cd Loan-Approval-Prediction-CourseProject

   ```
### 2. (optional) Create environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies:
```bash
   pip install -r requirements.txt
   ```
### 4. Open the notebook:
   ```bash
   jupyter notebook CS116_sources_code.ipynb
   ```
--- 
## Results

Our rigorous evaluation demonstrated significant performance across all models, with the **Ensemble Model** achieving the highest scores.

| Model                      |  F1 Score  |  Accuracy  |     AUC    |
| -------------------------- | :--------: | :--------: | :--------: |
| Logistic Regression        |   0.5614   |   0.9007   |      -     |
| Random Forest              |   0.8731   |   0.9404   |      -     |
| LightGBM                   |   0.8832   |   0.9450   |   0.9571   |
| CatBoost                   |   0.8880   |   0.9482   |   0.9582   |
| XGBoost                    |   0.8928   |   0.9518   |   0.9558   |
| **Soft Voting (Ensemble)** | **0.8939** | **0.9520** | **0.9631** |


> Ensemble model achieved the best overall performance with improved stability and robustness.

**Key Highlights:**
- The **Ensemble Model** achieved the **best overall performance**, demonstrating the power of combining diverse models.
- **XGBoost** showed strong individual performance, especially in F1 Score and Accuracy.
- **CatBoost** and **LightGBM** also performed competitively, contributing valuable insights to the ensemble.
---

## Conclusion
This project successfully built a robust ML pipeline for **loan approval prediction**:
- Achieved **F1 = 0.8939, Accuracy = 95.20%, AUC = 0.9631.**
- Demonstrated the importance of **careful preprocessing**, **feature engineering**, and **ensemble methods**.
- The approach can be generalized to other financial risk prediction tasks.

---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

