# Student Dropout Prediction in Online Learning Platforms

Capstone project (M.Sc. Data Science, University of Europe for Applied Sciences) comparing 10 machine learning models for early prediction of student dropout, with a focus on class-imbalance handling strategies.

## Summary

Ten ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM, and others) were trained and evaluated using both class-weighting and SMOTE oversampling to handle class imbalance. Logistic Regression trained on SMOTE-balanced data performed best, reaching an F1 score of 87.6% and ROC-AUC of 93.8% on the test set. First-semester approved credits, course enrollment patterns, and tuition payment status were the strongest predictors of dropout.

Full write-up: [`docs/report.pdf`](docs/report.pdf).

## Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — UCI Machine Learning Repository, created by researchers at the Polytechnic Institute of Portalegre, Portugal (Realinho et al.). 4,424 student records, 35 features covering demographics, semester-by-semester academic metrics, and financial indicators.

The dataset is not included in this repo — download it from the link above and place it as `dataset.csv` in the project root to run the notebook.

## Contents

- `Sprint1-3.ipynb` — data loading, cleaning, EDA, preprocessing, and model training/evaluation pipeline
- `dropout_eda.py` — standalone exploratory data analysis script
- `sprint4_dashboard.html` — results dashboard
- `docs/report.pdf` — final project report

## Usage

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook Sprint1-3.ipynb
```
