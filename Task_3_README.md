# Task 3: Customer Churn Prediction (Bank Customers)

## Objective
The objective of this project is to predict whether a bank customer is likely to leave (churn) based on their demographic and account information.

## Dataset
Churn Modelling Dataset

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Skills Demonstrated
- Data Cleaning
- Categorical Encoding
- Classification Modeling
- Feature Importance Analysis

## Step 1: Load Dataset

```python
import pandas as pd

churn = pd.read_csv("Churn_Modelling.csv")

churn.head()
```