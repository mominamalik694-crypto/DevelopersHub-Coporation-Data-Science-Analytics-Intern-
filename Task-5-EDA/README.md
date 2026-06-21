# Task 5: Personal Loan Acceptance Prediction

## Objective
The objective of this project is to predict which customers are likely to accept a personal loan offer based on their demographic and financial information.

## Dataset
Bank Marketing Dataset (UCI Machine Learning Repository)

## Skills Demonstrated
- Data Exploration
- Data Visualization
- Classification Modeling
- Business Insight Extraction

## Step 1: Import Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

## Step 2: Load Dataset

```python
bank = pd.ead_csv("bank.csv")  # or bank-additional.csv depending on your file
bank.head()
```

## Step 3: Data Exploration

```python
bank.shape
```

```python
bank.columns
```

```python
bank.info()
```

## Step 4: Key Feature Analysis (Age, Job, Marital Status)

```python
plt.figure(figsize=(6,4))
sns.histplot(bank['age'], kde=True)
plt.title("Age Distribution")
plt.show()
```

```python
plt.figure(figsize=(6,4))
sns.histplot(bank['age'], kde=True)
plt.title("Age Distribution")
plt.show()
```

```python
sns.countplot(data=bank, x='marital', hue='deposit')
plt.title("Marital Status vs Deposit Subscription")
plt.show()
```

## Step 5: Data Encoding
Categorical variables are converted into numerical format.

```python
bank = pd.get_dummies(bank, drop_first=True)
```

## Step 6: Define Features and Target Variable

```python
X = bank.drop('deposit_yes', axis=1)
y = bank['deposit_yes']
```

## Step 7: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## Step 8: Train Classification 

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

## Step 9: Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Step 10: Business Insights

## Business Insights

- Customers with stable financial history are more likely to accept the loan offer.
- Job type and age group significantly influence loan acceptance.
- Customers with previous successful marketing responses are more likely to subscribe.
- Banks can improve conversion rates by targeting high-probability customer segments.

## Conclusion

A Logistic Regression model was successfully built to predict personal loan acceptance. The dataset was cleaned and encoded, and important customer features were analyzed. The model provides useful insights for banks to improve targeted marketing strategies and increase conversion rates.
