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

## Step 2: Data Exploration
We explore the dataset to understand its structure, features, and data types.

```python
churn.shape
```

```python
churn.columns
```

```python
churn.info()
```

```python
churn.head()
```

## Step 3: Remove Irrelevant Columns
Some columns like RowNumber, CustomerId, and Surname do not affect churn prediction.

```python
churn = churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
```

## Step 4: Check Missing Values

```python
churn.isnull().sum()
```

## Step 5: Encode Categorical Variables

```python
churn = pd.get_dummies(churn, columns=['Geography'], drop_first=True)
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
churn['Gender'] = le.fit_transform(churn['Gender'])
```

## Step 6: Define Features and Target Variable

```python
X = churn.drop('Exited', axis=1)
y = churn['Exited']
```

## Step 7: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Step 8: Train Classification Model (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

## Step 9: Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
```

```python
print(confusion_matrix(y_test, y_pred))
```

## Step 10: Feature Importance Analysis
This shows which factors most influence customer churn.

```python
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, feature_importance)
plt.title("Feature Importance in Customer Churn Prediction")
plt.show()
```

## Conclusion

The churn dataset was successfully analyzed and prepared. Categorical variables such as Geography and Gender were encoded, and a Random Forest model was trained to predict customer churn. Feature importance analysis showed which factors influence customer exit behavior the most. The model can help banks identify at-risk customers and improve retention strategies.