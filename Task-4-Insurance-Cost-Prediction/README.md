# Task 4: Predicting Insurance Claim Amounts

## Objective
The objective of this project is to estimate medical insurance charges based on personal data such as age, BMI, smoking status, and other factors.

## Dataset
Medical Cost Personal Dataset

## Skills Demonstrated
- Regression Modeling
- Feature Analysis
- Data Visualization
- Model Evaluation (MAE, RMSE)

## Step 1: Import Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
```

## Step 2: Load Dataset

```python
insurance = pd.read_csv("Insurance.csv")
insurance.head()
```

## Step 3: Data Exploration

```python
insurance.shape
```

```python
insurance.info()
```

```python
insurance.describe()
```

## Step 4: Encode Categorical Features

```python
insurance = pd.get_dummies(insurance, drop_first=True)
insurance.head()
```

### Age vs Insurance Charges

```python
plt.scatter(insurance['age'], insurance['charges'])
plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()
```

### BMI vs Insurance Charges

```python
plt.scatter(insurance['bmi'], insurance['charges'])
plt.title("BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()
```

### Smoking Status Impact on Charges

```python
sns.boxplot(x='smoker_yes', y='charges', data=insurance)
plt.title("Smoking vs Charges")
plt.show()
```

## Step 6: Train Linear Regression Model

```python
X = insurance.drop('charges', axis=1)
y = insurance['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

## Step 7: Model Evaluation (MAE & RMSE)

```python
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
```

## Conclusion

The insurance dataset was analyzed and a Linear Regression model was trained to predict medical charges. It was observed that age, BMI, and smoking status have a strong impact on insurance costs. The model was evaluated using MAE and RMSE, showing reasonable prediction performance for real-world estimation tasks.
