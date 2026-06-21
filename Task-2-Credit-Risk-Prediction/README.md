# Task 2: Credit Risk Prediction

## Objective
The objective of this project is to predict whether a loan applicant is likely to default on a loan using machine learning techniques.

## Dataset
Loan Prediction Dataset

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Skills Demonstrated
- Data Cleaning
- Handling Missing Values
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning Classification
- Model Evaluation

## Step 1: Import Required Libraries

The necessary libraries are imported for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
```

## Step 2: Load the Dataset

The dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

```python
loan = pd.read_excel('train.xlsx')
loan.head()
```

## Step 3: Dataset Exploration

The structure of the dataset is examined to understand the available features and records.

```python
loan.shape
```

```python
loan.columns
```

```python
loan.info()
```

```python
loan.head()
```

## Step 4: Handling Missing Values

Missing values can negatively affect model performance. Therefore, missing values are identified and handled appropriately.

```python
loan.isnull().sum()
```

```python
loan['Gender'] = loan['Gender'].fillna('Male')
loan['Married'] = loan['Married'].fillna('Yes')
loan['Dependents'] = loan['Dependents'].fillna('0')
loan['Self_Employed'] = loan['Self_Employed'].fillna('No')

loan['LoanAmount'] = loan['LoanAmount'].fillna(
    loan['LoanAmount'].median()
)

loan['Loan_Amount_Term'] = loan['Loan_Amount_Term'].fillna(
    loan['Loan_Amount_Term'].median()
)

loan['Credit_History'] = loan['Credit_History'].fillna(1)
```

```python
loan.isnull().sum()
```

## Step 5: Exploratory Data Analysis (EDA)

Visualizations are created to understand important features in the dataset.

### Loan Amount Distribution

```python
plt.figure(figsize=(6,4))
sns.histplot(loan['LoanAmount'], kde=True)
plt.title('Loan Amount Distribution')
plt.show()
```

### Education and Loan Approval Status

```python
plt.figure(figsize=(6,4))
sns.countplot(data=loan, x='Education', hue='Loan_Status')
plt.title('Education vs Loan Approval')
plt.show()
```

### Applicant Income and Loan Status

```python
plt.figure(figsize=(6,4))
sns.boxplot(data=loan, x='Loan_Status', y='ApplicantIncome')
plt.title('Income vs Loan Status')
plt.show()
```

## Observations

- Most loan amounts fall within a moderate range.
- Education level appears to influence loan approval outcomes.
- Applicants with higher income generally show better approval rates.
- Credit history is expected to be an important factor in prediction.

## Step 6: Data Preparation

Categorical variables are converted into numerical values for machine learning.

```python
loan = loan.drop('Loan_ID', axis=1)
```

```python
categorical_columns = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

le = LabelEncoder()

for col in categorical_columns:
    loan[col] = le.fit_transform(loan[col].astype(str))
```

## Step 7: Train-Test Split

The dataset is divided into training and testing sets.

```python
X = loan.drop('Loan_Status', axis=1)
y = loan['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## Step 8: Model Training

A Logistic Regression model is trained to predict loan approval status.

```python
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

## Step 9: Model Evaluation

The model is evaluated using Accuracy Score and Confusion Matrix.

```python
y_pred = model.predict(X_test)
```

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

```python
print(confusion_matrix(y_test, y_pred))
```

## Conclusion

The loan prediction dataset was successfully cleaned and analyzed. Missing values were handled, important features were visualized, and a Logistic Regression model was trained. The model's performance was evaluated using accuracy and a confusion matrix, demonstrating how machine learning can be used to support credit risk prediction.
