# Task 1: Exploring and Visualizing the Iris Dataset

## Objective
The goal of this task is to explore the Iris dataset, understand its structure, and create visualizations to identify patterns, distributions, and outliers using Python.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
iris = sns.load_dataset('iris')
```

```python
print("Shape:", iris.shape)
print("Columns:", iris.columns)
iris.head()
```

```python
plt.figure(figsize=(6,4))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.show()
```

```python
plt.figure(figsize=(6,4))
sns.histplot(data=iris, x='petal_length', bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.show()
```

```python
plt.figure(figsize=(6,4))
sns.boxplot(data=iris, x='species', y='petal_width')
plt.title('Petal Width by Species')
plt.show()
```

# Key Observations
Setosa species has the smallest petal size among all species
Virginica shows the largest petals and sepal measurements
There is clear separation between species in scatter plot
Iris dataset shows strong patterns useful for classification

# Conclusion
The Iris dataset was successfully explored and visualized using Python libraries.
Different species of iris flowers show clear differences in measurements.
Data visualization helps to understand patterns and distributions easily.
This dataset is suitable for classification tasks in machine learning.
