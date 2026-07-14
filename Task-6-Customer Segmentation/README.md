# Customer Segmentation Using Unsupervised Learning

## 📌 Problem Statement

Businesses often serve customers with very different purchasing behaviors. Identifying groups of similar customers helps businesses design targeted marketing campaigns, improve customer satisfaction, and increase sales.

## 🎯 Objective

Segment customers based on their **annual income** and **spending habits** using **K-Means Clustering**. The resulting segments are visualized using dimensionality reduction (PCA), and a marketing strategy is proposed for each segment.

## 📂 Dataset

- **File:** `customer_segmentation_results.csv`
- **Rows / Columns:** 200 rows × 5 columns
- **Features:**
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## 🛠️ Tools & Libraries

- `pandas`, `numpy` – data handling
- `matplotlib`, `seaborn` – visualization
- `scikit-learn` – `KMeans`, `StandardScaler`, `PCA`, `silhouette_score`

## 🔍 Workflow

1. **Data Loading** – Read the dataset with pandas and preview its structure.
2. **Dataset Overview** – Checked shape, column names, data types, and summary statistics.
3. **Data Cleaning** – Verified there were no missing values or duplicate records.
4. **Exploratory Data Analysis (EDA)**
   - Gender distribution (count plot)
   - Age distribution (histogram + KDE)
   - Annual Income distribution (histogram + KDE)
   - Spending Score distribution (histogram + KDE)
   - Annual Income vs. Spending Score scatter plot, colored by Gender
5. **Feature Selection** – Chose `Annual Income (k$)` and `Spending Score (1-100)` as the clustering features since they best capture purchasing behavior.
6. **Feature Scaling** – Standardized the selected features with `StandardScaler` so both contribute equally to distance calculations.
7. **Optimal Cluster Count (Elbow Method)** – Computed WCSS for k = 1 to 10 and plotted the elbow curve to select the best number of clusters.
8. **K-Means Clustering** – Fit K-Means with **k = 5** clusters (based on the elbow plot) and assigned each customer a cluster label.
9. **Cluster Visualization** – Plotted Annual Income vs. Spending Score, colored by cluster, to visually inspect the segmentation.
10. **Model Evaluation (Silhouette Score)** – Measured how well-separated and cohesive the clusters are.
11. **PCA Visualization** – Reduced the scaled features to 2 principal components for an alternate view of cluster separation.
12. **Marketing Strategy Mapping** – Proposed a targeted marketing approach for each of the 5 clusters.

## 📊 Results

Five distinct customer segments were identified based on income and spending patterns, each representing a different purchasing profile:

| Cluster | Marketing Strategy |
|---|---|
| Cluster 0 | Offer premium membership and exclusive discounts |
| Cluster 1 | Provide loyalty rewards to encourage repeat purchases |
| Cluster 2 | Promote affordable products and seasonal discounts |
| Cluster 3 | Recommend luxury products and personalized services |
| Cluster 4 | Increase engagement through targeted advertisements and special promotions |

Cluster quality was evaluated using the **Silhouette Score**, and PCA was used to visualize the segmentation in two dimensions.

## ✅ Conclusion

This project successfully segmented customers using K-Means clustering based on annual income and spending score. The Elbow Method determined the optimal number of clusters, and the Silhouette Score validated clustering quality. PCA provided an intuitive 2D visualization of the resulting segments. These insights enable businesses to design targeted marketing campaigns, improve customer engagement, and support data-driven decision-making.

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place `customer_segmentation_results.csv` in the same directory as the notebook.
3. Open and run `TASK_1.ipynb` (Customer Segmentation) cell by cell in Jupyter Notebook / JupyterLab.

## 📁 Project Structure

```
├── TASK_1.ipynb                       # Main notebook (Customer Segmentation)
├── customer_segmentation_results.csv  # Dataset
└── README.md                          # Project documentation
```
