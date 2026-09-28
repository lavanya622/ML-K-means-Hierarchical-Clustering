# Unsupervised Learning – K-Means & Hierarchical Clustering

## 📌 Overview

This project introduces **Unsupervised Learning** using customer segmentation data.

Unlike supervised learning, unsupervised learning does not use a target/output column. Instead, the algorithms identify hidden patterns, similarities, and groups within the data.

In this practical, two clustering techniques were explored:

* **K-Means Clustering**
* **Hierarchical Clustering**

The dataset contains customer information such as age, annual income, and spending score. The main objective is to group customers based on similar characteristics.

---

## 🌱 Introduction to Unsupervised Learning

**Unsupervised Learning** is a type of Machine Learning where the model learns patterns from data without a predefined target variable.

Some common unsupervised learning techniques are:

* K-Means Clustering
* Hierarchical Clustering
* DBSCAN
* PCA

### Main Idea

```text
Input Data
    ↓
Preprocessing
    ↓
Feature Selection
    ↓
Feature Scaling
    ↓
Clustering Algorithm
    ↓
Identify Groups / Patterns
```

---

# 🎯 Project Objective

The objective of this practical is to:

* Understand Unsupervised Learning
* Perform customer segmentation
* Apply K-Means Clustering
* Evaluate K-Means using Silhouette Score
* Understand Hierarchical Clustering
* Plot a Dendrogram
* Identify groups of similar customers

---

# 📊 Dataset

The dataset contains customer information.

### Dataset Columns

| Column                 | Description                |
| ---------------------- | -------------------------- |
| CustomerID             | Unique ID of the customer  |
| Gender                 | Gender of the customer     |
| Age                    | Age of the customer        |
| Annual Income (k$)     | Annual income in thousands |
| Spending Score (1-100) | Customer spending score    |

### Example Data

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
| ---------: | ------ | --: | -----------------: | ---------------------: |
|          1 | Male   |  19 |                 15 |                     39 |
|          2 | Male   |  21 |                 15 |                     81 |
|          3 | Female |  20 |                 16 |                      6 |
|          4 | Female |  23 |                 16 |                     77 |
|          5 | Female |  31 |                 17 |                     40 |

---

# 🔹 1. K-Means Clustering

## What is K-Means?

**K-Means Clustering** is an unsupervised Machine Learning algorithm used to divide data into a predefined number of clusters.

The value of **K** represents the number of clusters.

For example:

```text
K = 2 → 2 clusters
K = 3 → 3 clusters
K = 5 → 5 clusters
```

K-Means works by assigning data points to the nearest cluster center (**centroid**) and repeatedly updating the centroids.

---

## K-Means Workflow

```text
Choose K
   ↓
Initialize Centroids
   ↓
Assign Data Points to Nearest Centroid
   ↓
Update Centroids
   ↓
Repeat
   ↓
Final Clusters
```

---

# 📈 Silhouette Score

The **Silhouette Score** is used to evaluate how well the data points are grouped into clusters.

The score generally ranges from:

```text
-1 to +1
```

A higher silhouette score generally indicates better-defined clusters.

In this practical, different values of K were tested from **2 to 10**.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_labels)

    silhouette_scores.append(score)

for k, score in zip(range(2, 11), silhouette_scores):
    print("K =", k, "Silhouette Score =", score)
```

The silhouette scores can be compared to help understand the clustering quality for different values of K.

---

# 🔹 2. Hierarchical Clustering

## What is Hierarchical Clustering?

**Hierarchical Clustering** is an unsupervised learning technique that creates a hierarchy of clusters.

It does not require choosing the final number of clusters at the beginning in the same way as K-Means.

The relationships between data points are represented using a **Dendrogram**.

---

# 🌳 Dendrogram

A **Dendrogram** is a tree-like diagram that shows how individual data points or clusters are merged together.

The height of a merge represents the distance between the clusters being joined.

Example:

```text
        ┌───────────────┐
        │               │
    ┌───┴───┐       ┌───┴───┐
    │       │       │       │
   C1      C2      C3      C4
```

A dendrogram can help us understand the hierarchical structure of the data and choose a suitable number of clusters.

---

## Dendrogram Code

```python
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

dendrogram(
    linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.show()
```

---

# 🔄 K-Means vs Hierarchical Clustering

| Feature            | K-Means           | Hierarchical                    |
| ------------------ | ----------------- | ------------------------------- |
| Learning Type      | Unsupervised      | Unsupervised                    |
| Type               | Clustering        | Clustering                      |
| Target Variable    | Not required      | Not required                    |
| Main Concept       | Centroids         | Hierarchy of clusters           |
| Output             | Clusters          | Dendrogram / clusters           |
| Number of Clusters | Usually specified | Can be selected from dendrogram |
| Evaluation         | Silhouette Score  | Dendrogram analysis             |

---

# 🧠 Important Understanding

**K-Means and Hierarchical Clustering are separate algorithms.**

In this project, the same customer dataset can be used to demonstrate both techniques.

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Scaling
       ↓
       ├───────────────┐
       ↓               ↓
   K-Means        Hierarchical
       ↓               ↓
Silhouette        Dendrogram
   Score
```

Therefore, Hierarchical Clustering is **not performed inside K-Means**. It is another clustering method applied to the same unsupervised learning problem.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* SciPy
* Jupyter Notebook / Google Colab

---

# 📚 Concepts Covered

* Unsupervised Learning
* Customer Segmentation
* Clustering
* K-Means Clustering
* Centroids
* Feature Scaling
* Silhouette Score
* Hierarchical Clustering
* Dendrogram
* Cluster Analysis

---

# 🎓 Key Learning

Through this practical, I learned how unsupervised learning can be used to discover hidden patterns in data without a target variable.

I also learned:

* How K-Means creates customer clusters
* Why feature scaling is important for clustering
* How Silhouette Score can be used to evaluate clusters
* How Hierarchical Clustering builds a hierarchy of clusters
* How to visualize hierarchical relationships using a Dendrogram
* The difference between K-Means and Hierarchical Clustering

---

# 📌 Conclusion

This project demonstrates two important Unsupervised Learning techniques: **K-Means Clustering and Hierarchical Clustering**.

Using the customer segmentation dataset, clustering techniques can be used to discover groups of customers with similar characteristics such as age, income, and spending behavior.

This practical provides a foundation for further Unsupervised Learning techniques such as **DBSCAN, PCA, and advanced clustering methods**.

---

