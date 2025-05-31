
# Machine Learning Projects with Real-World Datasets

### This repository contains machine learning projects implemented in **Python using Jupyter Notebooks**. Each project focuses on a different type of machine learning task — **Classification**, **Regression**, and **Clustering** — using clean, real-world datasets.
### Each model is trained using python libraries such as Pandas, Numpy and Matplotlib only. No external libraries such as sklearn were used.
---

## Project Structure

```
.
├── Classification
│   ├── dataset
│   │   ├──bank-full.csv
│   ├── decsionTree.ipynb
│   ├── logisticRegression.ipynb
│   ├── naiveBayes.ipynb
│   └── randomForest.ipynb
│   └── randomForest.ipynb
├── Clustering
│   ├── dataset
│   │   ├──Mall_Customers.csv
│   ├── hierarchicalClustering.ipynb
│   ├── K-Means.ipynb
├── Regression
│   ├── dataset
│   │   ├──tesla-stock-price.csv
│   ├── decsionTree.ipynb
│   ├── KNN.ipynb
│   ├── linearRegression.ipynb
│   └── randomForest.ipynb
│   └── ridgeRegression.ipynb
└── README.md
```

---

## Classification – Bank Marketing Dataset

Predict whether a customer will subscribe to a term deposit based on demographic and marketing-related features.

-  Problem Type: Binary Classification  
-  Algorithms Used: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM etc.
-  Evaluation Metrics: Accuracy, f1-score, Precision, Recall

### Sample Data
| age | job        | marital | education | default | balance | housing | loan | duration | campaign | poutcome | y  |
|-----|------------|---------|-----------|---------|---------|---------|------|----------|----------|----------|----|
| 58  | management | married | tertiary  | no      | 2143    | yes     | no   | 261      | 1        | unknown  | no |
| 44  | technician | single  | secondary | no      | 29      | yes     | no   | 151      | 1        | unknown  | no |

---

## Regression – Tesla Stock Price

Predict future closing stock prices of Tesla using past stock performance data.

-  Problem Type: Regression  
-  Algorithms Used: Linear Regression, Decision Teee Regressor, Random Forest Regressor, Lasso & Ridge Regression, K-Nearest Neighbour
-  Evaluation Metrics: MAE, MSE, R2-score

### Sample Data
| date       | open   | high   | low    | close  | volume   |
|------------|--------|--------|--------|--------|----------|
| 15-10-2018 | 259.06 | 263.28 | 254.53 | 259.59 | 6189026  |
| 12-10-2018 | 261.00 | 261.99 | 252.01 | 258.78 | 7189257  |

---

## Clustering – Mall Customers

Segment customers based on their purchasing behavior using unsupervised learning.

-  Problem Type: Clustering  
-  Algorithms Used: K-Means, Hierarchical Clustering
-  Evaluation: Elbow Method, Silhouette Score, Visualizations

### Sample Data
| CustomerID | Gender | Age | Annual Income (k$) | Spending Score |
|------------|--------|-----|--------------------|----------------|
| 1          | Male   | 19  | 15                 | 39             |
| 2          | Male   | 21  | 15                 | 81             |
| 3          | Female | 20  | 16                 | 6              |

---

## Technologies Used

- Python 3.x
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib
