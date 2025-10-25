# 🌟 Day 2 — Data Exploration and Visualization

### 📅 Part of: AI 50-Day Challenge  
### 👩‍💻 Author: Deekshitha Bhairav  

---

## 🎯 Objective

The goal of Day 2 was to understand how to **explore, clean, and visualize data** before applying any AI or Machine Learning model.  
This step helps us understand the dataset’s structure, relationships, and hidden patterns.

---

## 🧠 Concepts Covered

- **Data Exploration:** Checking how data looks — number of rows, columns, and feature types.  
- **Data Cleaning:** Handling missing values or incorrect data types.  
- **Data Visualization:** Using plots and charts to identify relationships and trends between features.

---

## 🧰 Libraries Used

| Library | Purpose |
|----------|----------|
| **NumPy** | For numerical operations |
| **Pandas** | For reading and analyzing data |
| **Matplotlib** | For basic plotting and charts |
| **Seaborn** | For beautiful and advanced visualizations |
| **Scikit-learn** | To load sample datasets like Iris |

---

## 🪜 Step-by-Step Workflow

### 🧩 Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set(style="whitegrid")
🌸 Step 2: Load the Dataset
We used the Iris dataset, which contains 150 records of flower measurements — sepal length, sepal width, petal length, and petal width.

python
Copy code
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()
✅ Output:
First five rows showing the features and target class.

🔍 Step 3: Explore the Data
python
Copy code
print("Shape of dataset:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nSummary statistics:\n", data.describe())
Observations:

Dataset shape: 150 rows × 5 columns

All columns are numeric

Values are within expected ranges

💧 Step 4: Check for Missing Values
python
Copy code
data.isnull().sum()
✅ Result:
No missing values — dataset is clean.

🎨 Step 5: Data Visualization
1. Histogram (Distribution of Each Feature)
python
Copy code
data.hist(figsize=(10,8), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()
Purpose:
To see how each feature is distributed (spread of data).

2. Pairplot (Feature Relationships)
python
Copy code
sns.pairplot(data, hue='target', palette='bright')
plt.suptitle("Pairplot of Iris Features", y=1.02, fontsize=16)
plt.show()
Purpose:
Shows how features relate to each other.
Observation:
Petal length and width clearly separate the flower species.

3. Correlation Heatmap
python
Copy code
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Iris Features", fontsize=16)
plt.show()
Purpose:
Displays relationships between numerical features.
Observation:
Petal length and petal width are highly correlated (~0.96).

💡 Insights
No missing values found.

Petal measurements are more useful for classifying species.

Sepal measurements overlap across classes.

Dataset is well-structured and ready for model training.

🧾 Interview Notes (Quick Revision)
Question	Short Answer
What is data exploration?	Understanding dataset shape, columns, and values.
Why do we use .describe()?	To get statistical details like mean, std, min, and max.
What is a histogram used for?	To view how data is distributed.
What does a pairplot show?	Relationships between every pair of features.
What is correlation?	It shows how two variables are related (positive or negative).
Why use data visualization?	To easily identify patterns and insights in data.

