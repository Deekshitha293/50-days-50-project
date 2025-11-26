# 🌟 Day 2 — Data Exploration & Visualization (Iris Dataset)

### 📅 Part of: AI 50-Day Challenge  
👩‍💻 **Author:** Deekshitha Bhairav  

---

## 🎯 Objective
- Load a real dataset (Iris) and explore it thoroughly.  
- Understand feature distributions, relationships, and correlations before building a machine learning model.  
- Gain hands-on experience with Python libraries: Pandas, NumPy, Matplotlib, Seaborn.

---

## 🧰 Key Libraries & Purpose

| Library       | Use Case                                         |
| ------------- | ----------------------------------------------- |
| **NumPy**     | Numerical computations (arrays, math ops)      |
| **Pandas**    | Data handling, cleaning, DataFrames            |
| **Matplotlib**| Basic plotting (line, histogram, boxplot)      |
| **Seaborn**   | Advanced visualization (pairplot, heatmap)     |
| **Scikit-learn** | Load datasets (Iris), preprocessing, modeling |

---

## 📌 Quick Reference — Data Exploration

| Function / Method        | Meaning / Use Case                          |
| ------------------------ | ------------------------------------------ |
| `df.head()`              | View first 5 rows                           |
| `df.shape`               | Rows × columns                              |
| `df.dtypes`              | Data types of each column                  |
| `df.describe()`          | Summary statistics (mean, std, min, max)  |
| `df.isnull().sum()`      | Count missing values                        |
| `df.corr()`              | Correlation between numerical features     |

---

## 📌 Key Visualization Tools

| Plot Type       | Purpose                                   | Example Feature      |
| --------------- | ---------------------------------------- | ------------------ |
| **Histogram**   | Distribution of numerical values         | Sepal & Petal sizes |
| **Pairplot**    | Feature relationships & species separation | Scatter + hue      |
| **Heatmap**     | Correlation between features             | Color-coded values |
| **Boxplot**     | Detect outliers & feature spread         | Feature-wise        |

---

## 🔍 Key Observations from Iris Dataset

- **Dataset Shape:** 150 rows × 5 columns  
- **Columns:** sepal length, sepal width, petal length, petal width, target  
- **Data Types:** Float64 for features, Int64 for target  
- **Missing Values:** None (clean dataset)  
- **Summary Statistics:**  
  - Sepal features vary less than petal features  
  - Petal length and width strongly correlated (good for classification)  
- **Visual Insights:**  
  - Setosa is clearly separable from Versicolor & Virginica  
  - Petal measurements are more discriminative than sepal measurements  

---

## 🧾 Quick Concepts for Interviews

- **Data Cleaning:** Always check for missing or inconsistent data before modeling  
- **Summary Statistics:** `.describe()` — mean, median, min, max, std, quartiles  
- **Correlation:** `.corr()` — find linear relationships between features  
- **Visual Patterns:** Pairplots, heatmaps, histograms, and boxplots reveal clusters, trends, and outliers  

---

## 🏁 Takeaways

- Data exploration provides **critical understanding** for feature selection and modeling  
- Visualization helps **communicate insights quickly**  
- Iris dataset is a classic example for **classification tasks** in ML  

---

> 🌷 *“Visualizing your data is the first step toward intelligent modeling.”* — Deekshitha Bhairav
